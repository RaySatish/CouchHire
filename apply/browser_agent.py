"""Semi-autonomous ATS form filler with LLM-assisted field mapping.

Navigates to an ATS application URL, extracts form fields, uses an LLM to
map them to applicant data, fills them in, and handles blockers via a
human-in-the-loop interrupt system via Telegram.

Public API:
    fill_form(url, documents) -> dict  — fill an ATS form, return result

Internal helpers (prefixed with _) are importable for testing.
"""

import json
import logging
import re
import time
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Form answers persistence
# ---------------------------------------------------------------------------

def _load_form_answers() -> dict:
    """Load stored form answers from config.FORM_ANSWERS_PATH.

    Returns an empty dict if the file doesn't exist or is malformed.
    """
    from config import FORM_ANSWERS_PATH

    if not FORM_ANSWERS_PATH.exists():
        return {}
    try:
        data = json.loads(FORM_ANSWERS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
        logger.warning("form_answers.json is not a dict — returning empty")
        return {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load form_answers.json: %s", exc)
        return {}


def _save_form_answer(key: str, value: str) -> None:
    """Save a single answer to the persistent form answers file."""
    from config import FORM_ANSWERS_PATH

    answers = _load_form_answers()
    answers[key] = value

    # Ensure parent directory exists
    FORM_ANSWERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FORM_ANSWERS_PATH.write_text(
        json.dumps(answers, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.debug("Saved form answer: %s = %s", key, value)


# ---------------------------------------------------------------------------
# Fuzzy select option matching
# ---------------------------------------------------------------------------

def _fuzzy_match_option(target: str, options: list[dict]) -> dict | None:
    """Find the best matching <option> for a target value.

    Handles cases like "India" matching "India (+91)" or "IN - India".

    Args:
        target: The value the LLM wants to select (e.g., "India").
        options: List of dicts with "value" and "text" keys from the <select>.

    Returns:
        The best matching option dict, or None if no reasonable match.
    """
    target_lower = target.lower().strip()

    # 1. Exact case-insensitive match
    for opt in options:
        if opt["text"].lower().strip() == target_lower:
            return opt

    # 2. Target is a substring of an option
    substring_matches = []
    for opt in options:
        if target_lower in opt["text"].lower():
            substring_matches.append(opt)
    if len(substring_matches) == 1:
        return substring_matches[0]
    # If multiple substring matches, pick the shortest (most specific)
    if substring_matches:
        return min(substring_matches, key=lambda o: len(o["text"]))

    # 3. Option text starts with target
    for opt in options:
        if opt["text"].lower().strip().startswith(target_lower):
            return opt

    # 4. Word overlap scoring
    target_words = set(target_lower.split())
    best_score = 0
    best_opt = None
    for opt in options:
        opt_words = set(opt["text"].lower().split())
        overlap = len(target_words & opt_words)
        score = overlap / max(len(target_words), 1)
        if score > best_score:
            best_score = score
            best_opt = opt

    if best_score >= 0.5 and best_opt:
        return best_opt

    return None

# ---------------------------------------------------------------------------
# Resume-upload-first strategy (Fix 1)
# ---------------------------------------------------------------------------


# Track selectors that have failed to fill — avoids infinite retry loops
_failed_selectors: dict[str, int] = {}  # selector -> failure count
_asked_fields: set[str] = set()  # field labels already asked via Telegram (prevents re-asking in retry cycles)

def _try_resume_upload_first(page, resume_path: str) -> bool:
    """Attempt to upload resume before filling any fields.

    Many ATS forms auto-populate name, email, phone, address, education,
    and work history after a resume is uploaded. Doing this first dramatically
    reduces the number of fields we need to fill manually.

    Returns True if a resume was successfully uploaded.
    """
    try:
        # Strategy 1: Find a visible file input that accepts documents
        file_inputs = page.locator("input[type='file']").all()

        for fi in file_inputs:
            try:
                # Check if this file input is for resume/CV (not profile photo etc.)
                accept = fi.get_attribute("accept") or ""
                input_id = fi.get_attribute("id") or ""
                input_name = fi.get_attribute("name") or ""

                # Get nearby label text
                label_text = ""
                try:
                    label_text = fi.evaluate("""el => {
                        // Check label[for]
                        if (el.id) {
                            const lbl = document.querySelector('label[for="' + el.id + '"]');
                            if (lbl) return lbl.textContent.trim();
                        }
                        // Check parent text
                        const parent = el.closest('[class*="upload"], [class*="resume"], [class*="file"], [class*="attach"]');
                        if (parent) return parent.textContent.trim().substring(0, 100);
                        return '';
                    }""")
                except Exception:
                    pass

                context_text = (accept + " " + input_id + " " + input_name + " " + label_text).lower()

                # Skip if this is clearly for a profile photo
                if any(word in context_text for word in ["photo", "avatar", "picture", "headshot", "image/png", "image/jpeg"]):
                    if "resume" not in context_text and "cv" not in context_text:
                        continue

                # Accept if: no accept filter, or accepts PDFs/docs, or context mentions resume
                is_resume_input = (
                    not accept  # no filter = probably general upload
                    or any(ext in accept.lower() for ext in [".pdf", ".doc", ".docx", "application/pdf"])
                    or any(word in context_text for word in ["resume", "cv", "document"])
                )

                if is_resume_input:
                    fi.set_input_files(resume_path)
                    logger.info("Resume uploaded via file input: id=%s, name=%s", input_id, input_name)
                    return True

            except Exception as exc:
                logger.debug("Skipping file input: %s", exc)
                continue

        # Strategy 2: Look for "Upload Resume" / "Attach Resume" buttons that trigger a hidden file input
        upload_buttons = page.locator(
            "button:has-text('Upload'), button:has-text('Attach'), "
            "a:has-text('Upload Resume'), a:has-text('Attach Resume'), "
            "[class*='upload']:has-text('Resume'), [class*='upload']:has-text('CV')"
        ).all()

        for btn in upload_buttons:
            try:
                if btn.is_visible(timeout=1000):
                    # Clicking this button usually triggers a hidden file input
                    # Set up a file chooser handler
                    with page.expect_file_chooser(timeout=3000) as fc_info:
                        btn.click()
                    file_chooser = fc_info.value
                    file_chooser.set_files(resume_path)
                    logger.info("Resume uploaded via button click + file chooser")
                    return True
            except Exception:
                continue

        logger.info("No resume upload element found on page — proceeding with manual fill")
        return False

    except Exception as exc:
        logger.warning("Resume upload attempt failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Resume upload gate detection (Fix: Phenom/SuccessFactors mandatory upload)
# ---------------------------------------------------------------------------

def _detect_resume_upload_gate(page) -> bool:
    """Detect if the current page requires a mandatory resume upload before proceeding.

    Many ATS (Phenom, SuccessFactors, Workday) show a page with
    "Upload Resume" / "Apply With LinkedIn" as the first step, and
    the form fields only appear AFTER a resume is uploaded.

    Returns True if a resume upload gate is detected.
    """
    try:
        return page.evaluate("""
        () => {
            const bodyText = document.body.innerText.toLowerCase();

            // Pattern 1: Prominent "Upload Resume" button visible
            const uploadBtns = document.querySelectorAll(
                'button, a, [role="button"], [class*="upload"], [class*="btn"]'
            );
            let hasUploadResumeBtn = false;
            for (const btn of uploadBtns) {
                const rect = btn.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) continue;
                const txt = (btn.textContent || '').trim().toLowerCase();
                if (txt.match(/upload\\s*(resume|cv|your resume|your cv)/i) ||
                    txt.match(/attach\\s*(resume|cv)/i)) {
                    hasUploadResumeBtn = true;
                    break;
                }
            }

            // Pattern 2: Page text mentions "Resume*" (required) with upload context
            const hasResumeRequired = bodyText.includes('resume*') ||
                bodyText.match(/resume\\s+is\\s+required/) ||
                bodyText.match(/please\\s+upload.*resume/) ||
                bodyText.match(/upload.*resume.*to\\s+continue/) ||
                bodyText.match(/submit.*resume.*or.*importing/);

            // Pattern 3: Very few form inputs visible (upload gate pages have minimal inputs)
            const formInputs = document.querySelectorAll(
                'input:not([type="hidden"]):not([type="file"]), select, textarea'
            );
            let visibleInputs = 0;
            for (const inp of formInputs) {
                const rect = inp.getBoundingClientRect();
                if (rect.width > 0 && rect.height > 0) visibleInputs++;
            }

            // Pattern 4: "Apply With LinkedIn" button present (common on upload gate pages)
            let hasLinkedInApply = false;
            for (const btn of uploadBtns) {
                const txt = (btn.textContent || '').trim().toLowerCase();
                if (txt.includes('linkedin') && (txt.includes('apply') || txt.includes('import'))) {
                    hasLinkedInApply = true;
                    break;
                }
            }

            // Decision: upload gate if we see upload resume button + resume required text,
            // OR upload resume button + LinkedIn apply + few inputs
            if (hasUploadResumeBtn && hasResumeRequired) return true;
            if (hasUploadResumeBtn && hasLinkedInApply && visibleInputs < 5) return true;
            if (hasResumeRequired && visibleInputs < 3) return true;

            return false;
        }
        """)
    except Exception as exc:
        logger.debug("Resume upload gate detection failed: %s", exc)
        return False


def _handle_resume_upload_gate(page, session_id: str, resume_path: str) -> bool:
    """Handle a mandatory resume upload gate via Telegram interrupt.

    Takes a screenshot, sends it to Telegram with instructions for the user
    to upload the resume via CDP browser takeover, then waits for Done.

    Returns True if the user confirmed they uploaded the resume.
    """
    from bot.telegram_bot import ask_yes_no, send_photo, send_notification
    from apply.session_handoff import get_takeover_instructions
    from config import FORM_ANSWERS_PATH

    # Take screenshot
    output_dir = Path(FORM_ANSWERS_PATH).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    screenshot_path = str(output_dir / f"resume_upload_{timestamp}.png")
    try:
        page.screenshot(path=screenshot_path)
    except Exception:
        screenshot_path = None

    # Get CDP takeover instructions
    takeover = get_takeover_instructions(session_id)

    msg = (
        "📎 <b>Resume Upload Required</b>\n\n"
        "The application form requires you to upload a resume before proceeding.\n\n"
        f"📄 Your tailored resume is at:\n<code>{resume_path}</code>\n\n"
        "Please:\n"
        "1. Connect to the browser via chrome://inspect → localhost:9222\n"
        "2. Upload the resume file\n"
        "3. Tap ✅ Done below when uploaded\n\n"
        f"{takeover}"
    )

    logger.info("Resume upload gate detected — sending Telegram interrupt")
    # Send screenshot first (if available), then ask yes/no
    if screenshot_path:
        send_photo(screenshot_path, caption="Resume upload required — see form state above")
    send_notification(msg)
    confirmed = ask_yes_no("Have you uploaded the resume? Tap ✅ when done.")

    if confirmed:
        logger.info("User confirmed resume upload — waiting for page to settle")
        page.wait_for_timeout(4000)  # Wait for ATS to parse the uploaded resume
        return True
    else:
        logger.warning("User did not confirm resume upload")
        return False
# ---------------------------------------------------------------------------
# LinkedIn autofill detection (Fix 2)
# ---------------------------------------------------------------------------

def _try_linkedin_autofill(page, session_id: str) -> bool:
    """Detect and offer LinkedIn autofill to the user.

    If an 'Apply with LinkedIn' or 'Autofill with LinkedIn' button is found,
    notify the user via Telegram and give them the option to complete it manually.

    Returns True if LinkedIn autofill was used.
    """
    try:
        # Skip detection on LinkedIn.com itself — every element there contains "LinkedIn"
        current_url = page.url.lower()
        if "linkedin.com" in current_url:
            logger.debug("Skipping LinkedIn autofill detection — already on linkedin.com")
            return False

        # Use more specific selectors to avoid false positives on generic pages
        linkedin_btn = page.locator(
            "button:has-text('Apply with LinkedIn'), "
            "button:has-text('Autofill with LinkedIn'), "
            "button:has-text('Sign in with LinkedIn'), "
            "a:has-text('Apply with LinkedIn'), "
            "a:has-text('Autofill with LinkedIn'), "
            "[data-provider='linkedin']:visible"
        ).first

        if linkedin_btn.is_visible(timeout=2000):
            from config import FORM_ANSWERS_PATH
            from bot.telegram_bot import ask_yes_no, send_photo, send_notification

            # Take screenshot showing the LinkedIn button
            timestamp = int(time.time())
            output_dir = Path(FORM_ANSWERS_PATH).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            ss_path = str(output_dir / f"linkedin_{timestamp}.png")
            try:
                page.screenshot(path=ss_path)
            except Exception:
                ss_path = ""

            use_linkedin = ask_yes_no(
                "🔗 <b>LinkedIn Autofill available!</b>\n\n"
                "This form has an 'Apply with LinkedIn' option that can auto-fill most fields.\n"
                "Would you like to complete the LinkedIn login manually?\n\n"
                "Tap ✅ to use LinkedIn (I'll wait), or ❌ to skip and fill manually."
            )

            if use_linkedin:
                linkedin_btn.click()
                # Wait for user to complete LinkedIn OAuth
                ask_yes_no("Complete the LinkedIn login in the browser. Tap ✅ when done.")
                page.wait_for_timeout(3000)  # wait for form to populate
                logger.info("LinkedIn autofill completed by user")
                return True
            else:
                logger.info("User skipped LinkedIn autofill")
                return False

    except Exception as exc:
        logger.debug("No LinkedIn autofill button found: %s", exc)
        return False




# ---------------------------------------------------------------------------
# Cookie/consent banner auto-dismiss (Fix — pre-form navigation)
# ---------------------------------------------------------------------------

def _dismiss_cookie_banners(page) -> bool:
    """Auto-dismiss cookie consent banners without bothering the user.

    These are ubiquitous on ATS sites and should never trigger a manual takeover.
    Idempotent — safe to call multiple times.
    Returns True if a banner was found and dismissed.
    """
    # Common cookie consent accept-button selectors
    # Covers OneTrust, CookieBot, Osano, Phenom, and generic patterns.
    ACCEPT_SELECTORS = [
        # OneTrust (Workday, many enterprise ATS)
        "#onetrust-accept-btn-handler",
        # CookieBot
        "#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll",
        # Osano
        ".osano-cm-accept-all",
        # Generic button text patterns
        "button:has-text('Accept All')",
        "button:has-text('Accept Cookies')",
        "button:has-text('Accept all cookies')",
        "button:has-text('I Accept')",
        "button:has-text('I Agree')",
        "button:has-text('Allow All')",
        "button:has-text('Allow all')",
        "button:has-text('Got it')",
        "button:has-text('OK')",
        "a:has-text('Accept All')",
        "a:has-text('Accept Cookies')",
        # Phenom / GE Aerospace style
        "[class*='cookie'] button:has-text('Accept')",
        "[class*='consent'] button:has-text('Accept')",
        "[class*='cookie'] button:has-text('Allow')",
        "[class*='gdpr'] button:has-text('Accept')",
        "[id*='cookie'] button:has-text('Accept')",
        # Banner dismiss/close buttons
        "[class*='cookie'] [class*='close']",
        "[class*='consent'] [class*='close']",
    ]

    BANNER_CONTAINER_SELECTORS = [
        "#onetrust-banner-sdk",
        "#CybotCookiebotDialog",
        "[class*='cookie-banner']",
        "[class*='cookie-consent']",
        "[class*='consent-banner']",
        "[id*='cookie-banner']",
        "[id*='cookie-consent']",
    ]

    for selector in ACCEPT_SELECTORS:
        try:
            btn = page.locator(selector).first
            if btn.is_visible(timeout=500):
                btn.click(timeout=2000)
                logger.info("Dismissed cookie banner via: %s", selector)
                page.wait_for_timeout(500)
                return True
        except Exception:
            continue

    # Fallback: if a banner container is visible, click the first visible button in it
    for container_sel in BANNER_CONTAINER_SELECTORS:
        try:
            container = page.locator(container_sel).first
            if container.is_visible(timeout=500):
                buttons = container.locator("button").all()
                for btn in buttons:
                    try:
                        if btn.is_visible(timeout=300):
                            btn.click(timeout=2000)
                            logger.info("Dismissed cookie banner via button in: %s", container_sel)
                            page.wait_for_timeout(500)
                            return True
                    except Exception:
                        continue
        except Exception:
            continue

    return False




def _dismiss_chatbots(page) -> bool:
    """Dismiss or minimize chatbot overlays that block page interaction.

    Phenom (athenahealth, etc.), Drift, Intercom, and other chatbot widgets
    often open automatically and intercept pointer events on Apply buttons.
    Must be called before attempting to click Apply.
    Returns True if a chatbot was dismissed.
    """
    dismissed = False

    # --- Phenom chatbot (most common on Phenom ATS sites) ---
    # The chatbot wrapper has data-chatbotstate="window-open" when expanded.
    # Strategy: close it via the minimize/close button, or hide it via JS.
    CHATBOT_CLOSE_SELECTORS = [
        # Phenom chatbot close/minimize buttons
        "#phenomChatbotWrapper button[aria-label='Close']",
        "#phenomChatbotWrapper button[aria-label='Minimize']",
        "#phenomChatbotWrapper [class*='close']",
        "#phenomChatbotWrapper [class*='minimize']",
        # Drift
        "#drift-widget-container iframe",
        "button[aria-label='Close chat']",
        # Intercom
        "[class*='intercom'] [class*='close']",
        # Generic chatbot close
        "[class*='chatbot'] button[aria-label='Close']",
        "[class*='chatbot'] button[aria-label='Minimize']",
        "[class*='chatbot'] [class*='close']",
        "[class*='chat-widget'] [class*='close']",
    ]

    for selector in CHATBOT_CLOSE_SELECTORS:
        try:
            btn = page.locator(selector).first
            if btn.is_visible(timeout=500):
                btn.click(timeout=2000)
                logger.info("Dismissed chatbot via: %s", selector)
                page.wait_for_timeout(500)
                dismissed = True
                break
        except Exception:
            continue

    # Fallback: hide the Phenom chatbot wrapper via JavaScript
    if not dismissed:
        try:
            hidden = page.evaluate("""() => {
                const wrapper = document.getElementById('phenomChatbotWrapper');
                if (wrapper) {
                    wrapper.style.display = 'none';
                    return true;
                }
                // Generic chatbot containers
                const chatbots = document.querySelectorAll(
                    '[class*="chatbot-wrapper"], [class*="chat-widget"], ' +
                    '[id*="chatbot"], [id*="drift-widget"]'
                );
                for (const el of chatbots) {
                    el.style.display = 'none';
                    return true;
                }
                return false;
            }""")
            if hidden:
                logger.info("Dismissed chatbot via JavaScript (display:none)")
                page.wait_for_timeout(300)
                dismissed = True
        except Exception:
            pass

    return dismissed

# ---------------------------------------------------------------------------
# Navigate from JD page to application form (Fix — pre-form navigation)
# ---------------------------------------------------------------------------


def _count_form_inputs(page) -> int:
    """Count visible form inputs, excluding chatbot/nav/notification elements."""
    try:
        return page.evaluate("""() => {
            const EXCLUDE = [
                '#phenomChatbotWrapper', '[class*="chatbot"]', '[class*="chat-widget"]',
                '#drift-widget-container', '[class*="intercom"]', '[class*="job-alert"]',
                '[class*="notify"]', '[class*="similar-jobs"]', 'nav', 'header'
            ];
            function isExcluded(el) {
                for (const sel of EXCLUDE) {
                    if (el.closest(sel)) return true;
                }
                return false;
            }
            let count = 0;
            document.querySelectorAll('input, select, textarea').forEach(el => {
                if (el.type === 'hidden') return;
                const rect = el.getBoundingClientRect();
                if (rect.width === 0 && rect.height === 0) return;
                if (isExcluded(el)) return;
                count++;
            });
            return count;
        }""")
    except Exception:
        return 0


def _navigate_to_application_form(page) -> bool:
    """Detect if we're on a job description page and click through to the application form.

    Many ATS URLs (Workday, Phenom, Greenhouse, Lever) land on a JD page first.
    The actual form only appears after clicking "Apply Now" / "Apply" / "Start Application".

    Returns True if we navigated to a new page (form should now be visible).
    Returns False if we're already on the form or no Apply button was found.
    """
    # Check if we're already on a form page (has visible input fields).
    # BUT: many JD pages have incidental inputs (search bar, email signup,
    # cookie consent) that can reach 3+.  Only short-circuit when there are
    # enough inputs AND no visible "Apply" button on the page.
    _QUICK_APPLY_CHECK = [
        "button:has-text('Apply Now')",
        "button:has-text('Apply')",
        "a:has-text('Apply Now')",
        "a:has-text('Apply')",
        "div[role='button']:has-text('Apply')",
    ]
    try:
        form_inputs = _count_form_inputs(page)
        if form_inputs >= 5:
            # 5+ inputs is very likely a real form — skip Apply search
            logger.info(
                "Already on form page (%d inputs found) — skipping Apply button search",
                form_inputs,
            )
            return False
        if form_inputs >= 3:
            # 3-4 inputs is ambiguous — check if an Apply button is also visible.
            # If an Apply button exists, these inputs are NOT the application form
            # (they're incidental: search bar, email signup, etc.).
            has_apply_btn = False
            for sel in _QUICK_APPLY_CHECK:
                try:
                    if page.locator(sel).first.is_visible(timeout=500):
                        has_apply_btn = True
                        break
                except Exception:
                    continue
            if not has_apply_btn:
                logger.info(
                    "Already on form page (%d inputs, no Apply button) "
                    "— skipping Apply button search",
                    form_inputs,
                )
                return False
            else:
                logger.info(
                    "Page has %d inputs BUT an Apply button is visible "
                    "— this is a JD page, not the form. Will click Apply.",
                    form_inputs,
                )
    except Exception:
        pass

    # Common "Apply" button selectors across ATS platforms.
    # IMPORTANT: buttons first — Phenom, Workday, GE Aerospace render "Apply" as <button>.
    APPLY_BUTTON_SELECTORS = [
        # --- BUTTONS first (most common on modern ATS) ---
        "button:has-text('Apply Now')",
        "button:has-text('Apply')",
        "button:has-text('Apply for this job')",
        "button:has-text('Apply for this position')",
        "button:has-text('Start Application')",
        "button:has-text('Begin Application')",
        "button:has-text('Apply Online')",
        # --- role="button" divs ---
        "div[role='button']:has-text('Apply Now')",
        "div[role='button']:has-text('Apply')",
        # --- Links as fallback (Greenhouse, Lever) ---
        "a:has-text('Apply Now')",
        "a:has-text('Apply')",
        "a:has-text('Apply for this job')",
        "a:has-text('Apply for this Job')",
        "a:has-text('Start Application')",
        # --- ATS-specific selectors ---
        "#apply_button",
        ".postings-btn",
        ".postings-btn-wrapper a",
        ".iCIMS_MainWrapper a:has-text('Apply')",
        # --- Attribute-based fallbacks ---
        "[class*='apply'] button",
        "[class*='apply'] a",
        "[id*='apply'] button",
        "[id*='apply'] a",
        "[data-automation*='apply']",
        "[data-testid*='apply']",
    ]

    # Allowed button text (case-insensitive). Only click if text matches one of these.
    ALLOWED_APPLY_TEXT = {
        "apply", "apply now", "start application", "begin application",
        "apply for this job", "apply for this position", "apply online",
    }

    original_url = page.url

    # Dismiss chatbot overlays that may intercept clicks on Apply button
    _dismiss_chatbots(page)

    # Capture input count BEFORE clicking Apply — used to detect real form appearance
    _pre_click_inputs = _count_form_inputs(page)

    for selector in APPLY_BUTTON_SELECTORS:
        try:
            btn = page.locator(selector).first
            if btn.is_visible(timeout=1000):
                btn_text = ""
                try:
                    btn_text = btn.text_content(timeout=1000) or ""
                except Exception:
                    pass

                btn_text_lower = btn_text.strip().lower()

                # For generic selectors (#apply_button, [class*='apply'] etc.)
                # accept any visible match. For text-based selectors, validate text.
                if btn_text_lower and btn_text_lower not in ALLOWED_APPLY_TEXT:
                    # Text doesn't match expected apply-button text — skip
                    continue

                logger.info(
                    "Found Apply button: '%s' via selector '%s'",
                    btn_text.strip(), selector,
                )

                # Handle links that open in new tabs (target="_blank")
                # by listening for popup events during the click.
                # NOTE: btn.click() inside expect_popup always executes —
                # if no popup appears, expect_popup raises TimeoutError
                # but the click already happened. Do NOT click again.
                new_page = None
                try:
                    with page.expect_popup(timeout=5000) as popup_info:
                        btn.click(timeout=5000)
                    new_page = popup_info.value
                    logger.info("Apply button opened new tab: %s", new_page.url)
                except Exception:
                    # No popup — click already happened on same page
                    # (normal navigation or SPA transition)
                    pass

                if new_page:
                    # Switch to the new tab — this is the application form
                    try:
                        new_page.wait_for_load_state("domcontentloaded", timeout=10000)
                    except Exception:
                        pass
                    try:
                        new_page.wait_for_load_state("networkidle", timeout=5000)
                    except Exception:
                        pass
                    _dismiss_cookie_banners(new_page)
                    # Store new_page reference for caller to pick up
                    page._apply_new_page = new_page
                    logger.info("Navigated to application form (new tab): %s", new_page.url)
                    return True

                # Wait for navigation or page content change (same-tab navigation)
                # Phenom and other SPA-based ATS can take 7-15s to transition.
                # Poll for URL change or form appearance over 15 seconds.
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=10000)
                except Exception:
                    pass
                try:
                    page.wait_for_load_state("networkidle", timeout=5000)
                except Exception:
                    pass

                # Poll for URL change or form fields appearing (up to 15s)
                _nav_detected = False
                for _poll in range(6):  # 6 x 2.5s = 15s max
                    new_url = page.url
                    if new_url != original_url:
                        logger.info("Navigated to application form: %s", new_url)
                        _dismiss_cookie_banners(page)
                        _nav_detected = True
                        break

                    # Check for NEW form fields (not requiring <form> ancestor).
                    # Compare against pre-click count — the page may already
                    # have incidental inputs (search bar, email signup, chatbot).
                    try:
                        form_inputs = _count_form_inputs(page)
                        _new_inputs = form_inputs - _pre_click_inputs
                        if form_inputs >= 5 and _new_inputs >= 2:
                            logger.info(
                                "Form appeared after clicking Apply "
                                "(SPA transition, %d inputs [+%d new], poll %d)",
                                form_inputs, _new_inputs, _poll + 1,
                            )
                            _nav_detected = True
                            break
                        elif form_inputs >= 3 and _new_inputs >= 2:
                            logger.info(
                                "Form appeared after clicking Apply "
                                "(SPA transition, %d inputs [+%d new], poll %d)",
                                form_inputs, _new_inputs, _poll + 1,
                            )
                            _nav_detected = True
                            break
                    except Exception:
                        pass

                    # Check for modal/overlay with form
                    try:
                        modal_inputs = page.locator(
                            "[role='dialog'] input, [class*='modal'] input, "
                            "[class*='overlay'] input"
                        ).count()
                        if modal_inputs >= 2:
                            logger.info(
                                "Application form appeared in modal (%d inputs)",
                                modal_inputs,
                            )
                            _nav_detected = True
                            break
                    except Exception:
                        pass

                    page.wait_for_timeout(2500)

                if _nav_detected:
                    return True

                logger.info("Clicked Apply but no form appeared after 15s — continuing search")

        except Exception:
            continue

    logger.info("No Apply button found — assuming we're already on the form")
    return False

# ---------------------------------------------------------------------------
# Build applicant data
# ---------------------------------------------------------------------------

def _build_applicant_data(documents: dict) -> dict:
    """Build a flat dict of all applicant info for the LLM prompt.

    Merges the documents dict (from pipeline state) with any stored answers
    from form_answers.json. Documents take priority over stored answers for
    overlapping keys.
    """
    # Start with stored answers as the base
    data = _load_form_answers()

    # Map documents dict keys to the canonical flat keys
    key_mapping = {
        "applicant_name": "full_name",
        "applicant_email": "email",
        "phone": "phone",
        "resume_pdf_path": "resume_path",
        "cover_letter_text": "cover_letter",
        "github_url": "github",
        "linkedin_url": "linkedin",
    }

    for doc_key, data_key in key_mapping.items():
        value = documents.get(doc_key)
        if value is not None:
            data[data_key] = value

    return data


# ---------------------------------------------------------------------------
# Extract form context (the critical two-part design)
# ---------------------------------------------------------------------------

_EXTRACT_FIELDS_JS = """
() => {
    const fields = [];

    // Detect Google Forms — used by getLabel and buildSelector
    const _isGoogleForms = !!(
        location.hostname === 'docs.google.com' ||
        document.querySelector('.freebirdFormviewerViewFormCard') ||
        (document.querySelector('[data-params]') && location.href.includes('/forms/'))
    );

    function _getGoogleFormsLabel(el) {
        // Google Forms: question text is NOT on the input element.
        // Walk up to the question container and find the heading text.
        // Containers: [data-params], [role="listitem"], [data-item-id], or freebird classes
        const container = el.closest('[data-params]') || el.closest('[data-item-id]')
                       || el.closest('[role="listitem"]')
                       || el.closest('.freebirdFormviewerComponentsQuestionBaseRoot')
                       || el.closest('.Qr7Oae');  // newer Google Forms question container
        if (!container) return '';

        // Primary: look for the question title element (multiple class names across versions)
        const titleEl = container.querySelector('.M7eMe')          // classic
                     || container.querySelector('[role="heading"]') // accessible heading
                     || container.querySelector('.freebirdFormviewerComponentsQuestionBaseTitle')
                     || container.querySelector('.exportItemTitle')
                     || container.querySelector('.Qr7Oae > div > div > span') // newer layout
                     || container.querySelector('[data-initial-value][aria-label]');  // section headers
        if (titleEl) {
            const text = titleEl.textContent.trim();
            if (text && text !== 'Your answer' && text.length < 200) return text;
        }

        // Try aria-describedby on the input itself (Google sometimes sets this)
        const describedBy = el.getAttribute('aria-describedby');
        if (describedBy) {
            const descEl = document.getElementById(describedBy);
            if (descEl) {
                const t = descEl.textContent.trim();
                if (t && t !== 'Your answer' && t.length < 200) return t;
            }
        }

        // Try aria-labelledby on the input (Google Forms often uses this)
        const labelledBy = el.getAttribute('aria-labelledby');
        if (labelledBy) {
            // aria-labelledby can have multiple IDs separated by spaces
            const ids = labelledBy.split(/\\s+/);
            for (const id of ids) {
                const lbl = document.getElementById(id);
                if (lbl) {
                    const t = lbl.textContent.trim();
                    if (t && t !== 'Your answer' && t !== '*' && t.length > 2 && t.length < 200) return t;
                }
            }
        }

        // Fallback: first substantial text node in the container that isn't "Your answer"
        const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, null, false);
        let node;
        while (node = walker.nextNode()) {
            const t = node.textContent.trim();
            if (t && t.length > 2 && t.length < 200 && t !== 'Your answer'
                && t !== '*' && !t.startsWith('http') && t !== 'Required') {
                return t;
            }
        }
        return '';
    }

    function getLabel(el) {
        // Google Forms: skip standard strategies (they return useless "Your answer")
        if (_isGoogleForms) {
            const gLabel = _getGoogleFormsLabel(el);
            if (gLabel) return gLabel;
            // Still try aria-labelledby and placeholder as final fallbacks
        }

        // 1. <label for="id">
        if (el.id) {
            const label = document.querySelector('label[for="' + el.id + '"]');
            if (label && label.textContent.trim()) return label.textContent.trim();
        }
        // 2. aria-label (skip generic "Your answer" from Google Forms)
        const ariaLabel = el.getAttribute('aria-label');
        if (ariaLabel && ariaLabel.trim() !== 'Your answer') return ariaLabel.trim();
        // 3. aria-labelledby
        const labelledBy = el.getAttribute('aria-labelledby');
        if (labelledBy) {
            const labelEl = document.getElementById(labelledBy);
            if (labelEl && labelEl.textContent.trim()) return labelEl.textContent.trim();
        }
        // 4. placeholder
        if (el.placeholder) return el.placeholder.trim();
        // 5. Closest parent label
        const parentLabel = el.closest('label');
        if (parentLabel) {
            const text = parentLabel.textContent.replace(el.textContent || '', '').trim();
            if (text) return text;
        }
        // 6. Preceding sibling or nearby text
        const prev = el.previousElementSibling;
        if (prev && (prev.tagName === 'LABEL' || prev.tagName === 'SPAN' || prev.tagName === 'DIV')) {
            const text = prev.textContent.trim();
            if (text && text.length < 100) return text;
        }
        // 7. name or id as fallback
        if (el.name) return el.name.replace(/[_-]/g, ' ');
        if (el.id) return el.id.replace(/[_-]/g, ' ');
        return '';
    }

    function buildSelector(el) {
        if (el.id) return '#' + CSS.escape(el.id);
        if (el.name) return el.tagName.toLowerCase() + '[name="' + el.name + '"]';

        // Google Forms: inputs have unique class combos and data attributes.
        // Build a selector anchored to the question container.
        if (_isGoogleForms) {
            // Strategy 1: Use aria-labelledby (most reliable on newer Google Forms)
            const ariaLbl = el.getAttribute('aria-labelledby');
            if (ariaLbl) {
                // Use the first ID from aria-labelledby as a unique anchor
                const firstId = ariaLbl.split(/\\s+/)[0];
                if (firstId && document.getElementById(firstId)) {
                    const tag = el.tagName.toLowerCase();
                    // Check if this selector is unique
                    const candidate = '[aria-labelledby*="' + firstId + '"]';
                    const matches = document.querySelectorAll(candidate);
                    if (matches.length === 1) {
                        return candidate;
                    }
                    // Not unique — add tag qualifier
                    const tagCandidate = tag + '[aria-labelledby*="' + firstId + '"]';
                    const tagMatches = document.querySelectorAll(tagCandidate);
                    if (tagMatches.length === 1) {
                        return tagCandidate;
                    }
                }
            }

            // Strategy 2: Use data-params container with question ID
            const container = el.closest('[data-params]');
            if (container) {
                const params = container.getAttribute('data-params');
                const idMatch = params && params.match(/\\[\\s*(\\d{5,})/);
                if (idMatch) {
                    const qId = idMatch[1];
                    const tag = el.tagName.toLowerCase();
                    const sameTag = container.querySelectorAll(tag);
                    if (sameTag.length === 1) {
                        return '[data-params*="' + qId + '"] ' + tag;
                    } else {
                        const idx = Array.from(sameTag).indexOf(el) + 1;
                        return '[data-params*="' + qId + '"] ' + tag + ':nth-of-type(' + idx + ')';
                    }
                }
            }

            // Strategy 3: Use data-item-id container (newer Google Forms)
            const itemContainer = el.closest('[data-item-id]');
            if (itemContainer) {
                const itemId = itemContainer.getAttribute('data-item-id');
                if (itemId) {
                    const tag = el.tagName.toLowerCase();
                    const sameTag = itemContainer.querySelectorAll(tag);
                    if (sameTag.length === 1) {
                        return '[data-item-id="' + itemId + '"] ' + tag;
                    } else {
                        const idx = Array.from(sameTag).indexOf(el) + 1;
                        return '[data-item-id="' + itemId + '"] ' + tag + ':nth-of-type(' + idx + ')';
                    }
                }
            }

            // Strategy 4: Class-based selector (last resort for Google Forms)
            const cls = typeof el.className === 'string' ? el.className.trim() : '';
            if (cls) {
                const primaryClass = cls.split(/\\s+/)[0];
                try {
                    const allWithClass = document.querySelectorAll('.' + CSS.escape(primaryClass));
                    const idx = Array.from(allWithClass).indexOf(el);
                    if (allWithClass.length === 1) {
                        return '.' + CSS.escape(primaryClass);
                    } else if (idx >= 0) {
                        return '.' + CSS.escape(primaryClass) + ':nth-of-type(' + (idx + 1) + ')';
                    }
                } catch(e) {
                    // CSS.escape might fail on unusual class names
                }
            }
        }

        // Standard fallback: positional within parent
        const tag = el.tagName.toLowerCase();
        const parent = el.parentElement;
        if (parent) {
            // Build a more specific parent selector
            let parentSel = '';
            const parentTag = parent.tagName.toLowerCase();
            if (parent.id) {
                parentSel = '#' + CSS.escape(parent.id);
            } else if (typeof parent.className === 'string' && parent.className.trim()) {
                const parentClass = parent.className.trim().split(/\\s+/)[0];
                try {
                    parentSel = parentTag + '.' + CSS.escape(parentClass);
                } catch(e) {
                    parentSel = parentTag;
                }
            } else {
                parentSel = parentTag;
            }

            // SAFETY: never produce "tag > tag" selectors (e.g., input > input)
            // This happens when parent is same element type — use a higher ancestor instead
            if (parentSel === tag || parentSel === parentTag && parentTag === tag) {
                // Walk up to find a suitable non-same-tag ancestor
                let ancestor = parent.parentElement;
                let depth = 0;
                while (ancestor && depth < 5) {
                    const aTag = ancestor.tagName.toLowerCase();
                    if (ancestor.id) {
                        parentSel = '#' + CSS.escape(ancestor.id);
                        break;
                    } else if (typeof ancestor.className === 'string' && ancestor.className.trim()) {
                        const aClass = ancestor.className.trim().split(/\\s+/)[0];
                        try {
                            parentSel = aTag + '.' + CSS.escape(aClass);
                        } catch(e) {
                            parentSel = aTag;
                        }
                        if (parentSel !== tag) break;
                    } else if (aTag !== tag) {
                        parentSel = aTag;
                        break;
                    }
                    ancestor = ancestor.parentElement;
                    depth++;
                }
            }

            const siblings = Array.from(parent.querySelectorAll(':scope > ' + tag));
            const idx = siblings.indexOf(el) + 1;
            if (idx > 0) {
                return parentSel + ' > ' + tag + ':nth-of-type(' + idx + ')';
            }
        }
        return tag;
    }

    function processElement(el, iframeSel) {
        const tag = el.tagName.toLowerCase();
        let type = el.type ? el.type.toLowerCase() : tag;
        // Normalize select-one / select-multiple to 'select' for Python code
        if (type === 'select-one' || type === 'select-multiple') type = 'select';

        // Skip hidden inputs and non-visible elements
        if (type === 'hidden') return null;
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 && rect.height === 0) return null;

        const label = getLabel(el);
        const selector = buildSelector(el);
        const required = el.required || el.getAttribute('aria-required') === 'true';
        const value = el.value || '';

        const field = {
            tag: tag,
            type: type,
            label: label,
            selector: selector,
            required: required,
            value: value,
            iframe: iframeSel || null,
        };

        // Select options
        if (tag === 'select') {
            const options = Array.from(el.options).map(o => o.text.trim()).filter(t => t);
            field.options = options;
        }

        // File accept types
        if (type === 'file' && el.accept) {
            field.accept = el.accept;
        }

        return field;
    }

    // Containers to EXCLUDE from form field extraction.
    // Chatbot widgets, notification signups, and search bars are not application forms.
    const EXCLUDE_ANCESTORS = [
        '#phenomChatbotWrapper',        // Phenom chatbot
        '[class*="chatbot"]',           // Generic chatbot
        '[class*="chat-widget"]',       // Chat widgets
        '#drift-widget-container',      // Drift
        '[class*="intercom"]',          // Intercom
        '[class*="job-alert"]',         // Job alert signup
        '[class*="notify"]',            // "Get notified" sections
        '[class*="similar-jobs"]',      // Similar jobs sidebar
        'nav', 'header',               // Navigation/header search bars
    ];

    function isInsideExcluded(el) {
        for (const sel of EXCLUDE_ANCESTORS) {
            if (el.closest(sel)) return true;
        }
        return false;
    }

    // Process main document
    const selectors = 'input, select, textarea, button[type="submit"], input[type="submit"]';
    document.querySelectorAll(selectors).forEach(el => {
        if (isInsideExcluded(el)) return;  // Skip chatbot/nav/alert fields
        const f = processElement(el, null);
        if (f) fields.push(f);
    });


    // -----------------------------------------------------------------------
    // Google Forms custom widgets (radio groups, checkbox groups, dropdowns)
    // These are NOT standard <input>/<select> elements — they use ARIA roles.
    // -----------------------------------------------------------------------
    if (_isGoogleForms) {
        // Track which question containers we've already processed (via standard inputs)
        const processedContainers = new Set();
        fields.forEach(f => {
            // Mark containers of already-extracted fields
            const el = document.querySelector(f.selector);
            if (el) {
                const c = el.closest('[data-params]') || el.closest('[data-item-id]')
                       || el.closest('[role="listitem"]');
                if (c) processedContainers.add(c);
            }
        });

        // Find all question containers in the form
        const questionContainers = document.querySelectorAll(
            '[data-params], [data-item-id], .freebirdFormviewerComponentsQuestionBaseRoot, .Qr7Oae'
        );

        questionContainers.forEach(container => {
            if (processedContainers.has(container)) return;
            // Also skip if a parent is already processed
            let dominated = false;
            processedContainers.forEach(pc => {
                if (pc.contains(container) || container.contains(pc)) dominated = true;
            });

            // Check if this container has a standard input already in fields
            const stdInputs = container.querySelectorAll('input:not([type="hidden"]), select, textarea');
            if (stdInputs.length > 0) return;  // Already handled by standard extraction

            // Get question label
            const titleEl = container.querySelector(
                '.freebirdFormviewerComponentsQuestionBaseTitle, '
                + '.M7eMe, [role="heading"], .Qr7Oae > div > div > span'
            );
            const label = titleEl ? titleEl.textContent.trim() : '';
            if (!label) return;  // No label = not a real question

            // Check for required marker
            const required = !!(
                container.querySelector('[aria-label="Required question"]')
                || container.querySelector('.freebirdFormviewerComponentsQuestionBaseRequiredAsterisk')
                || container.textContent.includes('*')
            );

            // --- Radio group ---
            const radioGroup = container.querySelector('[role="radiogroup"]');
            if (radioGroup) {
                const radios = radioGroup.querySelectorAll('[role="radio"], [data-value]');
                const options = [];
                radios.forEach(r => {
                    // data-value is the primary source; fallback to aria-label or text
                    const val = r.getAttribute('data-value')
                             || r.getAttribute('aria-label')
                             || r.textContent.trim();
                    if (val) options.push(val);
                });
                // Build a selector for the radiogroup
                let selector = '';
                const dp = container.getAttribute('data-params');
                if (dp) {
                    const m = dp.match(/\["([^"]+)"/);
                    if (m) selector = '[data-params*="' + m[1] + '"] [role="radiogroup"]';
                }
                const itemId = container.getAttribute('data-item-id');
                if (!selector && itemId) {
                    selector = '[data-item-id="' + itemId + '"] [role="radiogroup"]';
                }
                if (!selector) {
                    // Fallback: nth-of-type among radiogroups
                    const allRG = document.querySelectorAll('[role="radiogroup"]');
                    for (let i = 0; i < allRG.length; i++) {
                        if (allRG[i] === radioGroup) {
                            selector = '[role="radiogroup"]:nth-of-type(' + (i+1) + ')';
                            break;
                        }
                    }
                }
                if (!selector) selector = '[role="radiogroup"]';

                fields.push({
                    tag: 'div',
                    type: 'gf_radio',
                    label: label,
                    selector: selector,
                    required: required,
                    value: '',
                    iframe: null,
                    options: options,
                });
                processedContainers.add(container);
                return;
            }

            // --- Checkbox group ---
            const checkboxes = container.querySelectorAll('[role="checkbox"]');
            if (checkboxes.length > 0) {
                const options = [];
                checkboxes.forEach(cb => {
                    const val = cb.getAttribute('data-answer-value')
                             || cb.getAttribute('aria-label')
                             || cb.textContent.trim();
                    if (val) options.push(val);
                });
                let selector = '';
                const dp = container.getAttribute('data-params');
                if (dp) {
                    const m = dp.match(/\["([^"]+)"/);
                    if (m) selector = '[data-params*="' + m[1] + '"]';
                }
                const itemId = container.getAttribute('data-item-id');
                if (!selector && itemId) {
                    selector = '[data-item-id="' + itemId + '"]';
                }
                if (!selector) selector = '.Qr7Oae';

                fields.push({
                    tag: 'div',
                    type: 'gf_checkbox',
                    label: label,
                    selector: selector,
                    required: required,
                    value: '',
                    iframe: null,
                    options: options,
                });
                processedContainers.add(container);
                return;
            }

            // --- Custom dropdown (role="listbox") ---
            const listbox = container.querySelector('[role="listbox"]');
            if (listbox) {
                const opts = listbox.querySelectorAll('[role="option"], [data-value]');
                const options = [];
                opts.forEach(o => {
                    const val = o.getAttribute('data-value')
                             || o.textContent.trim();
                    if (val && val !== 'Choose') options.push(val);
                });
                let selector = '';
                const dp = container.getAttribute('data-params');
                if (dp) {
                    const m = dp.match(/\["([^"]+)"/);
                    if (m) selector = '[data-params*="' + m[1] + '"] [role="listbox"]';
                }
                const itemId = container.getAttribute('data-item-id');
                if (!selector && itemId) {
                    selector = '[data-item-id="' + itemId + '"] [role="listbox"]';
                }
                if (!selector) selector = '[role="listbox"]';

                fields.push({
                    tag: 'div',
                    type: 'gf_dropdown',
                    label: label,
                    selector: selector,
                    required: required,
                    value: '',
                    iframe: null,
                    options: options,
                });
                processedContainers.add(container);
                return;
            }
        });
    }

    // Process iframes (try/catch for cross-origin)
    document.querySelectorAll('iframe').forEach((iframe, idx) => {
        try {
            const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
            if (!iframeDoc) return;
            const iframeSel = iframe.id ? '#' + iframe.id : 'iframe:nth-of-type(' + (idx + 1) + ')';
            iframeDoc.querySelectorAll(selectors).forEach(el => {
                const f = processElement(el, iframeSel);
                if (f) fields.push(f);
            });
        } catch (e) {
            // Cross-origin iframe — skip silently
        }
    });

    return fields;
}
"""


def _extract_form_context(page) -> tuple[str, list[dict]]:
    """Extract form fields from the current page.

    Returns:
        form_text: Human-readable field list for the LLM (indexed 1, 2, 3...).
        field_registry: List of dicts with real selectors, parallel to form_text indices.
    """
    try:
        raw_fields = page.evaluate(_EXTRACT_FIELDS_JS)
    except Exception as exc:
        logger.warning("Failed to extract form fields: %s", exc)
        return "", []

    if not raw_fields:
        return "", []

    form_lines: list[str] = []
    field_registry: list[dict] = []
    index = 0

    for field in raw_fields:
        index += 1
        ftype = field.get("type", "text")
        # Normalize select-one / select-multiple to 'select' (belt-and-suspenders)
        if ftype in ("select-one", "select-multiple"):
            ftype = "select"
        label = field.get("label", "")
        required = field.get("required", False)
        value = field.get("value", "")
        options = field.get("options", [])
        accept = field.get("accept", "")

        # Build human-readable line
        req_str = ", required" if required else ""
        line = f'{index}. [{ftype}{req_str}] "{label}"'

        if ftype in ("select", "gf_radio", "gf_checkbox", "gf_dropdown") and options:
            line += f" (options: {', '.join(options)})"
        elif ftype == "file" and accept:
            line += f" (accepts: {accept})"
        elif value:
            line += f' (current: "{value}")'
        else:
            line += " (empty)"

        form_lines.append(line)

        # Build registry entry
        field_registry.append({
            "index": index,
            "label": label,
            "selector": field.get("selector", ""),
            "type": ftype,
            "iframe": field.get("iframe"),
            "options": options,
        })

    form_text = "Form fields on this page:\n" + "\n".join(form_lines) if form_lines else ""
    return form_text, field_registry


# ---------------------------------------------------------------------------
# LLM field mapping
# ---------------------------------------------------------------------------

_MAP_SYSTEM_PROMPT = """You are a form-filling assistant. Given a list of numbered form fields and applicant data, return a JSON array mapping each field to its action.

CRITICAL RULES — FOLLOW EXACTLY:
- Use the field INDEX number to identify each field (matches the number in the FORM FIELDS list)
- ONLY use field_index numbers that appear in the FORM FIELDS list. Do NOT invent indices.

ACTIONS:
- "type": ONLY if the EXACT value exists in APPLICANT DATA. Set value to the matching data.
- "select": ONLY if the EXACT answer exists in APPLICANT DATA. Set value to the EXACT text of the matching option from the options list shown in parentheses.
- "upload": For file uploads where the field is empty. Set value to the resume_path from applicant data.
- "click": For checkboxes/radio buttons (including gf_radio, gf_checkbox types) ONLY if the answer exists in APPLICANT DATA. Set value to the EXACT text of the option to select.
- "skip": For submit buttons, fields that already have a correct value (shown as "current: ..."), or file uploads where a file is already uploaded.
- "ask": For ANY field where you do NOT have a matching value in APPLICANT DATA. This includes dropdowns, text fields, checkboxes, radio buttons — anything where the answer is not explicitly provided.

NEVER GUESS OR FABRICATE VALUES. If the applicant data does not contain an answer for a field, you MUST use "ask". Examples of fields that MUST be "ask" if not in applicant data:
- Salary/CTC/compensation questions
- Notice period
- "Are you a current employee of X?"
- Relocation willingness
- Work authorization / visa status
- "How did you hear about us?" (unless explicitly in applicant data)
- Any yes/no question not answered in applicant data
- Any dropdown where the correct choice is not in applicant data
- Gender, veteran status, disability status, ethnicity
- Any free-text question (e.g. "Why do you want to work here?")

For "ask" actions:
- If the field is a dropdown/select, set value to a JSON array of all available option texts: ["Option 1", "Option 2", ...]
- If the field is a checkbox group, set value to a JSON array of all checkbox labels: ["Label 1", "Label 2", ...]
- If the field is a text/number input, set value to "" (empty string)
- If the field is a radio button group, set value to a JSON array of all radio labels: ["Choice 1", "Choice 2", ...]

For radio button groups (multiple fields with the same name attribute), only include ONE entry with action "ask" listing all choices. Skip the rest.
For gf_radio, gf_checkbox, and gf_dropdown types (Google Forms custom widgets), treat them like radio/checkbox/select respectively. Use "click" if you know the answer, "ask" if you don't.
Return ONLY a valid JSON array, no markdown fences, no explanation."""


def _map_form_fields(form_text: str, applicant_data: dict) -> list[dict]:
    """Use the LLM to map form fields to applicant data values.

    One LLM call per form page. Returns a list of field mapping dicts.
    """
    from llm.client import complete

    # Build user prompt
    # Remove resume_path and cover_letter from the data shown to LLM if too long
    display_data = dict(applicant_data)
    if "cover_letter" in display_data and display_data["cover_letter"]:
        # Truncate cover letter for the prompt to save tokens
        cl = display_data["cover_letter"]
        if len(cl) > 500:
            display_data["cover_letter"] = cl[:500] + "... [truncated]"

    user_prompt = (
        f"FORM FIELDS:\n{form_text}\n\n"
        f"APPLICANT DATA:\n{json.dumps(display_data, indent=2)}\n\n"
        'Return a JSON array where each element has:\n'
        '{"field_index": 1, "field_label": "First Name", '
        '"action": "type"|"select"|"upload"|"click"|"skip", "value": "the value to fill"}'
    )

    # First attempt
    raw_response = complete(user_prompt, system_prompt=_MAP_SYSTEM_PROMPT)
    mapping = _parse_json_response(raw_response)

    if mapping is not None:
        return mapping

    # Retry once with a nudge
    logger.warning("LLM returned invalid JSON for field mapping — retrying")
    nudge_prompt = (
        f"{user_prompt}\n\n"
        "IMPORTANT: Your previous response was not valid JSON. "
        "Return ONLY a valid JSON array. No markdown, no explanation."
    )
    raw_response = complete(nudge_prompt, system_prompt=_MAP_SYSTEM_PROMPT)
    mapping = _parse_json_response(raw_response)

    if mapping is not None:
        return mapping

    logger.warning("LLM failed to return valid JSON after retry — returning empty mapping")
    return []


def _parse_json_response(raw: str) -> list[dict] | None:
    """Parse an LLM response as a JSON array, stripping markdown fences if present."""
    text = raw.strip()

    # Strip markdown code fences
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key, val in parsed.items():
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    logger.info("Extracted list from dict key '%s' (LLM wrapped array)", key)
                    return val
            if "field_index" in parsed or "action" in parsed:
                logger.info("LLM returned single dict — wrapping in list")
                return [parsed]
            logger.warning("LLM response is dict with no usable array, keys: %s", list(parsed.keys()))
            return None
        logger.warning("LLM response parsed as %s, expected list", type(parsed).__name__)
        return None
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Fill mapped fields
# ---------------------------------------------------------------------------

def _validate_mapping(field_mapping: list[dict], applicant_data: dict, field_registry: list[dict] | None = None) -> list[dict]:
    """Validate LLM field mapping -- force any fabricated values to 'ask'.

    The LLM sometimes ignores instructions and fills values it doesn't have.
    This function checks every type/select/click action to verify the value
    actually exists in applicant_data or form_answers.json. If not, it
    converts the action to 'ask'.

    Exception: for 'select' actions, if the LLM picked an option that is a
    reasonable default (e.g. "Prefer not to say", "N/A", "No") for EEO/demographic
    fields, or if the field is a single-answer field with an obvious correct
    choice, we trust the LLM's selection.
    """
    stored = _load_form_answers()
    # Build a set of all known values (lowercased) for fast lookup
    known_values = set()
    for v in applicant_data.values():
        if isinstance(v, str) and v.strip():
            known_values.add(v.strip().lower())
    for v in stored.values():
        if isinstance(v, str) and v.strip():
            known_values.add(v.strip().lower())

    # Also build a set of known field labels that have stored answers
    known_labels = set()
    for k in stored:
        known_labels.add(k.lower().strip())

    # Safe default values that the LLM can auto-select without user confirmation
    _SAFE_SELECT_VALUES = {
        "prefer not to say", "prefer not to disclose", "prefer not to answer",
        "decline to self-identify", "decline to identify",
        "i don't wish to answer", "i do not wish to answer",
        "n/a", "not applicable", "none",
    }

    # Yes/No are ONLY safe for EEO/demographic fields — not for questions like
    # "Have you previously worked at X?" or "Are you willing to relocate?"
    _EEO_ONLY_VALUES = {"yes", "no", "male", "female", "other"}

    # EEO / demographic field labels where safe defaults are acceptable
    _EEO_LABELS = {
        "gender", "sex", "race", "ethnicity", "veteran", "disability",
        "veteran status", "disability status", "race/ethnicity",
        "gender identity", "sexual orientation", "pronouns",
        "are you hispanic/latino", "hispanic", "latino",
    }

    # Build registry label lookup for fallback
    _reg_labels = {}
    if field_registry:
        for reg in field_registry:
            _reg_labels[reg.get("index", -1)] = reg.get("label", "")

    validated = []
    for mapping in field_mapping:
        action = mapping.get("action", "skip")
        value = mapping.get("value", "")
        # Use registry label as fallback when LLM returns empty label
        llm_label = mapping.get("field_label", "")
        reg_label = _reg_labels.get(mapping.get("field_index", -1), "")
        label = llm_label or reg_label
        # Update the mapping so downstream code gets the resolved label
        if label and not llm_label:
            mapping["field_label"] = label

        if action in ("skip", "upload", "ask"):
            validated.append(mapping)
            continue

        # For type/select/click -- verify the value is known
        if action in ("type", "select", "click"):
            value_str = str(value).strip().lower() if value else ""
            label_lower = label.lower().strip()

            # Check if this field label has a stored answer
            if label_lower in known_labels:
                validated.append(mapping)
                continue

            # Check if the value exists in known data
            if value_str and value_str in known_values:
                validated.append(mapping)
                continue

            # Check partial matches for common fields (name parts, email, phone)
            is_known = False
            for known_val in known_values:
                if value_str and len(value_str) > 2 and (value_str in known_val or known_val in value_str):
                    is_known = True
                    break

            if is_known:
                validated.append(mapping)
                continue

            # For select actions: trust the LLM if it picked a safe default
            # for EEO/demographic fields, or if the value is a common safe default
            if action == "select" and value_str:
                # Trust safe defaults for EEO fields
                is_eeo = any(eeo in label_lower for eeo in _EEO_LABELS)
                is_safe_value = value_str in _SAFE_SELECT_VALUES

                is_eeo_only_value = value_str in _EEO_ONLY_VALUES
                if (is_eeo and is_safe_value) or (is_eeo and is_eeo_only_value):
                    logger.info(
                        "Validation: trusting LLM select for EEO field '%s' = '%s'",
                        label, value,
                    )
                    validated.append(mapping)
                    continue

            # Value is fabricated -- convert to "ask"
            logger.info(
                "Validation: converting fabricated value for '%s' (%s='%s') to 'ask'",
                label, action, value,
            )
            ask_mapping = dict(mapping)
            ask_mapping["action"] = "ask"
            if action == "select" and not isinstance(value, list):
                ask_mapping["value"] = ""  # Will be populated from page
            validated.append(ask_mapping)
            continue

        validated.append(mapping)

    return validated


def _handle_gf_widget(
    page, action: str, value: str, label: str, field_index: int,
    registry_entry: dict, filled_labels: list[str], session_id: str,
) -> None:
    """Handle Google Forms custom widgets (radio, checkbox, dropdown).

    These are div-based ARIA widgets, not standard HTML form elements.
    """
    from pathlib import Path
    reg_type = registry_entry["type"]  # gf_radio, gf_checkbox, gf_dropdown
    selector = registry_entry["selector"]
    options = registry_entry.get("options", [])

    if action == "skip":
        return

    # For "ask" action, we need to get the user's response first
    if action == "ask":
        from bot.telegram_bot import ask_user, send_photo
        from config import OUTPUT_BASE_DIR

        # Check form_answers.json first
        stored = _load_form_answers()
        stored_value = None
        label_lower = label.lower().strip()
        for k, v in stored.items():
            if k.lower().strip() == label_lower:
                stored_value = v
                break

        if stored_value is not None:
            logger.info("Found stored answer for GF widget '%s': %s", label, stored_value)
            value = stored_value
        else:
            # Take screenshot and ask user
            ss_path = str(Path(OUTPUT_BASE_DIR) / f"ask_field_{field_index}.png")
            try:
                page.screenshot(path=ss_path)
                send_photo(ss_path, caption=(
                    f"\u2753 <b>{label}</b>\n\n"
                    + (f"Options: {', '.join(options)}\n\n" if options else "")
                    + "Reply with your choice, or <code>takeover</code> to fill manually."
                ))
            except Exception:
                pass

            response = ask_user(
                f"\u270f\ufe0f <b>{label}</b>\n"
                + (f"Options: {', '.join(options)}\n" if options else "")
                + "Reply with your answer:"
            )

            if response and response.strip().lower() == "takeover":
                # Manual takeover
                from bot.telegram_bot import send_notification, ask_yes_no
                from apply.session_handoff import get_takeover_instructions
                try:
                    instructions = get_takeover_instructions()
                    send_notification(
                        f"\U0001f3ae <b>Manual takeover for:</b> {label}\n\n"
                        f"{instructions}\n\n"
                        "Fill this field in the browser, then tap <b>Done</b>."
                    )
                except Exception:
                    from bot.telegram_bot import send_notification as _sn
                    _sn(
                        f"\U0001f3ae <b>Manual takeover for:</b> {label}\n\n"
                        "Fill this field in the browser, then tap <b>Done</b>."
                    )
                ask_yes_no("Tap \u2705 <b>Done</b> when you\'ve filled this field.")
                filled_labels.append(label)
                return

            if not response or not response.strip():
                logger.warning("No response for GF widget '%s' — skipping", label)
                return

            value = response.strip()
            # Save for future use
            _save_form_answer(label, value)

    elif action == "click" or action == "type" or action == "select":
        # value is already set from LLM mapping
        pass
    else:
        return

    if not value:
        return

    # Now click the matching option in the Google Forms widget
    if reg_type == "gf_radio":
        _click_gf_radio(page, selector, value, label, field_index, filled_labels)
    elif reg_type == "gf_checkbox":
        _click_gf_checkboxes(page, selector, value, label, field_index, filled_labels)
    elif reg_type == "gf_dropdown":
        _select_gf_dropdown(page, selector, value, label, field_index, filled_labels)


def _click_gf_radio(page, group_selector: str, value: str, label: str,
                     field_index: int, filled_labels: list[str]) -> None:
    """Click the matching radio option in a Google Forms radiogroup."""
    target = value.strip().lower()

    # Strategy 1: Find radio divs with data-value attribute
    container = page.locator(group_selector).first
    radios = container.locator('[role="radio"], [data-value]')
    count = radios.count()

    for i in range(count):
        radio = radios.nth(i)
        # Get the option text
        opt_text = radio.evaluate("""el => {
            return el.getAttribute('data-value')
                || el.getAttribute('aria-label')
                || el.textContent.trim();
        }""")
        if not opt_text:
            continue

        if opt_text.strip().lower() == target or target in opt_text.strip().lower():
            try:
                radio.click(timeout=5000)
                logger.info("Clicked GF radio '%s' for '%s'", opt_text, label)
                filled_labels.append(label)
                return
            except Exception:
                # Try JS click
                radio.evaluate("el => el.click()")
                logger.info("JS-clicked GF radio '%s' for '%s'", opt_text, label)
                filled_labels.append(label)
                return

    # Strategy 2: Fuzzy match — find closest option
    best_match = None
    best_score = 0
    for i in range(count):
        radio = radios.nth(i)
        opt_text = radio.evaluate("""el => {
            return el.getAttribute('data-value')
                || el.getAttribute('aria-label')
                || el.textContent.trim();
        }""") or ""
        opt_lower = opt_text.strip().lower()
        # Simple overlap score
        overlap = len(set(target.split()) & set(opt_lower.split()))
        if overlap > best_score:
            best_score = overlap
            best_match = (i, opt_text)

    if best_match and best_score > 0:
        radio = radios.nth(best_match[0])
        try:
            radio.click(timeout=5000)
        except Exception:
            radio.evaluate("el => el.click()")
        logger.info("Fuzzy-clicked GF radio '%s' for '%s' (target='%s')", best_match[1], label, value)
        filled_labels.append(label)
        return

    logger.warning("No matching GF radio option for '%s' = '%s'", label, value)


def _click_gf_checkboxes(page, container_selector: str, value: str, label: str,
                          field_index: int, filled_labels: list[str]) -> None:
    """Click matching checkbox options in a Google Forms checkbox group.

    value can be a single string or a JSON array string like '["A", "B"]'.
    """
    import json as _json

    # Parse value — could be JSON array or single value
    try:
        targets = _json.loads(value)
        if isinstance(targets, str):
            targets = [targets]
    except (_json.JSONDecodeError, TypeError):
        targets = [value]

    targets_lower = [t.strip().lower() for t in targets]

    container = page.locator(container_selector).first
    checkboxes = container.locator('[role="checkbox"]')
    count = checkboxes.count()

    clicked_any = False
    for i in range(count):
        cb = checkboxes.nth(i)
        cb_text = cb.evaluate("""el => {
            return el.getAttribute('data-answer-value')
                || el.getAttribute('aria-label')
                || el.textContent.trim();
        }""") or ""
        cb_lower = cb_text.strip().lower()

        for t in targets_lower:
            if cb_lower == t or t in cb_lower or cb_lower in t:
                try:
                    cb.click(timeout=5000)
                except Exception:
                    cb.evaluate("el => el.click()")
                logger.info("Clicked GF checkbox '%s' for '%s'", cb_text, label)
                clicked_any = True
                break

    if clicked_any:
        filled_labels.append(label)
    else:
        logger.warning("No matching GF checkbox options for '%s' = '%s'", label, value)


def _select_gf_dropdown(page, listbox_selector: str, value: str, label: str,
                         field_index: int, filled_labels: list[str]) -> None:
    """Select an option from a Google Forms custom dropdown (role=listbox)."""
    target = value.strip().lower()

    # Google Forms dropdowns need to be opened first (click the dropdown trigger)
    container = page.locator(listbox_selector).first

    # Try clicking the dropdown to open it
    try:
        container.click(timeout=3000)
        page.wait_for_timeout(500)  # Wait for dropdown animation
    except Exception:
        pass

    # Now find and click the matching option
    options = page.locator('[role="option"], [role="listbox"] [data-value]')
    count = options.count()

    for i in range(count):
        opt = options.nth(i)
        if not opt.is_visible():
            continue
        opt_text = opt.evaluate("""el => {
            return el.getAttribute('data-value')
                || el.textContent.trim();
        }""") or ""
        if opt_text.strip().lower() == target or target in opt_text.strip().lower():
            try:
                opt.click(timeout=5000)
                logger.info("Selected GF dropdown '%s' for '%s'", opt_text, label)
                filled_labels.append(label)
                return
            except Exception:
                opt.evaluate("el => el.click()")
                logger.info("JS-selected GF dropdown '%s' for '%s'", opt_text, label)
                filled_labels.append(label)
                return

    logger.warning("No matching GF dropdown option for '%s' = '%s'", label, value)



def _fill_mapped_fields(
    page, field_mapping: list[dict], field_registry: list[dict],
    session_id: str = ""
) -> list[str]:
    """Execute the field mapping by looking up real selectors from field_registry.

    Includes fuzzy select matching, disabled element handling, JS-click fallback,
    bounds checking, and reduced timeouts.

    Returns a list of field labels that were successfully filled.
    """
    # Build index lookup for field_registry
    registry_by_index: dict[int, dict] = {
        entry["index"]: entry for entry in field_registry
    }
    max_index = max(registry_by_index.keys()) if registry_by_index else 0

    filled_labels: list[str] = []

    for mapping in field_mapping:
        action = mapping.get("action", "skip")
        if action == "skip":
            continue

        field_index = mapping.get("field_index")
        value = mapping.get("value", "")
        # Use registry label as fallback — LLM sometimes returns empty field_label
        llm_label = mapping.get("field_label", "")
        registry_label = ""
        if field_index is not None:
            reg = registry_by_index.get(field_index)
            if reg:
                registry_label = reg.get("label", "")
        # Prefer LLM label, but fall back to registry label, then field_N
        label = llm_label or registry_label or f"field_{field_index}"

        # Bounds checking (Fix 8)
        if field_index is None or field_index < 1 or field_index > max_index:
            logger.warning(
                "Invalid field_index=%s (valid range: 1-%d, label=%s) — skipping",
                field_index, max_index, label,
            )
            continue

        registry_entry = registry_by_index.get(field_index)
        if registry_entry is None:
            logger.warning(
                "No registry entry for field_index=%d (label=%s)", field_index, label
            )
            continue

        selector = registry_entry["selector"]
        iframe_sel = registry_entry.get("iframe")

        try:
            # Get the right locator context (iframe or main page)
            if iframe_sel:
                context = page.frame_locator(iframe_sel)
            else:
                context = page

            el = context.locator(selector)

            # Disabled element handling (Fix 5)
            try:
                is_disabled = el.evaluate("el => el.disabled")
            except Exception:
                is_disabled = False

            if is_disabled:
                # Wait briefly — some forms enable fields dynamically
                page.wait_for_timeout(2000)
                try:
                    is_disabled = el.evaluate("el => el.disabled")
                except Exception:
                    pass

            if is_disabled:
                # Force-enable as last resort
                try:
                    el.evaluate("el => { el.disabled = false; el.removeAttribute('disabled'); }")
                    logger.info("Force-enabled disabled field: %s", label)
                except Exception:
                    logger.warning("Skipping disabled field '%s'", label)
                    continue

            # --- Google Forms custom widget handling ---
            reg_type = registry_entry.get("type", "text")
            if reg_type.startswith("gf_"):
                try:
                    _handle_gf_widget(
                        page, action, value, label, field_index,
                        registry_entry, filled_labels, session_id,
                    )
                except Exception as gf_exc:
                    logger.warning("Google Forms widget fill failed for '%s': %s", label, gf_exc)
                continue  # gf_ fields are fully handled — skip standard logic

            if action == "type":
                el.wait_for(timeout=5000)
                # Detect if this is actually a <select> (LLM sometimes
                # returns "type" for select elements)
                try:
                    tag_name = el.evaluate("el => el.tagName.toLowerCase()")
                except Exception:
                    tag_name = "input"
                if tag_name == "select":
                    # Redirect to select_option logic
                    try:
                        el.select_option(label=value, timeout=8000)
                        filled_labels.append(label)
                    except Exception:
                        try:
                            options = el.evaluate(
                                'el => Array.from(el.options).map(o => ({value: o.value, text: o.text.trim()}))'
                            )
                            match = _fuzzy_match_option(value, options)
                            if match:
                                el.select_option(value=match["value"], timeout=5000)
                                filled_labels.append(label)
                                logger.info("Type->Select fallback for '%s': '%s' → '%s'", label, value, match["text"])
                            else:
                                logger.warning("Type action on <select> '%s' — no matching option for '%s'", label, value)
                        except Exception as exc_sel:
                            logger.warning("Type->Select fallback failed for '%s': %s", label, exc_sel)
                else:
                    el.fill(value)
                    filled_labels.append(label)

            elif action == "select":
                el.wait_for(timeout=5000)
                try:
                    el.select_option(label=value, timeout=8000)
                    filled_labels.append(label)
                except Exception:
                    # Fuzzy match fallback (Fix 4)
                    try:
                        options = el.evaluate(
                            "el => Array.from(el.options).map(o => ({value: o.value, text: o.text.trim()}))"
                        )
                        match = _fuzzy_match_option(value, options)
                        if match:
                            try:
                                el.select_option(value=match["value"], timeout=5000)
                                filled_labels.append(label)
                                logger.info(
                                    "Fuzzy-matched select '%s': '%s' → '%s'",
                                    label, value, match["text"],
                                )
                            except Exception as exc2:
                                logger.warning("Fuzzy select failed for '%s': %s", label, exc2)
                        else:
                            # LLM value doesn't match any option — ask LLM to pick
                            # from the actual options list
                            logger.warning(
                                "No match found for select '%s' with value '%s' (options: %s)",
                                label, value, [o["text"] for o in options[:5]],
                            )
                            try:
                                from llm.client import complete as _complete
                                opt_texts = [o["text"] for o in options if o["text"].strip()]
                                _pick_prompt = (
                                    f"A form dropdown labeled \"{label}\" has these options:\n"
                                    + "\n".join(f"- {t}" for t in opt_texts)
                                    + f"\n\nThe applicant found this job via Indeed (job board). "
                                    f"Which option is the best match? Reply with ONLY the exact option text, nothing else."
                                )
                                _picked = _complete(_pick_prompt).strip().strip('"').strip("'")
                                _pick_match = _fuzzy_match_option(_picked, options)
                                if _pick_match:
                                    el.select_option(value=_pick_match["value"], timeout=5000)
                                    filled_labels.append(label)
                                    logger.info(
                                        "LLM re-picked select '%s': '%s' → '%s'",
                                        label, _picked, _pick_match["text"],
                                    )
                            except Exception as exc_pick:
                                logger.warning("LLM re-pick failed for '%s': %s", label, exc_pick)
                    except Exception as exc3:
                        logger.warning("Failed to extract options for '%s': %s", label, exc3)

            elif action == "upload":
                el.set_input_files(value)
                filled_labels.append(label)

            elif action == "click":
                try:
                    el.click(timeout=5000)
                    filled_labels.append(label)
                except Exception:
                    # Element might be visually hidden (custom-styled radio/checkbox)
                    try:
                        el.evaluate("el => el.click()")
                        filled_labels.append(label)
                        logger.info("JS-clicked hidden element: %s", label)
                    except Exception as exc2:
                        logger.warning("Failed to click '%s' (even with JS): %s", label, exc2)
                        # Notify user — take screenshot and ask for manual help
                        try:
                            from bot.telegram_bot import send_notification, send_photo
                            from config import OUTPUT_BASE_DIR
                            ss_path = str(Path(OUTPUT_BASE_DIR) / f"click_fail_{field_index}.png")
                            page.screenshot(path=ss_path)
                            send_photo(ss_path, caption=(
                                f"\u26a0\ufe0f <b>Could not select: {label}</b>\n\n"
                                "Please select this field manually in the browser, "
                                "then reply <code>done</code>."
                            ))
                            from bot.telegram_bot import ask_user
                            ask_user(
                                f"\u270f\ufe0f Please select <b>{label}</b> manually.\n"
                                "Reply <code>done</code> when finished."
                            )
                        except Exception as notify_exc:
                            logger.warning("Failed to notify user about click failure: %s", notify_exc)

            elif action == "ask":
                # Field value not in applicant data -- ask user via Telegram
                from bot.telegram_bot import ask_user, send_photo
                from config import OUTPUT_BASE_DIR

                # Skip if we already asked this field in a previous cycle
                ask_key = f"{label}::{field_index}"
                if ask_key in _asked_fields:
                    logger.info("Skipping already-asked field '%s' (index=%d)", label, field_index)
                    continue

                # First check form_answers.json -- maybe user already answered
                stored = _load_form_answers()
                stored_value = None
                label_lower = label.lower().strip()
                for k, v in stored.items():
                    if k.lower().strip() == label_lower:
                        stored_value = v
                        break

                if stored_value is not None:
                    # Found a stored answer -- use it
                    logger.info("Found stored answer for '%s': %s", label, stored_value)
                    reg_type = registry_entry.get("type", "text")
                    # Belt-and-suspenders: also check actual element tagName
                    if reg_type not in ("select",) and el.is_visible(timeout=1000):
                        try:
                            actual_tag = el.evaluate("el => el.tagName.toLowerCase()")
                            if actual_tag == "select":
                                reg_type = "select"
                        except Exception:
                            pass
                    if reg_type == "select":
                        try:
                            el.select_option(label=stored_value, timeout=8000)
                            filled_labels.append(label)
                        except Exception:
                            opts = el.evaluate(
                                'el => Array.from(el.options).map(o => ({value: o.value, text: o.text.trim()}))'
                            )
                            match = _fuzzy_match_option(stored_value, opts)
                            if match:
                                el.select_option(value=match["value"], timeout=5000)
                                filled_labels.append(label)
                            else:
                                logger.warning("Stored answer '%s' no match for '%s'", stored_value, label)
                    elif reg_type in ("checkbox", "radio"):
                        el.click(timeout=5000)
                        filled_labels.append(label)
                    else:
                        el.fill(stored_value, timeout=5000)
                        filled_labels.append(label)
                    continue

                # Not in memory -- build a question for the user
                options_list = value  # LLM puts options array in value for "ask"
                reg_type = registry_entry.get("type", "text")
                # Belt-and-suspenders: check actual tagName
                if reg_type not in ("select",):
                    try:
                        actual_tag = el.evaluate("el => el.tagName.toLowerCase()")
                        if actual_tag == "select":
                            reg_type = "select"
                    except Exception:
                        pass

                # For dropdowns, get the actual options from the page
                if reg_type == "select":
                    try:
                        page_options = el.evaluate(
                            'el => Array.from(el.options).map(o => o.text.trim()).filter(t => t && t !== "")'
                        )
                        if page_options:
                            options_list = page_options
                    except Exception:
                        pass

                # Build the Telegram question
                if isinstance(options_list, list) and options_list:
                    if len(options_list) > 10:
                        # Many options -- send a screenshot
                        try:
                            ss_path = str(Path(OUTPUT_BASE_DIR) / f"ask_field_{field_index}.png")
                            page.screenshot(path=ss_path)
                            question = (
                                f"\U0001f4cb <b>Field:</b> {label}\n"
                                f"This field has {len(options_list)} options (see screenshot).\n\n"
                                "Reply with the EXACT option text to select.\n\n"
                                "\u2022 Type <code>skip</code> to skip this field\n"
                                "\u2022 Type <code>takeover</code> to fill manually in browser"
                            )
                            response = ask_user(question, screenshot_path=ss_path)
                        except Exception:
                            opts_text = "\n".join(f"  \u2022 {o}" for o in options_list)
                            question = (
                                f"\U0001f4cb <b>Field:</b> {label}\n\n"
                                f"<b>Options:</b>\n{opts_text}\n\n"
                                "Reply with the EXACT option text to select.\n\n"
                                "\u2022 Type <code>skip</code> to skip this field\n"
                                "\u2022 Type <code>takeover</code> to fill manually in browser"
                            )
                            response = ask_user(question)
                    else:
                        opts_text = "\n".join(f"  \u2022 {o}" for o in options_list)
                        question = (
                            f"\U0001f4cb <b>Field:</b> {label}\n\n"
                            f"<b>Options:</b>\n{opts_text}\n\n"
                            "Reply with the EXACT option text to select.\n\n"
                                "\u2022 Type <code>skip</code> to skip this field\n"
                                "\u2022 Type <code>takeover</code> to fill manually in browser"
                        )
                        response = ask_user(question)
                else:
                    # Free text field — always include screenshot for context
                    try:
                        ss_path = str(Path(OUTPUT_BASE_DIR) / f"ask_field_{field_index}.png")
                        page.screenshot(path=ss_path)
                        question = (
                            f"\u270f\ufe0f <b>Field:</b> {label}\n\n"
                            "Reply with the value to fill in.\n\n"
                            "\u2022 Type <code>skip</code> to skip this field\n"
                            "\u2022 Type <code>takeover</code> to fill manually in browser"
                        )
                        response = ask_user(question, screenshot_path=ss_path)
                    except Exception:
                        question = (
                            f"\u270f\ufe0f <b>Field:</b> {label}\n\n"
                            "Reply with the value to fill in.\n\n"
                            "\u2022 Type <code>skip</code> to skip this field\n"
                            "\u2022 Type <code>takeover</code> to fill manually in browser"
                        )
                        response = ask_user(question)

                if response and response != "__timeout__":
                    # --- Handle skip: don't fill, don't save to JSON ---
                    if response.strip().lower() == "skip":
                        logger.info("User skipped field '%s' (index=%d)", label, field_index)
                        _asked_fields.add(ask_key)
                        continue

                    # --- Handle takeover: user fills manually in browser ---
                    if response.strip().lower() == "takeover":
                        logger.info("User requested takeover for field '%s' (index=%d)", label, field_index)
                        try:
                            from bot.telegram_bot import send_notification as _sn_takeover
                            from apply.session_handoff import get_takeover_instructions
                            cdp_instructions = get_takeover_instructions(session_id)
                            _sn_takeover(
                                f"\U0001f3ae <b>Manual takeover for:</b> {label}\n\n"
                                f"<pre>{cdp_instructions}</pre>\n\n"
                                "Fill this field in the browser, then tap <b>Done</b>."
                            )
                        except Exception as _cdp_err:
                            from bot.telegram_bot import send_notification as _sn_takeover2
                            _sn_takeover2(
                                f"\U0001f3ae <b>Manual takeover for:</b> {label}\n\n"
                                "Fill this field in the browser, then tap <b>Done</b>."
                            )
                        from bot.telegram_bot import ask_yes_no
                        ask_yes_no("Tap \u2705 <b>Done</b> when you\'ve filled this field.")
                        logger.info("User completed manual takeover for field '%s'", label)
                        _asked_fields.add(ask_key)
                        filled_labels.append(label)
                        continue

                    # Belt-and-suspenders: check actual tagName
                    if reg_type not in ("select",) and el.is_visible(timeout=1000):
                        try:
                            actual_tag = el.evaluate("el => el.tagName.toLowerCase()")
                            if actual_tag == "select":
                                reg_type = "select"
                        except Exception:
                            pass
                    if reg_type == "select":
                        try:
                            el.select_option(label=response.strip(), timeout=8000)
                            filled_labels.append(label)
                        except Exception:
                            opts = el.evaluate(
                                'el => Array.from(el.options).map(o => ({value: o.value, text: o.text.trim()}))'
                            )
                            match = _fuzzy_match_option(response.strip(), opts)
                            if match:
                                el.select_option(value=match["value"], timeout=5000)
                                filled_labels.append(label)
                                logger.info("Fuzzy-matched user response for '%s': '%s' -> '%s'", label, response, match["text"])
                            else:
                                logger.warning("User response '%s' no match for '%s'", response, label)
                    elif reg_type in ("checkbox", "radio"):
                        # For radio buttons, find the specific radio matching the user's response
                        if reg_type == "radio":
                            user_val = response.strip().lower()
                            # Try to find the radio button with matching label text
                            try:
                                # Look for radio buttons with the same name attribute
                                name_attr = el.evaluate("el => el.name")
                                if name_attr:
                                    radios = page.locator(f'input[type="radio"][name="{name_attr}"]')
                                    radio_count = radios.count()
                                    clicked = False
                                    for ri in range(radio_count):
                                        radio = radios.nth(ri)
                                        # Get the label for this radio
                                        radio_label = radio.evaluate("""el => {
                                            // Check for associated label
                                            if (el.id) {
                                                const lbl = document.querySelector('label[for="' + el.id + '"]');
                                                if (lbl) return lbl.textContent.trim();
                                            }
                                            // Check parent label
                                            const parent = el.closest('label');
                                            if (parent) return parent.textContent.trim();
                                            // Check next sibling text
                                            const next = el.nextSibling;
                                            if (next && next.textContent) return next.textContent.trim();
                                            return el.value || '';
                                        }""")
                                        if radio_label.lower().strip() == user_val or radio_label.lower().strip().startswith(user_val):
                                            radio.click(timeout=5000)
                                            clicked = True
                                            logger.info("Clicked radio '%s' for field '%s'", radio_label, label)
                                            break
                                    if not clicked:
                                        # Fallback: click the original element
                                        el.click(timeout=5000)
                                else:
                                    el.click(timeout=5000)
                            except Exception as radio_exc:
                                logger.warning("Radio selection failed for '%s': %s", label, radio_exc)
                                try:
                                    el.click(timeout=5000)
                                except Exception:
                                    # Radio click completely failed — notify user for manual action
                                    try:
                                        from bot.telegram_bot import send_notification, send_photo, ask_user as _ask_user_radio
                                        from config import OUTPUT_BASE_DIR
                                        ss_path = str(Path(OUTPUT_BASE_DIR) / f"radio_fail_{field_index}.png")
                                        page.screenshot(path=ss_path)
                                        send_photo(ss_path, caption=(
                                            f"\u26a0\ufe0f <b>Could not select '{response}' for: {label}</b>\n\n"
                                            "Please select this option manually in the browser, "
                                            "then reply <code>done</code>."
                                        ))
                                        _ask_user_radio(
                                            f"\u270f\ufe0f Please select <b>{response}</b> for <b>{label}</b> manually.\n"
                                            "Reply <code>done</code> when finished."
                                        )
                                    except Exception as _ne:
                                        logger.warning("Failed to notify user about radio failure: %s", _ne)
                        else:
                            el.click(timeout=5000)
                        filled_labels.append(label)
                    else:
                        # Belt-and-suspenders: check if this is actually a <select>
                        try:
                            actual_tag = el.evaluate("el => el.tagName.toLowerCase()")
                        except Exception:
                            actual_tag = "input"
                        if actual_tag == "select":
                            try:
                                el.select_option(label=response.strip(), timeout=8000)
                            except Exception:
                                opts = el.evaluate(
                                    'el => Array.from(el.options).map(o => ({value: o.value, text: o.text.trim()}))'
                                )
                                match = _fuzzy_match_option(response.strip(), opts)
                                if match:
                                    el.select_option(value=match["value"], timeout=5000)
                                else:
                                    logger.warning("Fallback select failed for '%s'", label)
                        else:
                            el.fill(response.strip(), timeout=5000)
                        filled_labels.append(label)

                    # Save to form_answers.json for future reuse
                    _save_form_answer(label, response.strip())
                    logger.info("Saved user answer for '%s' to form_answers.json", label)
                    _asked_fields.add(ask_key)
                else:
                    logger.warning("No response from user for field '%s' -- skipping", label)
                    _asked_fields.add(ask_key)  # Don't re-ask on retry

            else:
                logger.warning("Unknown action '%s' for field %s", action, label)

        except Exception as exc:
            logger.warning(
                "Failed to fill field '%s' (index=%d, action=%s): %s",
                label, field_index, action, exc,
            )

    return filled_labels


# ---------------------------------------------------------------------------
# Blocker detection
# ---------------------------------------------------------------------------

def _detect_blockers(page) -> list[dict] | None:
    """Scan the page for anything blocking progress.

    Returns None if no blockers found, or a list of dicts describing all blockers.
    """
    from config import FORM_ANSWERS_PATH

    output_dir = Path(FORM_ANSWERS_PATH).parent

    try:
        blockers = page.evaluate("""
        () => {
            const results = [];

            // 0. SKIP cookie consent banners (handled by _dismiss_cookie_banners)
            const cookieSelectors = [
                '#onetrust-banner-sdk', '#CybotCookiebotDialog',
                '[class*="cookie-banner"]', '[class*="cookie-consent"]',
                '[class*="consent-banner"]', '[id*="cookie"]',
                '[class*="gdpr"]', '.osano-cm-dialog'
            ];
            for (const sel of cookieSelectors) {
                const el = document.querySelector(sel);
                if (el) {
                    const rect = el.getBoundingClientRect();
                    if (rect.width > 0 && rect.height > 0) {
                        // Cookie banner visible — not a blocker, handled separately
                        return results.length > 0 ? results : null;
                    }
                }
            }

            // 1. CAPTCHA detection (highest priority)
            const iframes = document.querySelectorAll('iframe');
            for (const iframe of iframes) {
                const src = (iframe.src || '').toLowerCase();
                if (src.includes('captcha') || src.includes('recaptcha') || src.includes('hcaptcha')) {
                    results.push({type: 'captcha', details: 'CAPTCHA detected: ' + src, field_label: null, field_selector: null});
                    return results;
                }
            }
            const captchaEls = document.querySelectorAll('[class*="captcha"], #captcha, .g-recaptcha, .h-captcha');
            for (const el of captchaEls) {
                const rect = el.getBoundingClientRect();
                if (rect.width > 0 && rect.height > 0) {
                    results.push({type: 'captcha', details: 'CAPTCHA element detected', field_label: null, field_selector: null});
                    return results;
                }
            }

            // 2. Modal popups (EXCLUDING cookie consent)
            const dialogs = document.querySelectorAll('[role="dialog"], [role="alertdialog"]');
            for (const d of dialogs) {
                const rect = d.getBoundingClientRect();
                if (rect.width > 0 && rect.height > 0) {
                    // Skip cookie consent dialogs
                    const dText = (d.textContent || '').toLowerCase();
                    const dCls = (d.className || '').toLowerCase();
                    const dId = (d.id || '').toLowerCase();
                    if (dText.includes('cookie') || dText.includes('consent') || dText.includes('gdpr') ||
                        dCls.includes('cookie') || dCls.includes('consent') ||
                        dId.includes('cookie') || dId.includes('consent')) {
                        continue;
                    }
                    results.push({type: 'modal_popup', details: 'Modal dialog blocking form: ' + (d.textContent || '').substring(0, 200).trim(), field_label: null, field_selector: null});
                    return results;
                }
            }

            // 3. Validation errors and required field errors
            const errorSelectors = '[class*="error"]:not([class*="error-boundary"]), [class*="invalid"], [role="alert"], [aria-invalid="true"]';
            const errorEls = document.querySelectorAll(errorSelectors);
            for (const el of errorEls) {
                const rect = el.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) continue;

                const text = (el.textContent || '').trim().toLowerCase();
                if (!text) continue;

                // Check if this is a form-related error (not a generic page error)
                const isFormRelated = el.closest('form') ||
                    el.closest('[class*="field"]') ||
                    el.closest('[class*="input"]') ||
                    el.previousElementSibling?.tagName?.match(/^(INPUT|SELECT|TEXTAREA)$/i) ||
                    el.nextElementSibling?.tagName?.match(/^(INPUT|SELECT|TEXTAREA)$/i);

                if (!isFormRelated && !text.match(/required|mandatory|please fill|please enter|invalid|must be/i)) continue;

                if (text.match(/required|mandatory|please fill|please enter/i)) {
                    // Try to identify the field
                    const nearInput = el.closest('[class*="field"]')?.querySelector('input, select, textarea');
                    let fieldLabel = null;
                    let fieldSelector = null;
                    if (nearInput) {
                        if (nearInput.id) {
                            // CSS.escape handles dots, brackets, colons in IDs
                            fieldSelector = '#' + CSS.escape(nearInput.id);
                        } else if (nearInput.name) {
                            fieldSelector = nearInput.tagName.toLowerCase() + '[name="' + nearInput.name + '"]';
                        } else {
                            fieldSelector = null;
                        }
                        const label = nearInput.id ? document.querySelector('label[for="' + nearInput.id + '"]') : null;
                        fieldLabel = label ? label.textContent.trim() : (nearInput.getAttribute('aria-label') || nearInput.placeholder || nearInput.name || null);
                    }
                    results.push({type: 'missing_field', details: text.substring(0, 200), field_label: fieldLabel, field_selector: fieldSelector});
                } else {
                    results.push({type: 'validation_error', details: text.substring(0, 200), field_label: null, field_selector: null});
                }
            }

            // 4. Drag-and-drop upload zones (not supported in v1)
            const dropzones = document.querySelectorAll('[class*="dropzone"], [class*="drag-drop"], [class*="file-drop"]');
            for (const dz of dropzones) {
                const rect = dz.getBoundingClientRect();
                if (rect.width > 0 && rect.height > 0) {
                    // Check if there's a standard file input nearby
                    const fileInput = dz.querySelector('input[type="file"]');
                    if (!fileInput) {
                        results.push({type: 'unknown', details: 'Drag-and-drop file upload detected — not supported in v1', field_label: null, field_selector: null});
                    }
                }
            }

            // 5. Mandatory resume upload gate (Phenom, SuccessFactors, Workday)
            // Detect when the page is a resume upload gate that blocks progress
            const uploadBtns2 = document.querySelectorAll('button, a, [role="button"]');
            let hasResumeUploadBtn = false;
            for (const btn of uploadBtns2) {
                const rect = btn.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) continue;
                const txt = (btn.textContent || '').trim().toLowerCase();
                if (txt.match(/upload\\s*(resume|cv|your resume)/i) ||
                    txt.match(/attach\\s*(resume|cv)/i)) {
                    hasResumeUploadBtn = true;
                    break;
                }
            }
            const bodyTextLower = document.body.innerText.toLowerCase();
            const resumeRequired = bodyTextLower.includes('resume*') ||
                bodyTextLower.match(/resume\\s+is\\s+required/) ||
                bodyTextLower.match(/please\\s+upload.*resume/) ||
                bodyTextLower.match(/submit.*personal.*professional.*by.*uploading/);
            if (hasResumeUploadBtn && resumeRequired) {
                results.push({
                    type: 'resume_upload_required',
                    details: 'Mandatory resume upload required before form can proceed',
                    field_label: 'Resume Upload',
                    field_selector: null
                });
            }

            return results.length > 0 ? results : null;
        }
        """)
    except Exception as exc:
        logger.warning("Blocker detection JS failed: %s", exc)
        return None

    if not blockers:
        return None

    # Take screenshot once for all blockers
    timestamp = int(time.time())
    output_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = str(output_dir / f"blocker_{timestamp}.png")
    try:
        page.screenshot(path=screenshot_path)
    except Exception as exc:
        logger.warning("Failed to take blocker screenshot: %s", exc)
        screenshot_path = ""

    # Attach screenshot to all blockers
    for b in blockers:
        b["screenshot_path"] = screenshot_path

    return blockers  # Return ALL blockers


# ---------------------------------------------------------------------------
# Handle blockers (Telegram stubs)
# ---------------------------------------------------------------------------

def _handle_blocker(blocker: dict, page, session_id: str) -> str:
    """Handle a detected blocker via Telegram interrupt (stubs for now).

    For simple blockers: check form_answers.json first, then ask user.
    For complex blockers: provide CDP takeover instructions.

    Returns:
        "resolved_from_memory" | "resolved" | "resolved_manually"
    """
    from apply.session_handoff import get_takeover_instructions

    blocker_type = blocker.get("type", "unknown")
    details = blocker.get("details", "")
    field_label = blocker.get("field_label")
    field_selector = blocker.get("field_selector")
    screenshot_path = blocker.get("screenshot_path", "")

    # If field_label is empty, try to derive a readable label from the selector or page
    if not field_label and field_selector:
        try:
            loc = page.locator(field_selector)
            # Try to get the associated <label> text from the page
            derived = loc.evaluate("""el => {
                // 1. <label for="id">
                if (el.id) {
                    const lbl = document.querySelector('label[for="' + el.id + '"]');
                    if (lbl && lbl.textContent.trim()) return lbl.textContent.trim();
                }
                // 2. aria-label
                if (el.getAttribute('aria-label')) return el.getAttribute('aria-label').trim();
                // 3. placeholder
                if (el.placeholder) return el.placeholder.trim();
                // 4. Closest parent label
                const parent = el.closest('label');
                if (parent) return parent.textContent.replace(el.textContent || '', '').trim();
                // 5. Preceding sibling
                const prev = el.previousElementSibling;
                if (prev && (prev.tagName === 'LABEL' || prev.tagName === 'SPAN'))
                    return prev.textContent.trim();
                // 6. Clean up id/name
                if (el.name) return el.name.replace(/[_.-]/g, ' ').replace(/([a-z])([A-Z])/g, ' ');
                if (el.id) return el.id.replace(/[_.-]/g, ' ').replace(/([a-z])([A-Z])/g, ' ');
                return '';
            }""")
            if derived:
                field_label = derived
                blocker["field_label"] = field_label
        except Exception:
            # Fallback: clean up the selector itself
            import re as _re
            sel_match = _re.search(r'#([\w.-]+)', field_selector)
            if sel_match:
                raw_id = sel_match.group(1)
                # Convert camelCase/snake_case to readable
                readable = _re.sub(r'([a-z])([A-Z])', r' ', raw_id)
                readable = readable.replace('_', ' ').replace('.', ' ').replace('-', ' ').strip()
                if readable:
                    field_label = readable.title()
                    blocker["field_label"] = field_label

    # --- Simple blockers: missing_field, validation_error ---
    if blocker_type in ("missing_field", "validation_error"):
        # Skip if already asked in this session
        blocker_ask_key = f"blocker::{field_label}::{field_selector}"
        if blocker_ask_key in _asked_fields:
            logger.info("Skipping already-asked blocker field '%s'", field_label)
            return "resolved"

        # Check form_answers.json for a stored answer
        if field_label:
            stored_answers = _load_form_answers()
            # Normalize the key for lookup
            normalized_key = field_label.lower().strip().replace(" ", "_")
            for stored_key, stored_value in stored_answers.items():
                if stored_key.lower().replace(" ", "_") == normalized_key:
                    # Found a stored answer — fill it
                    if field_selector:
                        try:
                            loc = page.locator(field_selector)
                            loc.wait_for(state="attached", timeout=5000)
                            # Detect if this is a <select> element
                            tag_name = loc.evaluate("el => el.tagName.toLowerCase()")
                            if tag_name == "select":
                                # Use select_option for <select> elements
                                try:
                                    loc.select_option(label=stored_value, timeout=8000)
                                except Exception:
                                    opts = loc.evaluate(
                                        'el => Array.from(el.options).map(o => ({value: o.value, text: o.text.trim()}))'
                                    )
                                    match = _fuzzy_match_option(stored_value, opts)
                                    if match:
                                        loc.select_option(value=match["value"], timeout=5000)
                                    else:
                                        raise ValueError(f"No matching option for '{stored_value}'")
                            else:
                                loc.fill(stored_value, timeout=5000)
                            # Trigger change/blur events to clear validation state
                            # (Phenom and other ATS use custom validation that
                            # doesn't clear on .fill() alone)
                            try:
                                loc.dispatch_event("input")
                                loc.dispatch_event("change")
                                loc.dispatch_event("blur")
                            except Exception:
                                pass
                            logger.info(
                                "Resolved blocker from memory: %s = %s",
                                field_label, stored_value,
                            )
                            return "resolved_from_memory"
                        except Exception as exc:
                            logger.warning(
                                "Failed to fill stored answer for '%s': %s",
                                field_label, exc,
                            )
                            if field_selector:
                                _failed_selectors[field_selector] = _failed_selectors.get(field_selector, 0) + 1

        # Not in memory — ask the user via Telegram
        from bot.telegram_bot import ask_user
        from config import OUTPUT_BASE_DIR

        # For <select> fields, extract the actual options and show them
        select_options = None
        if field_selector:
            try:
                loc = page.locator(field_selector)
                tag_name = loc.evaluate("el => el.tagName.toLowerCase()")
                if tag_name == "select":
                    select_options = loc.evaluate(
                        'el => Array.from(el.options).map(o => o.text.trim()).filter(t => t && t !== "")'
                    )
            except Exception:
                pass

        if select_options:
            # Show dropdown options to user
            if len(select_options) > 15:
                # Many options — take a screenshot with dropdown open, list all options
                try:
                    # Try to open the dropdown for the screenshot
                    loc = page.locator(field_selector)
                    loc.click(timeout=3000)
                    page.wait_for_timeout(500)
                except Exception:
                    pass

                ss_path = str(Path(OUTPUT_BASE_DIR) / f"blocker_select_{int(time.time())}.png")
                try:
                    page.screenshot(path=ss_path)
                except Exception:
                    ss_path = screenshot_path

                # Close dropdown after screenshot
                try:
                    page.keyboard.press("Escape")
                except Exception:
                    pass

                opts_text = "\n".join(f"  • {o}" for o in select_options)
                question = (
                    f"📋 <b>Field:</b> {field_label}\n\n"
                    f"<b>Options ({len(select_options)}):</b>\n{opts_text}\n\n"
                    "Reply with the EXACT option text to select.\n\n"
                                "\u2022 Type <code>skip</code> to skip this field\n"
                                "\u2022 Type <code>takeover</code> to fill manually in browser"
                )
                response = ask_user(question, screenshot_path=ss_path if ss_path else None)
            else:
                opts_text = "\n".join(f"  • {o}" for o in select_options)
                question = (
                    f"📋 <b>Field:</b> {field_label}\n\n"
                    f"<b>Options:</b>\n{opts_text}\n\n"
                    "Reply with the EXACT option text to select.\n\n"
                                "\u2022 Type <code>skip</code> to skip this field\n"
                                "\u2022 Type <code>takeover</code> to fill manually in browser"
                )
                response = ask_user(question, screenshot_path=screenshot_path if screenshot_path else None)
        else:
            question = f"🚫 <b>BLOCKER:</b> {blocker_type}\n{details}"
            if field_label:
                question += (
                    f"\n\n<b>Field:</b> {field_label}\n"
                    "Reply with the value to fill.\n\n"
                    "\u2022 Type <code>skip</code> to skip this field\n"
                    "\u2022 Type <code>takeover</code> to fill manually in browser"
                )
            else:
                question += (
                    "\n\nReply with the value to fill.\n\n"
                    "\u2022 Type <code>skip</code> to skip this field\n"
                    "\u2022 Type <code>takeover</code> to fill manually in browser"
                )

            response = ask_user(question, screenshot_path=screenshot_path if screenshot_path else None)

        if response == "__timeout__":
            logger.warning("Telegram response timed out for blocker: %s", details)
            return _handle_complex_blocker(blocker, session_id)

        if response.strip().lower() == "skip":
            logger.info("User skipped blocker field '%s'", field_label or "unknown")
            _asked_fields.add(blocker_ask_key)
            return "resolved"

        if response.strip().lower() == "takeover":
            # Fall through to complex handler
            return _handle_complex_blocker(blocker, session_id)

        # Fill the value
        if field_selector and response:
            try:
                loc = page.locator(field_selector)
                loc.wait_for(state="attached", timeout=5000)
                # Detect if this is a <select> element
                tag_name = loc.evaluate("el => el.tagName.toLowerCase()")
                if tag_name == "select":
                    # Use select_option for <select> elements
                    try:
                        loc.select_option(label=response.strip(), timeout=8000)
                    except Exception:
                        opts = loc.evaluate(
                            'el => Array.from(el.options).map(o => ({value: o.value, text: o.text.trim()}))'
                        )
                        match = _fuzzy_match_option(response.strip(), opts)
                        if match:
                            loc.select_option(value=match["value"], timeout=5000)
                            logger.info("Fuzzy-matched blocker response for '%s': '%s' -> '%s'", field_label, response, match["text"])
                        else:
                            logger.warning("User response '%s' no match for select '%s'", response, field_label)
                else:
                    loc.fill(response.strip(), timeout=5000)
                # Trigger change/blur events to clear validation state
                try:
                    loc.dispatch_event("input")
                    loc.dispatch_event("change")
                    loc.dispatch_event("blur")
                except Exception:
                    pass
            except Exception as exc:
                logger.warning("Failed to fill user response for '%s': %s", field_label, exc)
                if field_selector:
                    _failed_selectors[field_selector] = _failed_selectors.get(field_selector, 0) + 1

        # Save to form_answers.json
        if field_label and response:
            _save_form_answer(field_label, response)

        # Mark as asked so we don't re-ask in retry cycles
        _asked_fields.add(blocker_ask_key)

        return "resolved"

    # --- Resume upload required ---
    if blocker_type == "resume_upload_required":
        # This is handled specially in fill_form() via _handle_resume_upload_gate()
        # If we reach here, it means the gate wasn't caught earlier — treat as complex
        return _handle_complex_blocker(blocker, session_id)

    # --- Complex blockers: captcha, modal_popup, unknown ---
    return _handle_complex_blocker(blocker, session_id)


def _handle_complex_blocker(blocker: dict, session_id: str) -> str:
    """Handle a complex blocker that requires manual CDP takeover."""
    blocker_type = blocker.get("type", "unknown")
    details = blocker.get("details", "")
    screenshot_path = blocker.get("screenshot_path", "")

    from bot.telegram_bot import send_notification, send_photo, request_interrupt
    from apply.session_handoff import get_takeover_instructions

    # Send screenshot if available
    if screenshot_path:
        send_photo(screenshot_path, caption="\u26a0\ufe0f <b>Manual Takeover Required</b>")

    # Send CDP instructions
    try:
        cdp_instructions = get_takeover_instructions(session_id)
        send_notification(f"<pre>{cdp_instructions}</pre>")
    except Exception as _cdp_err:
        logger.warning("Failed to get CDP instructions: %s", _cdp_err)

    # Single blocking prompt — [Done] button
    response = request_interrupt(
        interrupt_type="complex_blocker",
        message=(
            f"\u26a0\ufe0f <b>Blocker detected:</b> {blocker_type}\n"
            f"{details}\n\n"
            "Fix this in the browser, then tap <b>Done</b>."
        ),
        buttons=[
            {"text": "\u2705 Done", "callback_data": "blocker_done"},
        ],
        timeout=600,
    )
    if response is None:
        logger.warning("User did not confirm resolution for blocker: %s (timed out)", details)
    return "resolved_manually"


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def fill_form(url: str, documents: dict, submit: bool = True) -> dict:
    """Fill an ATS application form using LLM-assisted field mapping.

    Semi-autonomous: uploads resume first (to trigger auto-population),
    detects LinkedIn autofill, fills remaining fields via LLM mapping,
    retries on validation errors, and asks the user only as a last resort.

    Args:
        url: The ATS application URL to navigate to.
        documents: dict containing:
            - resume_pdf_path: str — path to tailored PDF
            - cover_letter_text: str | None
            - applicant_name: str
            - applicant_email: str
            - github_url: str | None
            - linkedin_url: str | None
            - phone: str | None

    Returns:
        dict with keys:
            - "status": "submitted" | "partially_filled" | "failed" | "handed_off"
            - "url": str — the final URL after form filling
            - "notes": str — human-readable summary of what happened
    """
    from config import BROWSER_HEADLESS
    from apply.session_handoff import launch_browser, close_browser
    from playwright.sync_api import sync_playwright

    session_id = f"form_{int(time.time())}"

    # 1. Launch browser via session_handoff
    browser_info = launch_browser(session_id=session_id, headless=BROWSER_HEADLESS)

    # 2. Connect Playwright via CDP
    pw = sync_playwright().start()
    browser = pw.chromium.connect_over_cdp(browser_info["cdp_url"])
    context = browser.contexts[0] if browser.contexts else browser.new_context()
    page = context.new_page()

    # 3. Build applicant data (merge config + documents + stored answers)
    applicant_data = _build_applicant_data(documents)

    status = "failed"  # default
    final_url = url
    notes = ""

    try:
        # 4. Navigate to the URL
        # Use domcontentloaded (fast, reliable) then best-effort networkidle.
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        try:
            page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass  # page is usable even without full network quiescence

        filled_fields: list[str] = []
        max_pages = 10  # safety limit for multi-page forms

        # 4.1 — Auto-dismiss cookie/consent banners and chatbot overlays
        _dismiss_cookie_banners(page)
        _dismiss_chatbots(page)

        # 4.2 — Navigate from JD page to actual application form
        # (handles "Apply Now" buttons on Phenom, Workday, Greenhouse, etc.)
        navigated = _navigate_to_application_form(page)

        # If Apply button opened a new tab, switch to it
        if navigated and hasattr(page, '_apply_new_page') and page._apply_new_page:
            old_page = page
            page = page._apply_new_page
            del old_page._apply_new_page
            logger.info("Switched to new tab for application form: %s", page.url)

        # 4.3 — Detect expired / filled / closed job pages
        # Runs AFTER Apply click + tab switch so we check the final page.
        # Some ATS pages (Phenom, iCIMS, Workday) show "job filled" with only
        # a notification signup form.  Detect this early to avoid filling the
        # wrong form.
        _EXPIRED_KEYWORDS = [
            "this job has been filled",
            "this position has been filled",
            "this job is no longer available",
            "this position is no longer available",
            "this job is no longer accepting applications",
            "no longer accepting applications",
            "this job posting has expired",
            "this job has expired",
            "this requisition has been closed",
            "this position has been closed",
            "job has been removed",
            "position has been removed",
            # NOTE: "get notified for similar jobs" intentionally excluded —
            # it appears on active Phenom/iCIMS pages too (as a sidebar widget).
        ]
        try:
            _page_text_lower = (
                page.evaluate("() => document.body ? document.body.innerText : ''") or ""
            ).lower()
            _expired_match = next(
                (kw for kw in _EXPIRED_KEYWORDS if kw in _page_text_lower), None
            )
            if _expired_match:
                logger.warning(
                    "Job appears expired/filled — detected '%s'. Aborting form fill.",
                    _expired_match,
                )
                status = "failed"
                final_url = page.url
                notes = (
                    f"Job appears expired or filled (detected: '{_expired_match}'). "
                    "No application form available."
                )
                close_browser(session_id)
                pw.stop()
                return {"status": status, "url": final_url, "notes": notes}
        except Exception as _exp_err:
            logger.debug("Expired-job check failed: %s", _exp_err)

        # 4.5 — Try uploading resume first
        # Many ATS forms auto-populate after resume upload
        resume_path = applicant_data.get("resume_path", "")
        uploaded = False
        if resume_path and Path(resume_path).exists():
            uploaded = _try_resume_upload_first(page, resume_path)
            if uploaded:
                logger.info("Resume uploaded — waiting for auto-population")
                page.wait_for_timeout(4000)  # wait for ATS to parse & populate

        # 4.5.1 — If automated upload failed, check for resume upload gate
        # (Phenom/SuccessFactors show mandatory upload before any form fields)
        if not uploaded and _detect_resume_upload_gate(page):
            logger.info("Resume upload gate detected — automated upload failed, requesting manual upload")
            gate_resolved = _handle_resume_upload_gate(page, session_id, resume_path)
            if gate_resolved:
                uploaded = True
                # Re-check if we're now past the upload gate
                # The page may have changed after upload
                _dismiss_cookie_banners(page)

        # 4.6 — Check for LinkedIn autofill option
        linkedin_used = False
        if not uploaded:
            linkedin_used = _try_linkedin_autofill(page, session_id)

        # 4.7 — Post-LinkedIn stabilization
        if linkedin_used:
            logger.info("Post-LinkedIn stabilization — waiting for page to settle")
            page.wait_for_timeout(3000)
            _dismiss_cookie_banners(page)
            try:
                page.wait_for_load_state("domcontentloaded", timeout=5000)
            except Exception:
                pass
            _navigate_to_application_form(page)

        _validation_retry_count = 0
        _failed_selectors.clear()
        _asked_fields.clear()

        # Track whether resume upload gate has been resolved (prevents infinite re-prompting)
        resume_gate_resolved = uploaded

        page_num = 0
        _page_hashes_seen: dict[str, int] = {}  # hash -> count
        _consecutive_no_progress = 0

        while page_num < max_pages:
            # 5. Extract form context from current page
            form_text, field_registry = _extract_form_context(page)

            if not form_text or "no form fields" in form_text.lower():
                logger.info("No form fields found on page %d — maybe done", page_num + 1)
                break

            # Detect stuck loops: if we've seen this exact form content before
            import hashlib
            _content_hash = hashlib.md5(form_text.encode()).hexdigest()[:12]
            _page_hashes_seen[_content_hash] = _page_hashes_seen.get(_content_hash, 0) + 1
            if _page_hashes_seen[_content_hash] > 3:
                logger.warning(
                    "Stuck loop detected — same page content seen %d times. "
                    "Attempting to click Next/Submit to advance.",
                    _page_hashes_seen[_content_hash],
                )
                # Try clicking Next directly to break out of the loop
                _stuck_next = page.locator(
                    "button:has-text('Next'), button:has-text('Continue'), "
                    "button:has-text('Save and Continue'), "
                    "button:has-text('Save & Continue'), "
                    "button:has-text('Proceed')"
                ).first
                try:
                    if _stuck_next.is_visible(timeout=2000):
                        _stuck_next.click(timeout=5000)
                        page.wait_for_timeout(2000)
                        page_num += 1
                        continue
                except Exception:
                    pass

                if _page_hashes_seen[_content_hash] > 5:
                    logger.error("Stuck loop — escalating to manual takeover after %d iterations", _page_hashes_seen[_content_hash])
                    from bot.telegram_bot import send_notification
                    send_notification("⚠️ <b>Browser agent stuck in a loop.</b>\nPlease take over manually via CDP.")
                    status = "partially_filled"
                    break

            # 6. LLM maps fields to applicant data
            field_mapping = _map_form_fields(form_text, applicant_data)

            if not field_mapping:
                logger.warning("LLM returned no field mapping for page %d", page_num + 1)
                # Fall through to blocker detection
            else:
                # 6b. Validate mapping -- catch any LLM-fabricated values
                field_mapping = _validate_mapping(field_mapping, applicant_data, field_registry)

                # 7. Fill the mapped fields (includes "ask" action for unknown fields)
                newly_filled = _fill_mapped_fields(page, field_mapping, field_registry, session_id=session_id)
                filled_fields.extend(newly_filled)

            # 8. Small delay for any client-side validation to trigger
            page.wait_for_timeout(1000)

            # 9. Check for blockers (Fix 3 + Fix 11 — returns list now)
            blockers = _detect_blockers(page)

            if blockers:
                # Separate validation errors from other blocker types
                validation_blockers = [
                    b for b in blockers
                    if b["type"] in ("missing_field", "validation_error")
                ]
                other_blockers = [
                    b for b in blockers
                    if b["type"] not in ("missing_field", "validation_error")
                ]

                # Handle non-validation blockers first (captcha, modal, resume upload, etc.)
                for blocker in other_blockers:
                    # Special handling for resume upload requirement
                    if blocker.get("type") == "resume_upload_required":
                        if resume_gate_resolved:
                            logger.info("Resume upload gate already resolved — skipping re-prompt")
                            continue
                        resume_path_for_gate = applicant_data.get("resume_path", "")
                        gate_ok = _handle_resume_upload_gate(page, session_id, resume_path_for_gate)
                        if gate_ok:
                            resume_gate_resolved = True
                            logger.info("Resume upload gate resolved via Telegram — re-scanning page")
                            break  # Page likely changed after upload; re-scan in next loop iteration
                        else:
                            logger.warning("Resume upload gate not resolved")

                    resolution = _handle_blocker(blocker, page, session_id)
                    logger.info("Blocker resolved: %s", resolution)
                    if resolution == "resolved_manually":
                        from bot.telegram_bot import ask_yes_no, send_photo, send_notification

                        hand_back = ask_yes_no("Did you finish the form manually?")
                        if hand_back:
                            status = "handed_off"
                            final_url = page.url
                            notes = f"Handed off to user. Filled {len(filled_fields)} fields."
                            break

                if status == "handed_off":
                    break

                # For validation errors: re-extract and re-fill before asking user (Fix 3)
                if validation_blockers:
                    _validation_retry_count += 1
                    logger.info(
                        "Found %d validation errors — retry cycle %d",
                        len(validation_blockers), _validation_retry_count,
                    )

                    # After retry limit, escalate to manual takeover
                    if _validation_retry_count > 4:
                        logger.warning("Validation retry limit reached (%d/4) — escalating to manual takeover", _validation_retry_count)
                        from bot.telegram_bot import send_notification, send_photo, request_interrupt
                        from apply.session_handoff import get_takeover_instructions

                        # Take screenshot first
                        ss_path = None
                        try:
                            from config import FORM_ANSWERS_PATH
                            import time as _time
                            _ts = int(_time.time())
                            ss_path = str(Path(FORM_ANSWERS_PATH).parent / f"takeover_{_ts}.png")
                            page.screenshot(path=ss_path)
                        except Exception:
                            ss_path = None

                        # Send context + screenshot
                        send_notification(
                            f"\u26a0\ufe0f <b>Form has {len(validation_blockers)} remaining errors "
                            f"after {_validation_retry_count} retry cycles.</b>\n\n"
                            "Please fix the errors in the browser, then choose an option below."
                        )
                        if ss_path:
                            send_photo(ss_path, caption="\u26a0\ufe0f <b>Manual Takeover Required</b>")

                        # Send CDP instructions
                        try:
                            cdp_instructions = get_takeover_instructions(session_id)
                            send_notification(f"<pre>{cdp_instructions}</pre>")
                        except Exception as _cdp_err:
                            logger.warning("Failed to get CDP instructions: %s", _cdp_err)

                        # Single blocking prompt with clear options
                        takeover_response = request_interrupt(
                            interrupt_type="form_takeover",
                            message=(
                                "\u2705 <b>When you're done, choose:</b>\n\n"
                                "\u2022 <b>Done, Continue</b> — I fixed the errors, let the agent resume filling\n"
                                "\u2022 <b>Done, I Completed It</b> — I finished the entire form myself"
                            ),
                            buttons=[
                                {"text": "🔄 Done, Continue", "callback_data": "takeover_continue"},
                                {"text": "✅ Done, I Completed It", "callback_data": "takeover_completed"},
                            ],
                            timeout=600,
                        )

                        if takeover_response == "takeover_continue":
                            # User fixed errors — reset retry counter, let agent resume
                            logger.info("User fixed errors manually — agent resuming fill loop")
                            _validation_retry_count = 0
                            continue
                        else:
                            # User completed form or timed out
                            status = "handed_off"
                            final_url = page.url
                            notes = f"Handed off after {_validation_retry_count} retry cycles. Filled {len(filled_fields)} fields."
                            break

                    # Re-extract to see current field values (some may have been filled)
                    form_text2, field_registry2 = _extract_form_context(page)
                    if form_text2:
                        # Build error context for the LLM
                        error_fields = [
                            b.get("field_label") or b.get("details", "unknown")
                            for b in validation_blockers
                        ]
                        error_context = (
                            "IMPORTANT: The following fields have validation errors "
                            "(required but empty): " + ", ".join(error_fields) + "\n"
                            "Focus on filling THESE fields. If the value is in APPLICANT DATA, "
                            "use action 'type' or 'select'. If NOT in APPLICANT DATA, "
                            "use action 'ask' so the user can provide the value. "
                            "NEVER guess or fabricate values.\n\n"
                        )

                        retry_mapping = _map_form_fields(
                            error_context + form_text2, applicant_data
                        )
                        if retry_mapping:
                            retry_mapping = _validate_mapping(retry_mapping, applicant_data, field_registry2)
                            retry_filled = _fill_mapped_fields(
                                page, retry_mapping, field_registry2,
                                session_id=session_id
                            )
                            filled_fields.extend(retry_filled)
                            logger.info("Retry filled %d additional fields", len(retry_filled))
                            page.wait_for_timeout(1000)

                            # Re-check blockers after retry
                            blockers2 = _detect_blockers(page)
                            if not blockers2:
                                # All resolved!
                                pass
                            else:
                                # Still have validation errors — ask user per-field
                                remaining_validation = [
                                    b for b in blockers2
                                    if b["type"] in ("missing_field", "validation_error")
                                ]
                                for vb in remaining_validation:
                                    sel = vb.get("field_selector", "")
                                    lbl = vb.get("field_label", "")
                                    if sel and _failed_selectors.get(sel, 0) >= 2:
                                        logger.warning("Skipping '%s' (selector '%s') — failed %d times", lbl, sel, _failed_selectors[sel])
                                        continue
                                    if lbl or sel:
                                        resolution = _handle_blocker(vb, page, session_id)
                                        logger.info("Validation blocker resolved: %s", resolution)
                                # Fall through to Next/Submit check instead of infinite re-check
                        else:
                            # Retry mapping failed — ask user per-field
                            for vb in validation_blockers:
                                sel = vb.get("field_selector", "")
                                lbl = vb.get("field_label", "")
                                if sel and _failed_selectors.get(sel, 0) >= 2:
                                    logger.warning("Skipping '%s' (selector '%s') — failed %d times", lbl, sel, _failed_selectors[sel])
                                    continue
                                if lbl or sel:
                                    resolution = _handle_blocker(vb, page, session_id)
                                    logger.info("Validation blocker resolved: %s", resolution)
                            # Fall through to Next/Submit check (was: continue)
                            pass
                    else:
                        # Re-extraction failed — ask user per-field
                        for vb in validation_blockers:
                            sel = vb.get("field_selector", "")
                            lbl = vb.get("field_label", "")
                            if sel and _failed_selectors.get(sel, 0) >= 2:
                                logger.warning("Skipping '%s' (selector '%s') — failed %d times", lbl, sel, _failed_selectors[sel])
                                continue
                            if lbl or sel:
                                resolution = _handle_blocker(vb, page, session_id)
                                logger.info("Validation blocker resolved: %s", resolution)
                        # Fall through to Next/Submit check

                # After handling blockers, fall through to Next/Submit check
                # instead of blindly continuing (which causes infinite loops)
                pass

            # 9.5. Before clicking Next, check for resume upload gate
            # (catches cases where the upload wasn't detected earlier,
            # e.g., after LinkedIn autofill lands on a page with upload requirement)
            if not resume_gate_resolved and _detect_resume_upload_gate(page):
                logger.info("Resume upload gate detected before Next — requesting manual upload")
                resume_path_for_gate = applicant_data.get("resume_path", "")
                gate_ok = _handle_resume_upload_gate(page, session_id, resume_path_for_gate)
                if gate_ok:
                    resume_gate_resolved = True
                    # Page likely changed after upload — restart this iteration
                    continue

            # 10. Look for a "Next" or "Continue" button
            # Includes "Save and Continue" / "Save & Continue" for Phenom ATS
            next_btn = page.locator(
                "button:has-text('Next'), button:has-text('Continue'), "
                "button:has-text('Save and Continue'), "
                "button:has-text('Save & Continue'), "
                "button:has-text('Save and Next'), "
                "button:has-text('Proceed'), "
                "input[type='submit']:has-text('Next'), "
                "a:has-text('Next'), a:has-text('Continue')"
            ).first

            try:
                if next_btn.is_visible(timeout=2000):
                    url_before_next = page.url
                    next_btn.click(timeout=5000)
                    try:
                        page.wait_for_load_state("networkidle", timeout=5000)
                    except Exception:
                        page.wait_for_load_state("domcontentloaded", timeout=5000)

                    # Check if page actually advanced (URL or content changed)
                    url_after_next = page.url
                    if url_before_next == url_after_next:
                        # Page didn't change — might be blocked by resume upload requirement
                        page.wait_for_timeout(1500)
                        if not resume_gate_resolved and _detect_resume_upload_gate(page):
                            logger.info("Next clicked but page didn't advance — resume upload gate detected")
                            resume_path_for_gate = applicant_data.get("resume_path", "")
                            gate_ok = _handle_resume_upload_gate(page, session_id, resume_path_for_gate)
                            if gate_ok:
                                continue  # Retry the page after upload
                            else:
                                logger.warning("Resume upload gate not resolved — continuing")

                    page_num += 1
                    continue  # process next page
            except Exception:
                pass  # No next button visible

            # 11. Look for a "Submit" button
            # NOTE: Do NOT include button:has-text('Apply') here — it conflicts
            # with "Apply Now" buttons on JD pages. Use Submit-specific selectors only.
            submit_btn = page.locator(
                "button:has-text('Submit'), button:has-text('Submit Application'), "
                "input[type='submit'], button[type='submit'], "
                "div[role='button']:has-text('Submit'), "
                "[role='button']:has-text('Submit')"
            ).first

            try:
                if submit_btn.is_visible(timeout=2000):
                    if not submit:
                        # submit=False: stop before clicking submit (Gate 2 review)
                        logger.info("Submit button found. submit=False — stopping for review.")
                        status = "partially_filled"
                        break

                    # DON'T auto-submit — flag for user confirmation
                    logger.info("Submit button found. NOT auto-submitting.")
                    from bot.telegram_bot import ask_yes_no, send_photo, send_notification

                    confirm = ask_yes_no("⚠️ <b>Submit button found.</b>\n\nReview the form. Ready to submit?")
                    if confirm:
                        submit_btn.click(timeout=5000)
                        try:
                            page.wait_for_load_state("networkidle", timeout=8000)
                        except Exception:
                            page.wait_for_load_state("domcontentloaded", timeout=8000)

                        # Wait for confirmation page to render
                        page.wait_for_timeout(3000)

                        # Take screenshot of the post-submit page
                        try:
                            from config import OUTPUT_BASE_DIR
                            post_submit_ss = str(
                                Path(OUTPUT_BASE_DIR) / f"post_submit_{int(time.time())}.png"
                            )
                            page.screenshot(path=post_submit_ss)
                            logger.info("Post-submit screenshot: %s", post_submit_ss)

                            # Detect success indicators on the confirmation page
                            page_text = page.evaluate("() => document.body ? document.body.innerText : ''") or ""
                            page_text_lower = page_text.lower()
                            success_keywords = [
                                "thank you", "thanks for applying", "application submitted",
                                "successfully submitted", "application received",
                                "application complete", "we have received",
                                "your application has been", "congratulations",
                            ]
                            is_success = any(kw in page_text_lower for kw in success_keywords)

                            if is_success:
                                logger.info("Post-submit page shows success confirmation")
                                send_photo(post_submit_ss,
                                           caption="✅ <b>Application Submitted Successfully!</b>\n\n"
                                                   "The confirmation page indicates your application was received.")
                            else:
                                logger.info("Post-submit page — no clear success indicator, assuming submitted")
                                send_photo(post_submit_ss,
                                           caption="✉️ <b>Form Submitted</b>\n\n"
                                                   "Submit was clicked. Please verify from the screenshot "
                                                   "that the application went through.")

                        except Exception as ss_err:
                            logger.warning("Could not capture post-submit screenshot: %s", ss_err)

                        status = "submitted"
                        # Give user a moment to see the confirmation before browser closes
                        page.wait_for_timeout(3000)
                    else:
                        status = "partially_filled"
                    break
            except Exception:
                pass  # No submit button visible

            # No next or submit button found — might be done or stuck
            # Instead of silently closing, offer the user manual control
            logger.warning("No navigation buttons found on page %d — requesting manual takeover", page_num + 1)
            try:
                from bot.telegram_bot import send_photo, request_interrupt, send_notification
                from apply.session_handoff import get_takeover_instructions
                from config import OUTPUT_BASE_DIR

                # Take screenshot so user can see current state
                no_nav_ss = str(
                    Path(OUTPUT_BASE_DIR) / f"no_nav_page_{page_num + 1}_{int(time.time())}.png"
                )
                page.screenshot(path=no_nav_ss)
                send_photo(
                    no_nav_ss,
                    caption=(
                        "\u26a0\ufe0f <b>No Submit/Next button found</b>\n\n"
                        "The form may need manual submission or there may be "
                        "an issue. Take manual control to review and submit."
                    ),
                )

                # Send CDP takeover instructions
                try:
                    cdp_instructions = get_takeover_instructions(session_id)
                    send_notification(f"<pre>{cdp_instructions}</pre>")
                except Exception as _cdp_err:
                    logger.warning("Failed to get CDP instructions: %s", _cdp_err)

                # Wait for user to finish
                response = request_interrupt(
                    interrupt_type="no_nav_handoff",
                    message=(
                        "\u26a0\ufe0f <b>Manual control active.</b>\n\n"
                        "Submit the form or fix any issues in the browser, "
                        "then tap <b>Done</b>."
                    ),
                    buttons=[
                        {"text": "\u2705 Done", "callback_data": "blocker_done"},
                    ],
                    timeout=600,
                )
                if response is not None:
                    status = "handed_off"
                    notes = f"Handed off to user (no nav buttons on page {page_num + 1}). User confirmed done."
                else:
                    status = "partially_filled"
                    notes = f"Handed off to user (no nav buttons on page {page_num + 1}). Timed out waiting for confirmation."
            except Exception as handoff_err:
                logger.error("Failed to request handoff: %s", handoff_err)
                status = "partially_filled"
                notes = f"No navigation buttons found on page {page_num + 1}. Handoff failed: {handoff_err}"
            break
        if page_num >= max_pages:
            status = "partially_filled"  # hit max_pages

        if status == "failed":
            # We exited the loop without setting status — means we broke out normally
            status = "partially_filled"

        final_url = page.url
        if not notes:
            notes = f"Filled {len(filled_fields)} fields: {', '.join(filled_fields[:10])}"

    except Exception as exc:
        logger.error("Form filling failed: %s", exc, exc_info=True)
        # Notify user via Telegram and offer manual takeover before closing
        try:
            from bot.telegram_bot import send_notification, send_photo, request_interrupt
            from apply.session_handoff import get_takeover_instructions
            from config import OUTPUT_BASE_DIR

            # Try to capture a screenshot of the error state
            error_ss = None
            try:
                error_ss = str(
                    Path(OUTPUT_BASE_DIR) / f"error_state_{int(time.time())}.png"
                )
                page.screenshot(path=error_ss)
            except Exception:
                error_ss = None

            error_msg = str(exc)[:200] if str(exc) else "Unknown error"
            caption = (
                "\u274c <b>Form filling encountered an error</b>\n\n"
                f"<code>{error_msg}</code>\n\n"
                "You can take manual control to finish, or let it close."
            )
            if error_ss:
                send_photo(error_ss, caption=caption)
            else:
                send_notification(caption)

            # Send CDP takeover instructions
            try:
                cdp_instructions = get_takeover_instructions(session_id)
                send_notification(f"<pre>{cdp_instructions}</pre>")
            except Exception:
                pass

            response = request_interrupt(
                interrupt_type="error_handoff",
                message=(
                    "\u26a0\ufe0f <b>Take manual control?</b>\n\n"
                    "Fix the issue in the browser and tap <b>Done</b>, "
                    "or tap <b>Close</b> to end the session."
                ),
                buttons=[
                    {"text": "\u2705 Done", "callback_data": "blocker_done"},
                    {"text": "\u274c Close", "callback_data": "blocker_close"},
                ],
                timeout=600,
            )
            if response and response != "blocker_close":
                status = "handed_off"
                notes = f"Error occurred ({error_msg}), user took manual control."
            else:
                status = "failed"
                notes = f"Error: {exc}"
        except Exception as notify_err:
            logger.warning("Could not notify user about error: %s", notify_err)
            status = "failed"
            notes = f"Error: {exc}"
        final_url = url

    finally:
        # Cleanup
        try:
            browser.close()
        except Exception:
            pass
        pw.stop()
        close_browser(session_id)

    result = {"status": status, "url": final_url, "notes": notes}
    logger.info("Form fill result: %s", result)
    return result


def submit_form(session_id: str | None = None) -> dict:
    """Click the submit button on an already-filled ATS form.

    Called by the pipeline's execute_send node after Gate 2 approval.
    Connects to the existing browser session (launched by fill_form with
    submit=False) and clicks the submit button.

    Parameters
    ----------
    session_id : str or None
        The browser session ID. If None, uses the default session.

    Returns
    -------
    dict
        Result with keys: status, url, notes.
    """
    from apply.session_handoff import get_cdp_url, close_browser
    from playwright.sync_api import sync_playwright

    session_id = session_id or "default"
    status = "failed"
    final_url = ""
    notes = ""

    try:
        cdp_url = get_cdp_url(session_id)
        if not cdp_url:
            return {
                "status": "failed",
                "url": "",
                "notes": "No active browser session found for submit.",
            }

        pw = sync_playwright().start()
        browser = pw.chromium.connect_over_cdp(cdp_url)
        context = browser.contexts[0] if browser.contexts else browser.new_context()
        page = context.pages[0] if context.pages else context.new_page()

        # Look for submit button
        submit_btn = page.locator(
            "button:has-text('Submit'), button:has-text('Submit Application'), "
            "input[type='submit'], button[type='submit']"
        ).first

        try:
            if submit_btn.is_visible(timeout=3000):
                logger.info("Clicking submit button (Gate 2 approved).")
                submit_btn.click()
                try:
                    page.wait_for_load_state("networkidle", timeout=5000)
                except Exception:
                    page.wait_for_load_state("domcontentloaded", timeout=5000)
                status = "submitted"
                final_url = page.url
                notes = "Form submitted after Gate 2 approval."
            else:
                status = "failed"
                final_url = page.url
                notes = "Submit button not visible."
        except Exception as exc:
            status = "failed"
            final_url = page.url
            notes = f"Submit click failed: {exc}"

        browser.close()
        pw.stop()

    except Exception as exc:
        logger.error("submit_form failed: %s", exc, exc_info=True)
        notes = f"Error: {exc}"

    finally:
        try:
            close_browser(session_id)
        except Exception:
            pass

    result = {"status": status, "url": final_url, "notes": notes}
    logger.info("submit_form result: %s", result)
    return result
