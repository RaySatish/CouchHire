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

    function getLabel(el) {
        // 1. <label for="id">
        if (el.id) {
            const label = document.querySelector('label[for="' + el.id + '"]');
            if (label && label.textContent.trim()) return label.textContent.trim();
        }
        // 2. aria-label
        if (el.getAttribute('aria-label')) return el.getAttribute('aria-label').trim();
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
        // Positional fallback
        const tag = el.tagName.toLowerCase();
        const parent = el.parentElement;
        if (parent) {
            const siblings = Array.from(parent.querySelectorAll(':scope > ' + tag));
            const idx = siblings.indexOf(el) + 1;
            if (idx > 0) {
                const parentSel = parent.id ? '#' + CSS.escape(parent.id) : tag;
                return parentSel + ' > ' + tag + ':nth-of-type(' + idx + ')';
            }
        }
        return tag;
    }

    function processElement(el, iframeSel) {
        const tag = el.tagName.toLowerCase();
        const type = el.type ? el.type.toLowerCase() : tag;

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

    // Process main document
    const selectors = 'input, select, textarea, button[type="submit"], input[type="submit"]';
    document.querySelectorAll(selectors).forEach(el => {
        const f = processElement(el, null);
        if (f) fields.push(f);
    });

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
        label = field.get("label", "")
        required = field.get("required", False)
        value = field.get("value", "")
        options = field.get("options", [])
        accept = field.get("accept", "")

        # Build human-readable line
        req_str = ", required" if required else ""
        line = f'{index}. [{ftype}{req_str}] "{label}"'

        if ftype == "select" and options:
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

_MAP_SYSTEM_PROMPT = """You are a form-filling assistant. Given a list of numbered form fields and applicant data, return a JSON array mapping each field to its value.

Rules:
- Use the field INDEX number to identify each field (matches the number in the form fields list)
- Only fill fields where you have the data. For fields with no matching data, set action to "skip"
- For file uploads, set action to "upload" and value to the file path from applicant data
- For dropdowns/selects, set action to "select" and value to the EXACT option text
- For text/email/tel fields, set action to "type" and value to the text to enter
- For checkboxes, set action to "click" if it should be checked
- For submit buttons, set action to "skip"
- Return ONLY a valid JSON array, no markdown fences, no explanation"""


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
        logger.warning("LLM response parsed as %s, expected list", type(parsed).__name__)
        return None
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Fill mapped fields
# ---------------------------------------------------------------------------

def _fill_mapped_fields(
    page, field_mapping: list[dict], field_registry: list[dict]
) -> list[str]:
    """Execute the field mapping by looking up real selectors from field_registry.

    Returns a list of field labels that were successfully filled.
    """
    # Build index lookup for field_registry
    registry_by_index: dict[int, dict] = {
        entry["index"]: entry for entry in field_registry
    }

    filled_labels: list[str] = []

    for mapping in field_mapping:
        action = mapping.get("action", "skip")
        if action == "skip":
            continue

        field_index = mapping.get("field_index")
        value = mapping.get("value", "")
        label = mapping.get("field_label", f"field_{field_index}")

        if field_index is None:
            logger.warning("Mapping entry missing field_index: %s", mapping)
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

            if action == "type":
                context.locator(selector).wait_for(timeout=5000)
                context.locator(selector).fill(value)
                filled_labels.append(label)

            elif action == "select":
                context.locator(selector).wait_for(timeout=5000)
                context.locator(selector).select_option(label=value)
                filled_labels.append(label)

            elif action == "upload":
                context.locator(selector).set_input_files(value)
                filled_labels.append(label)

            elif action == "click":
                context.locator(selector).click()
                filled_labels.append(label)

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

def _detect_blockers(page) -> dict | None:
    """Scan the page for anything blocking progress.

    Returns None if no blockers found, or a dict describing the blocker.
    """
    from config import FORM_ANSWERS_PATH

    output_dir = Path(FORM_ANSWERS_PATH).parent

    try:
        blockers = page.evaluate("""
        () => {
            const results = [];

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

            // 2. Modal popups
            const dialogs = document.querySelectorAll('[role="dialog"], [role="alertdialog"]');
            for (const d of dialogs) {
                const rect = d.getBoundingClientRect();
                if (rect.width > 0 && rect.height > 0) {
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
                        fieldSelector = nearInput.id ? '#' + nearInput.id : (nearInput.name ? nearInput.tagName.toLowerCase() + '[name="' + nearInput.name + '"]' : null);
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

            return results.length > 0 ? results : null;
        }
        """)
    except Exception as exc:
        logger.warning("Blocker detection JS failed: %s", exc)
        return None

    if not blockers:
        return None

    # Return the first (highest-priority) blocker
    blocker = blockers[0]

    # Take screenshot
    timestamp = int(time.time())
    output_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = str(output_dir / f"blocker_{timestamp}.png")
    try:
        page.screenshot(path=screenshot_path)
    except Exception as exc:
        logger.warning("Failed to take blocker screenshot: %s", exc)
        screenshot_path = ""

    blocker["screenshot_path"] = screenshot_path
    return blocker


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

    # --- Simple blockers: missing_field, validation_error ---
    if blocker_type in ("missing_field", "validation_error"):
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
                            page.locator(field_selector).fill(stored_value)
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

        # Not in memory — ask the user via Telegram
        from bot.telegram_bot import ask_user

        question = f"🚫 <b>BLOCKER:</b> {blocker_type}\n{details}"
        if field_label:
            question += (
                f"\n\n<b>Field:</b> {field_label}\n"
                "Reply with the value to fill, or type <code>takeover</code> for manual control."
            )
        else:
            question += "\n\nReply with the value, or type <code>takeover</code> for manual control."

        response = ask_user(question, screenshot_path=screenshot_path if screenshot_path else None)

        if response == "__timeout__":
            logger.warning("Telegram response timed out for blocker: %s", details)
            return _handle_complex_blocker(blocker, session_id)

        if response.lower() == "takeover":
            # Fall through to complex handler
            return _handle_complex_blocker(blocker, session_id)

        # Fill the value
        if field_selector and response:
            try:
                page.locator(field_selector).fill(response)
            except Exception as exc:
                logger.warning("Failed to fill user response for '%s': %s", field_label, exc)

        # Save to form_answers.json
        if field_label and response:
            _save_form_answer(field_label, response)

        return "resolved"

    # --- Complex blockers: captcha, modal_popup, unknown ---
    return _handle_complex_blocker(blocker, session_id)


def _handle_complex_blocker(blocker: dict, session_id: str) -> str:
    """Handle a complex blocker that requires manual CDP takeover."""
    blocker_type = blocker.get("type", "unknown")
    details = blocker.get("details", "")
    screenshot_path = blocker.get("screenshot_path", "")

    from bot.telegram_bot import send_takeover_instructions, ask_yes_no

    send_takeover_instructions(session_id, screenshot_path=screenshot_path if screenshot_path else None)

    # Block until user taps "Done" or replies
    resolved = ask_yes_no("Have you resolved the issue? Tap ✅ when ready to continue.")
    if not resolved:
        logger.warning("User did not confirm resolution for blocker: %s", details)
    return "resolved_manually"


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def fill_form(url: str, documents: dict) -> dict:
    """Fill an ATS application form using LLM-assisted field mapping.

    Semi-autonomous: fills what it can, asks the user (via Telegram stubs)
    for anything it can't resolve.

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
        page.goto(url, wait_until="networkidle", timeout=30000)

        filled_fields: list[str] = []
        max_pages = 10  # safety limit for multi-page forms

        for page_num in range(max_pages):
            # 5. Extract form context from current page
            form_text, field_registry = _extract_form_context(page)

            if not form_text or "no form fields" in form_text.lower():
                logger.info("No form fields found on page %d — maybe done", page_num + 1)
                break

            # 6. LLM maps fields to applicant data
            field_mapping = _map_form_fields(form_text, applicant_data)

            if not field_mapping:
                logger.warning("LLM returned no field mapping for page %d", page_num + 1)
                # Fall through to blocker detection
            else:
                # 7. Fill the mapped fields
                newly_filled = _fill_mapped_fields(page, field_mapping, field_registry)
                filled_fields.extend(newly_filled)

            # 8. Small delay for any client-side validation to trigger
            page.wait_for_timeout(1000)

            # 9. Check for blockers
            blocker = _detect_blockers(page)

            if blocker:
                resolution = _handle_blocker(blocker, page, session_id)
                logger.info("Blocker resolved: %s", resolution)
                if resolution == "resolved_manually":
                    # User took over via CDP — check if they want to continue or hand back
                    from bot.telegram_bot import ask_yes_no

                    hand_back = ask_yes_no("Did you finish the form manually?")
                    if hand_back:
                        status = "handed_off"
                        final_url = page.url
                        notes = (
                            f"Handed off to user after {blocker['type']} blocker. "
                            f"Filled {len(filled_fields)} fields before handoff."
                        )
                        break
                # After resolving, re-check the page (loop continues)
                continue

            # 10. Look for a "Next" or "Continue" button
            next_btn = page.locator(
                "button:has-text('Next'), button:has-text('Continue'), "
                "input[type='submit']:has-text('Next'), a:has-text('Next')"
            ).first

            try:
                if next_btn.is_visible(timeout=2000):
                    next_btn.click()
                    page.wait_for_load_state("networkidle", timeout=10000)
                    continue  # process next page
            except Exception:
                pass  # No next button visible

            # 11. Look for a "Submit" button
            submit_btn = page.locator(
                "button:has-text('Submit'), button:has-text('Apply'), "
                "input[type='submit'], button[type='submit']"
            ).first

            try:
                if submit_btn.is_visible(timeout=2000):
                    # DON'T auto-submit — flag for user confirmation
                    logger.info("Submit button found. NOT auto-submitting.")
                    from bot.telegram_bot import ask_yes_no

                    confirm = ask_yes_no("⚠️ <b>Submit button found.</b>\n\nReview the form. Ready to submit?")
                    if confirm:
                        submit_btn.click()
                        page.wait_for_load_state("networkidle", timeout=15000)
                        status = "submitted"
                    else:
                        status = "partially_filled"
                    break
            except Exception:
                pass  # No submit button visible

            # No next or submit button found — might be done or stuck
            logger.info("No navigation buttons found on page %d", page_num + 1)
            break
        else:
            status = "partially_filled"  # hit max_pages

        if status == "failed":
            # We exited the loop without setting status — means we broke out normally
            status = "partially_filled"

        final_url = page.url
        if not notes:
            notes = f"Filled {len(filled_fields)} fields: {', '.join(filled_fields[:10])}"

    except Exception as exc:
        logger.error("Form filling failed: %s", exc, exc_info=True)
        status = "failed"
        final_url = url
        notes = f"Error: {exc}"

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
