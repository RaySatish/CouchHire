"""CouchHire Telegram bot — notifications, /outcome command, /search command, and browser agent interrupts.

Two modes of operation:

Mode 1 — Sync notification senders (called by pipeline, browser_agent, etc.):
    send_notification(), send_photo(), send_job_card(), send_draft_ready(),
    send_form_started(), send_manual_notice(), send_sent_confirmation()

Mode 2 — Long-running async bot (for /outcome, /search commands + interrupt replies):
    start_bot()  — blocking polling loop

Browser agent interrupt system:
    ask_user(), ask_yes_no(), send_takeover_instructions()

Job discovery:
    /search command triggers JobSpy job search → filter → present with [Apply] buttons

This file uses async handlers because python-telegram-bot requires it.
This is one of the two allowed async exceptions per CLAUDE.md.
"""

# ---------------------------------------------------------------------------
# Prevent double-import when run as __main__
# When this file runs as `python -m bot.telegram_bot`, Python loads it as
# __main__. Later, `from bot.telegram_bot import ...` would create a SECOND
# copy with separate globals (_pending_interrupt, etc.). This ensures only
# one copy exists.
# ---------------------------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path

# Ensure project root is on sys.path so `from config import ...` works
# when this file is run directly as `python bot/telegram_bot.py`.
_project_root = str(_Path(__file__).resolve().parent.parent)
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

if __name__ == "__main__" and "bot.telegram_bot" not in _sys.modules:
    _sys.modules["bot.telegram_bot"] = _sys.modules[__name__]



import asyncio
import re
import concurrent.futures
import html
import logging
import threading
from pathlib import Path

import telegram
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state — lazy bot singleton
# ---------------------------------------------------------------------------
_bot_instance: telegram.Bot | None = None

# ---------------------------------------------------------------------------
# Module-level state — interrupt system (thread-safe)
# ---------------------------------------------------------------------------
_interrupt_lock = threading.Lock()
_pending_interrupt: dict | None = None

# Valid outcome labels (must match DB CHECK constraint)
_VALID_OUTCOMES = frozenset({"interview", "rejected", "no_response", "offer", "withdrawn"})

# ---------------------------------------------------------------------------
# Module-level state — search results cache for callback handling
# ---------------------------------------------------------------------------
_search_results_cache: list[dict] = []

# ---------------------------------------------------------------------------
# Persistent event loop — keeps telegram.Bot's httpx client alive across calls
# ---------------------------------------------------------------------------
_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    """Return a persistent background event loop (created once, reused)."""
    global _loop
    if _loop is None or _loop.is_closed():
        with _loop_lock:
            if _loop is None or _loop.is_closed():
                _loop = asyncio.new_event_loop()
                t = threading.Thread(target=_loop.run_forever, daemon=True)
                t.start()
    return _loop


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_bot() -> telegram.Bot:
    """Return a cached telegram.Bot instance, creating it on first call."""
    global _bot_instance
    if _bot_instance is None:
        from config import TELEGRAM_BOT_TOKEN
        if not TELEGRAM_BOT_TOKEN:
            raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
        _bot_instance = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
    return _bot_instance


def _get_chat_id() -> int | str:
    """Return TELEGRAM_CHAT_ID as an integer (numeric) or string (@username)."""
    from config import TELEGRAM_CHAT_ID
    if not TELEGRAM_CHAT_ID:
        raise RuntimeError("TELEGRAM_CHAT_ID not set")
    # Numeric chat IDs (including negative for groups) → int
    # @username strings → keep as str (Telegram API accepts both)
    stripped = TELEGRAM_CHAT_ID.lstrip("-")
    if stripped.isdigit():
        return int(TELEGRAM_CHAT_ID)
    return TELEGRAM_CHAT_ID


def _run_async(coro):
    """Run an async coroutine from sync code.

    Uses a persistent background event loop to keep the telegram.Bot's
    internal httpx client alive across multiple calls.
    """
    loop = _get_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=30)



def _escape_html(text: str) -> str:
    """Escape HTML special characters in user-provided text."""
    return html.escape(text or "Unknown")


# ---------------------------------------------------------------------------
# Mode 1 — Sync notification senders
# ---------------------------------------------------------------------------

def send_notification(text: str, reply_markup=None) -> None:
    """Send an HTML-formatted text message to the configured Telegram chat.

    All other send_* functions delegate to this. On failure, logs a warning
    and does NOT crash — notifications are best-effort.
    """
    try:
        bot = _get_bot()
        chat_id = _get_chat_id()
    except RuntimeError as exc:
        logger.warning("Cannot send Telegram notification: %s", exc)
        return

    async def _send():
        await bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode="HTML",
            reply_markup=reply_markup,
        )

    try:
        _run_async(_send())
        logger.info("Telegram notification sent (%d chars)", len(text))
    except telegram.error.TelegramError as exc:
        logger.warning("Failed to send Telegram notification: %s", exc)
    except Exception as exc:
        logger.warning("Unexpected error sending Telegram notification: %s", exc)


def send_photo(photo_path: str, caption: str = "") -> None:
    """Send a screenshot image to Telegram with an optional HTML caption.

    On failure, falls back to sending the caption as text.
    """
    try:
        bot = _get_bot()
        chat_id = _get_chat_id()
    except RuntimeError as exc:
        logger.warning("Cannot send Telegram photo: %s", exc)
        return

    photo_file = Path(photo_path)

    async def _send():
        with photo_file.open("rb") as f:
            await bot.send_photo(
                chat_id=chat_id,
                photo=f,
                caption=caption,
                parse_mode="HTML",
            )

    try:
        _run_async(_send())
        logger.info("Telegram photo sent: %s", photo_path)
    except (telegram.error.TelegramError, OSError) as exc:
        logger.warning("Failed to send Telegram photo (%s): %s", photo_path, exc)
        if caption:
            send_notification(caption)
    except Exception as exc:
        logger.warning("Unexpected error sending Telegram photo: %s", exc)
        if caption:
            send_notification(caption)




def send_document(pdf_path: str, caption: str = "") -> None:
    """Send a PDF document to the user via Telegram (sync helper).

    Same pattern as send_notification() — best-effort, never crashes.
    """
    try:
        bot = _get_bot()
        chat_id = _get_chat_id()
        file_path = Path(pdf_path)
        if not file_path.exists():
            logger.warning("send_document: file not found: %s", pdf_path)
            send_notification(f"⚠️ File not found: {file_path.name}")
            return

        async def _send():
            with open(file_path, "rb") as f:
                await bot.send_document(
                    chat_id=chat_id,
                    document=f,
                    filename=file_path.name,
                    caption=caption[:1024] if caption else None,
                    parse_mode="HTML",
                )

        _run_async(_send())
        logger.info("Sent document: %s", file_path.name)
    except Exception as exc:
        logger.error("send_document failed: %s", exc, exc_info=True)

def send_job_card(company: str, role: str, score: float, route: str) -> None:
    """Send a job card notification when the pipeline starts processing."""
    safe_company = _escape_html(company)
    safe_role = _escape_html(role)
    safe_route = _escape_html(route)

    text = (
        f"📋 <b>New Application</b>\n"
        f"\n"
        f"<b>Company:</b> {safe_company}\n"
        f"<b>Role:</b> {safe_role}\n"
        f"<b>Match Score:</b> {score:.0f}%\n"
        f"<b>Route:</b> {safe_route}\n"
        f"\n"
        f"Processing..."
    )
    send_notification(text)


def send_draft_ready(company: str, role: str, draft_url: str) -> None:
    """Send a 'draft ready' notification with a Review Draft inline button."""
    safe_company = _escape_html(company)
    safe_role = _escape_html(role)

    text = (
        f"✅ <b>Draft Ready</b>\n"
        f"\n"
        f"Draft ready for: <b>{safe_company} — {safe_role}</b>.\n"
        f"Open Gmail to review and send."
    )
    markup = InlineKeyboardMarkup([
        [InlineKeyboardButton("📧 Review Draft", url=draft_url)]
    ])
    send_notification(text, reply_markup=markup)


def send_sent_confirmation(company: str, role: str, sent_url: str) -> None:
    """Send a confirmation that the application email was sent successfully."""
    safe_company = _escape_html(company)
    safe_role = _escape_html(role)

    text = (
        f"✉️ <b>Application Sent!</b>\n"
        f"\n"
        f"Your application for <b>{safe_company} — {safe_role}</b> "
        f"was sent successfully."
    )
    markup = InlineKeyboardMarkup([
        [InlineKeyboardButton("📨 View Sent Email", url=sent_url)]
    ])
    send_notification(text, reply_markup=markup)


def send_form_started(company: str, role: str) -> None:
    """Send a notification when the browser agent starts form filling."""
    safe_company = _escape_html(company)
    safe_role = _escape_html(role)

    text = (
        f"🌐 <b>Form Application Started</b>\n"
        f"\n"
        f"Form application started for: <b>{safe_company} — {safe_role}</b>.\n"
        f"You'll receive updates as the form is filled."
    )
    send_notification(text)


def send_manual_notice(company: str, role: str, draft_url: str | None = None) -> None:
    """Send a notification when no apply method was found (manual route)."""
    safe_company = _escape_html(company)
    safe_role = _escape_html(role)

    text = (
        f"⚠️ <b>Manual Application Required</b>\n"
        f"\n"
        f"No apply method found for: <b>{safe_company} — {safe_role}</b>.\n"
        f"Resume and cover letter have been generated. Apply manually."
    )
    markup = None
    if draft_url:
        markup = InlineKeyboardMarkup([
            [InlineKeyboardButton("📧 Review Draft", url=draft_url)]
        ])
    send_notification(text, reply_markup=markup)


# ---------------------------------------------------------------------------
# Browser agent interrupt system
# ---------------------------------------------------------------------------

def ask_user(question: str, screenshot_path: str | None = None) -> str:
    """Send a question to Telegram and block until the user replies.

    Called by browser_agent.py (synchronous). Uses threading.Event to block.
    Returns the user's reply text, or '__timeout__' after 5 minutes.
    """
    global _pending_interrupt

    with _interrupt_lock:
        _pending_interrupt = {"event": threading.Event(), "response": None}

    # Send the question
    if screenshot_path:
        send_photo(screenshot_path, caption=question)
    else:
        send_notification(question)

    # Block until user replies or timeout
    event = _pending_interrupt["event"]
    got_response = event.wait(timeout=300)  # 5 minutes

    with _interrupt_lock:
        if not got_response:
            _pending_interrupt = None
            logger.warning("ask_user timed out after 300s")
            return "__timeout__"

        response = _pending_interrupt["response"]
        _pending_interrupt = None

    return response if response is not None else "__timeout__"


def ask_yes_no(question: str) -> bool:
    """Send a yes/no question with inline buttons and block until answered.

    Returns True for 'yes', False for 'no' or timeout.
    """
    global _pending_interrupt

    with _interrupt_lock:
        _pending_interrupt = {"event": threading.Event(), "response": None}

    markup = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Yes", callback_data="yes"),
            InlineKeyboardButton("❌ No", callback_data="no"),
        ]
    ])
    send_notification(question, reply_markup=markup)

    # Block until user taps a button or timeout
    event = _pending_interrupt["event"]
    got_response = event.wait(timeout=300)

    with _interrupt_lock:
        if not got_response:
            _pending_interrupt = None
            logger.warning("ask_yes_no timed out after 300s")
            return False

        response = _pending_interrupt["response"]
        _pending_interrupt = None

    return response == "yes"


def send_takeover_instructions(session_id: str, screenshot_path: str | None = None) -> None:
    """Send CDP takeover instructions to Telegram for complex blockers.

    Does NOT block — the caller should follow up with ask_yes_no() to wait
    for the user to confirm they've resolved the issue.
    """
    from apply.session_handoff import get_takeover_instructions

    try:
        instructions = get_takeover_instructions(session_id)
    except RuntimeError as exc:
        instructions = f"(Could not get takeover instructions: {exc})"
        logger.warning("Failed to get takeover instructions for session '%s': %s", session_id, exc)

    if screenshot_path:
        send_photo(screenshot_path, caption="⚠️ <b>Manual Takeover Required</b>")

    send_notification(f"<pre>{_escape_html(instructions)}</pre>")

    # Send a "Done" button (non-blocking — caller uses ask_yes_no to block)
    markup = InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Done", callback_data="yes")]
    ])
    send_notification("Tap <b>Done</b> when you've resolved the issue.", reply_markup=markup)




def request_interrupt(
    interrupt_type: str,
    message: str,
    buttons: list[dict] | None = None,
    timeout: int = 300,
) -> str | None:
    """Block the calling thread until the user responds via Telegram.

    Used by pipeline.py for Gate 1 (resume review) and Gate 2 (send review).

    Parameters
    ----------
    interrupt_type : str
        Identifier like "resume_review" or "send_review".
    message : str
        HTML-formatted message to display.
    buttons : list[dict] or None
        List of button dicts with "text" and "callback_data" keys.
        If None, waits for a free-text reply.
    timeout : int
        Seconds to wait before timing out (default 300).

    Returns
    -------
    str or None
        The callback_data of the tapped button, or the text reply.
        None if timed out.
    """
    global _pending_interrupt

    with _interrupt_lock:
        _pending_interrupt = {
            "event": threading.Event(),
            "response": None,
            "type": interrupt_type,
        }

    print(f"[DEBUG-INTERRUPT] _pending_interrupt SET for type={interrupt_type}, id={id(_pending_interrupt)}, module={__name__}")

    if buttons:
        keyboard_buttons = [
            [InlineKeyboardButton(text=btn["text"], callback_data=btn["callback_data"])]
            for btn in buttons
        ]
        markup = InlineKeyboardMarkup(keyboard_buttons)
        send_notification(message, reply_markup=markup)
    else:
        send_notification(message)

    event = _pending_interrupt["event"]
    print(f"[DEBUG-INTERRUPT] Waiting on event (timeout={timeout}s)...")
    event.wait(timeout=timeout)
    print(f"[DEBUG-INTERRUPT] event.wait() returned. is_set={event.is_set()}")

    with _interrupt_lock:
        if not event.is_set():
            _pending_interrupt = None
            logger.warning("request_interrupt(%s) timed out after %ds", interrupt_type, timeout)
            return None

        response = _pending_interrupt["response"]
        _pending_interrupt = None

    logger.info("request_interrupt(%s) got response: %s", interrupt_type, response)
    return response


# ---------------------------------------------------------------------------
# Mode 2 — Async handlers for the long-running bot
# ---------------------------------------------------------------------------

async def _handle_start(update: Update, context) -> None:
    """Handle the /start command."""
    text = (
        "👋 <b>CouchHire Bot Active</b>\n"
        "\n"
        "Commands:\n"
        "/apply &lt;JD or URL&gt; — Start a new application\n"
        "/search &lt;query&gt; [, location] [, job_type] — Search job boards for jobs\n"
        "/outcome &lt;id&gt; &lt;label&gt; — Label application outcome\n"
        "/status — Check bot status\n"
        "\n"
        "Labels: interview, rejected, no_response, offer, withdrawn"
    )
    await update.message.reply_text(text, parse_mode="HTML")


async def _handle_outcome(update: Update, context) -> None:
    """Handle the /outcome command: /outcome <application_id> <label>."""
    if not context.args or len(context.args) != 2:
        await update.message.reply_text(
            "Usage: /outcome &lt;application_id&gt; &lt;label&gt;\n\n"
            "Labels: interview, rejected, no_response, offer, withdrawn",
            parse_mode="HTML",
        )
        return

    application_id = context.args[0]
    label = context.args[1].lower()

    if label not in _VALID_OUTCOMES:
        await update.message.reply_text(
            f"\u274c Invalid label: <code>{_escape_html(label)}</code>\n\n"
            f"Valid labels: interview, rejected, no_response, offer, withdrawn",
            parse_mode="HTML",
        )
        return

    try:
        from db.supabase_client import update_outcome
        update_outcome(application_id, label)
        await update.message.reply_text(
            f'\u2705 Outcome "<b>{_escape_html(label)}</b>" recorded for application '
            f"<code>{_escape_html(application_id)}</code>.",
            parse_mode="HTML",
        )
    except Exception as exc:
        logger.error("Failed to update outcome for %s: %s", application_id, exc)
        await update.message.reply_text(
            f"\u274c Failed to update: {_escape_html(str(exc))}",
            parse_mode="HTML",
        )
        return

    # --- Auto-retrain hook ---------------------------------------------------
    # After a successful outcome update, check if we should retrain the
    # match scorer model on accumulated outcome data.
    try:
        from nlp.retrain import should_retrain, retrain

        if should_retrain():
            await update.message.reply_text(
                "\U0001f504 Retraining match scorer with new outcome data..."
            )

            # Run retrain in a thread to avoid blocking the bot event loop
            def _retrain_background():
                try:
                    result = retrain()
                    if result["status"] == "success":
                        send_notification(
                            f"\u2705 <b>Retrain Complete</b>\n\n"
                            f"Trained on {result['num_positive']} positive + "
                            f"{result['num_negative']} negative examples.\n"
                            f"Model saved to <code>{result['model_path']}</code>"
                        )
                    else:
                        send_notification(
                            f"\u26a0\ufe0f Retrain {result['status']}: {result['reason']}"
                        )
                except Exception as exc:
                    logger.error("Background retrain failed: %s", exc, exc_info=True)
                    send_notification(f"\u274c Retrain failed: {exc}")

            threading.Thread(target=_retrain_background, daemon=True).start()
    except ImportError:
        pass  # retrain module not available yet
    except Exception as exc:
        logger.warning("Auto-retrain check failed: %s", exc)


async def _handle_status(update: Update, context) -> None:
    """Handle the /status command."""
    with _interrupt_lock:
        pending = 1 if _pending_interrupt is not None else 0

    text = (
        f"✅ <b>CouchHire Bot is running.</b>\n"
        f"\n"
        f"Pending interrupts: {pending}"
    )
    await update.message.reply_text(text, parse_mode="HTML")



async def _handle_apply(update: Update, context) -> None:
    """Handle the /apply command: paste a JD or URL to kick off the pipeline.

    Usage:
        /apply <job URL>
        /apply <pasted JD text>

    Detects whether the input is a URL or raw JD text, confirms receipt,
    and triggers the pipeline in a background thread.
    """
    if not context.args:
        await update.message.reply_text(
            "📋 <b>Usage:</b>\n\n"
            "<code>/apply https://jobs.lever.co/company/...</code>\n"
            "or\n"
            "<code>/apply &lt;paste full JD text here&gt;</code>",
            parse_mode="HTML",
        )
        return

    raw_input = " ".join(context.args)

    # Detect if input is a URL or raw JD text
    url_pattern = re.compile(
        r"""https?://[^\s<>"']+""",
        re.IGNORECASE,
    )
    urls = url_pattern.findall(raw_input)

    if urls:
        # URL mode — extract the first URL
        job_url = urls[0]
        input_type = "url"
        preview = f"🔗 <code>{_escape_html(job_url)}</code>"
    else:
        # Raw JD text mode
        input_type = "jd"
        # Show first 200 chars as preview
        snippet = raw_input[:200].strip()
        if len(raw_input) > 200:
            snippet += "..."
        preview = f"<i>{_escape_html(snippet)}</i>"

    # Confirm receipt
    await update.message.reply_text(
        f"📥 <b>Application Received</b>\n\n"
        f"<b>Input:</b> {preview}\n\n"
        f"⏳ Parsing job description...",
        parse_mode="HTML",
    )

    logger.info(
        "Received /apply command (type=%s, length=%d chars)",
        input_type,
        len(raw_input),
    )

    # Trigger pipeline in background thread
    def _run_pipeline_thread():
        from pipeline import run_pipeline
        try:
            if input_type == "url":
                run_pipeline(url=job_url, source="telegram")
            else:
                run_pipeline(jd_text=raw_input, source="telegram")
        except Exception as exc:
            logger.error("Pipeline failed: %s", exc, exc_info=True)
            send_notification(f"❌ Pipeline error: {html.escape(str(exc))}")

    t = threading.Thread(target=_run_pipeline_thread, daemon=True)
    t.start()
    logger.info("Pipeline thread started for /apply (type=%s)", input_type)


# ---------------------------------------------------------------------------
# /search command — JobSpy job discovery
# ---------------------------------------------------------------------------

async def _handle_search(update: Update, context) -> None:
    """Handle /search command — search job boards for matching jobs.

    Usage:
        /search Python Backend Engineer
        /search Python Backend Engineer, Remote
        /search Python Backend Engineer, Bangalore, fulltime
    """
    if not context.args:
        await update.message.reply_text(
            "Usage: /search &lt;query&gt; [, location] [, job_type]\n\n"
            "Examples:\n"
            "<code>/search Python Backend Engineer</code>\n"
            "<code>/search Python Backend Engineer, Remote</code>\n"
            "<code>/search ML Engineer, Bangalore, fulltime</code>",
            parse_mode="HTML",
        )
        return

    # Parse comma-separated args: query, location, job_type
    full_input = " ".join(context.args)
    parts = [p.strip() for p in full_input.split(",")]
    query = parts[0]
    location = parts[1] if len(parts) > 1 else ""
    job_type = parts[2] if len(parts) > 2 else ""

    await update.message.reply_text(
        f"🔍 Searching for: <b>{html.escape(query)}</b>"
        + (f"\n📍 Location: {html.escape(location)}" if location else "")
        + (f"\n📋 Type: {html.escape(job_type)}" if job_type else "")
        + "\n\n⏳ Searching and scoring against your CV...",
        parse_mode="HTML",
    )

    # Run search + filter in a background thread (it's CPU-heavy due to scoring)
    def _search_background():
        global _search_results_cache

        try:
            from jobs.job_search import search_jobs
            from jobs.job_filter import filter_and_score
            from config import MIN_MATCH_SCORE

            # 1. Search job boards
            raw_jobs = search_jobs(query=query, location=location, job_type=job_type)
            logger.info("JobSpy returned %d raw results", len(raw_jobs))

            if not raw_jobs:
                send_notification("😕 No jobs found for that search. Try different keywords.")
                return

            # 2. Score and filter
            scored_jobs = filter_and_score(raw_jobs)

            if not scored_jobs:
                # Check if ALL jobs were skipped due to missing descriptions
                # (common with LinkedIn scraping — returns titles but no text)
                has_descriptions = any(
                    (j.get("description") or j.get("snippet") or "")
                    for j in raw_jobs
                )
                if not has_descriptions:
                    # Fallback: show unscored individual cards
                    count = min(len(raw_jobs), 10)
                    header = (
                        f"🔍 <b>Found {count} job{'s' if count != 1 else ''}</b> "
                        f"(descriptions unavailable — showing unscored):"
                    )
                    send_notification(header)

                    # Store raw jobs in cache so Apply callback can find them
                    _search_results_cache = raw_jobs[:10]

                    for i, job in enumerate(raw_jobs[:10]):
                        title = html.escape(job.get("title", "Unknown"))
                        company = html.escape(job.get("company", "Unknown"))
                        location_text = job.get("location", "")
                        job_url = job.get("url", "")

                        card_text = (
                            f"📋 <b>{title}</b>\n"
                            f"🏢 {company}"
                        )
                        if location_text:
                            card_text += f"\n📍 {html.escape(str(location_text))}"

                        buttons = []
                        if job_url:
                            buttons.append(
                                InlineKeyboardButton(
                                    "✅ Apply",
                                    callback_data=f"apply_{i}_{hash(job_url) % 10000}",
                                )
                            )
                            buttons.append(
                                InlineKeyboardButton("🔗 View Job", url=job_url),
                            )
                        if buttons:
                            keyboard = InlineKeyboardMarkup([buttons])
                            send_notification(card_text, reply_markup=keyboard)
                        else:
                            send_notification(card_text)


                else:
                    send_notification(
                        f"😕 Found {len(raw_jobs)} jobs but none scored above "
                        f"your {MIN_MATCH_SCORE}% threshold. Try broader keywords."
                    )
                return

            # 3. Send header + individual job cards with [Apply] buttons
            header = (
                f"🔍 <b>Found {len(scored_jobs)} matching "
                f"job{'s' if len(scored_jobs) != 1 else ''}:</b>"
            )
            send_notification(header)

            for i, job in enumerate(scored_jobs[:5]):  # Top 5 get individual cards
                title = html.escape(job.get("title", "Unknown"))
                company = html.escape(job.get("company", "Unknown"))
                score_val = job.get("match_score", 0)
                job_url = job.get("url", "")
                location_text = job.get("location", "")
                salary = job.get("salary", "")

                card_text = (
                    f"📋 <b>{title}</b>\n"
                    f"🏢 {company}\n"
                    f"📊 Match: <b>{score_val:.0f}%</b>"
                )
                if location_text:
                    card_text += f"\n📍 {html.escape(str(location_text))}"
                if salary:
                    card_text += f"\n💰 {html.escape(str(salary))}"

                # Inline buttons: Apply (triggers pipeline) + View (job URL)
                keyboard = InlineKeyboardMarkup([
                    [
                        InlineKeyboardButton(
                            "✅ Apply",
                            callback_data=f"apply_{i}_{hash(job_url) % 10000}",
                        ),
                        InlineKeyboardButton("🔗 View Job", url=job_url),
                    ]
                ])

                send_notification(card_text, reply_markup=keyboard)

            # 5. Store scored_jobs in module-level state for callback handling
            _search_results_cache = scored_jobs

        except ConnectionError as exc:
            logger.error("Job search failed: %s", exc)
            send_notification(f"❌ Job search error: {html.escape(str(exc))}")
        except Exception as exc:
            logger.error("Search failed: %s", exc, exc_info=True)
            send_notification(f"❌ Search failed: {html.escape(str(exc))}")

    threading.Thread(target=_search_background, daemon=True).start()


# ---------------------------------------------------------------------------
# Text reply handler (browser agent interrupts)
# ---------------------------------------------------------------------------

async def _handle_text_reply(update: Update, context) -> None:
    """Handle free-text replies (for browser agent interrupt responses)."""
    global _pending_interrupt

    handled = False
    with _interrupt_lock:
        if _pending_interrupt is not None:
            _pending_interrupt["response"] = update.message.text
            _pending_interrupt["event"].set()
            handled = True

    if handled:
        await update.message.reply_text("✅ Got it.", parse_mode="HTML")
    else:
        await update.message.reply_text(
            "No active prompt. Use /outcome to label an application.",
            parse_mode="HTML",
        )


# ---------------------------------------------------------------------------
# Callback query handler (inline buttons)
# ---------------------------------------------------------------------------

async def _handle_callback(update: Update, context) -> None:
    """Handle inline button presses (yes/no/done/apply_*)."""
    global _pending_interrupt

    query = update.callback_query
    data = query.data

    # --- Handle apply_ callbacks from /search results ---
    if data.startswith("apply_"):
        parts = data.split("_")
        try:
            job_index = int(parts[1])
        except (IndexError, ValueError):
            await query.answer("Invalid callback data.")
            return

        if job_index < len(_search_results_cache):
            job = _search_results_cache[job_index]
            await query.answer("Starting application...")

            # Update the card message to show progress
            try:
                await query.edit_message_text(
                    query.message.text_html + "\n\n⏳ <i>Fetching full job description...</i>",
                    parse_mode="HTML",
                )
            except telegram.error.TelegramError as exc:
                logger.warning("Failed to edit message: %s", exc)

            # Trigger pipeline in background
            title = html.escape(job.get("title", ""))
            company = html.escape(job.get("company", ""))

            def _apply_from_search():
                try:
                    # Get full JD — already available from JobSpy search results
                    full_jd = job.get("description", "")
                    if not full_jd:
                        # Fallback: try fetching (unlikely needed with JobSpy)
                        from jobs.job_search import get_job_details
                        details = get_job_details(job.get("url", ""))
                        full_jd = details.get("description", "")

                    from pipeline import run_pipeline
                    run_pipeline(
                        jd_text=full_jd,
                        url=job.get("url", ""),
                        source="search",
                    )
                except Exception as exc:
                    logger.error("Apply from search failed: %s", exc, exc_info=True)
                    send_notification(f"❌ Failed to start application: {html.escape(str(exc))}")

            threading.Thread(target=_apply_from_search, daemon=True).start()
        else:
            await query.answer("Job not found in cache. Search again.")
        return

    # --- Handle gate callbacks (pipeline approval gates) ---
    if data.startswith("gate1_") or data.startswith("gate2_"):
        handled = False
        with _interrupt_lock:
            if _pending_interrupt is not None:
                _pending_interrupt["response"] = data
                _pending_interrupt["event"].set()
                handled = True

        if handled:
            # Annotate the original message with the user's choice
            choice_label = {
                "gate1_approve": "✅ Approved",
                "gate1_regenerate": "✏️ Regenerate",
                "gate1_cancel": "❌ Cancelled",
                "gate2_approve": "📤 Sending",
                "gate2_cancel": "❌ Cancelled",
            }.get(data, data)

            await query.answer(choice_label)
            try:
                await query.edit_message_text(
                    query.message.text_html + f"\n\n<b>{choice_label}</b>",
                    parse_mode="HTML",
                )
            except telegram.error.TelegramError:
                pass
        else:
            await query.answer("No active gate prompt.")
        return

    # --- Handle yes/no/done callbacks (interrupt system) ---
    print(f"[DEBUG-CALLBACK] Generic handler: data={data!r}, _pending_interrupt is None? {_pending_interrupt is None}, module={__name__}")
    if _pending_interrupt is not None:
        print(f"[DEBUG-CALLBACK] _pending_interrupt type={_pending_interrupt.get('type')}, id={id(_pending_interrupt)}")
    handled = False
    with _interrupt_lock:
        if _pending_interrupt is not None:
            _pending_interrupt["response"] = data
            _pending_interrupt["event"].set()
            handled = True

    if handled:
        await query.answer("✅")
    else:
        await query.answer("No active prompt.")


# ---------------------------------------------------------------------------
# start_bot — Mode 2 entry point
# ---------------------------------------------------------------------------

def start_bot() -> None:
    """Start the Telegram bot polling loop. Blocking call.

    This is one of the two async exceptions in the project (per CLAUDE.md).
    python-telegram-bot requires async handlers.
    """
    from config import TELEGRAM_BOT_TOKEN

    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not set — cannot start bot")
        return

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Register handlers
    app.add_handler(CommandHandler("start", _handle_start))
    app.add_handler(CommandHandler("outcome", _handle_outcome))
    app.add_handler(CommandHandler("apply", _handle_apply))
    app.add_handler(CommandHandler("search", _handle_search))
    app.add_handler(CommandHandler("status", _handle_status))
    app.add_handler(CallbackQueryHandler(_handle_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_text_reply))

    logger.info("Telegram bot starting polling...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    start_bot()
