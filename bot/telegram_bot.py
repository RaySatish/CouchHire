"""CouchHire Telegram bot — notifications, /outcome command, and browser agent interrupts.

Two modes of operation:

Mode 1 — Sync notification senders (called by pipeline, browser_agent, etc.):
    send_notification(), send_photo(), send_job_card(), send_draft_ready(),
    send_form_started(), send_manual_notice()

Mode 2 — Long-running async bot (for /outcome command + interrupt replies):
    start_bot()  — blocking polling loop

Browser agent interrupt system:
    ask_user(), ask_yes_no(), send_takeover_instructions()

This file uses async handlers because python-telegram-bot requires it.
This is one of the two allowed async exceptions per CLAUDE.md.
"""

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
    return html.escape(text)


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
            f"❌ Invalid label: <code>{_escape_html(label)}</code>\n\n"
            f"Valid labels: interview, rejected, no_response, offer, withdrawn",
            parse_mode="HTML",
        )
        return

    try:
        from db.supabase_client import update_outcome
        update_outcome(application_id, label)
        await update.message.reply_text(
            f'✅ Outcome "<b>{_escape_html(label)}</b>" recorded for application '
            f"<code>{_escape_html(application_id)}</code>.",
            parse_mode="HTML",
        )
    except Exception as exc:
        logger.error("Failed to update outcome for %s: %s", application_id, exc)
        await update.message.reply_text(
            f"❌ Failed to update: {_escape_html(str(exc))}",
            parse_mode="HTML",
        )


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

    # Trigger pipeline in background thread
    # TODO: Wire to pipeline.py in Step 25
    # Expected integration:
    #   import threading
    #   from pipeline import run_pipeline
    #   if input_type == "url":
    #       t = threading.Thread(target=run_pipeline, kwargs={"url": job_url}, daemon=True)
    #   else:
    #       t = threading.Thread(target=run_pipeline, kwargs={"jd_text": raw_input}, daemon=True)
    #   t.start()

    logger.info(
        "Received /apply command (type=%s, length=%d chars)",
        input_type,
        len(raw_input),
    )

    # Temporary acknowledgement until pipeline.py is wired
    await update.message.reply_text(
        "🚧 <b>Pipeline not yet wired.</b>\n\n"
        "The /apply command is ready — it will trigger the full pipeline "
        "once <code>pipeline.py</code> (Step 25) is implemented.\n\n"
        f"Detected input type: <b>{input_type}</b>",
        parse_mode="HTML",
    )


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


async def _handle_callback(update: Update, context) -> None:
    """Handle inline button presses (yes/no/done)."""
    global _pending_interrupt

    query = update.callback_query
    data = query.data

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
