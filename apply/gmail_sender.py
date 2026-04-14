"""Gmail draft creation and sending via MCP server.

CouchHire is an MCP *client*. This module connects to an external Gmail MCP
server (google_workspace_mcp) via Streamable HTTP, calls its tools, and
returns a Gmail deeplink URL so the user can review and send manually.

Actual MCP tool names (google_workspace_mcp v1.18.0):
  - ``draft_gmail_message``  — create a draft
  - ``send_gmail_message``   — send a message (used for sending drafts)
  - ``start_google_auth``    — trigger OAuth flow

All tool calls require ``user_google_email`` parameter (even in single-user
mode) to identify which credentials to use.

After Gate 2 approval, ``send_draft()`` sends the draft via the MCP server's
``send_gmail_message`` tool.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from config import GMAIL_MCP_URL, APPLICANT_EMAIL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gmail deeplink template
# ---------------------------------------------------------------------------
_GMAIL_DRAFT_URL_TEMPLATE = "https://mail.google.com/mail/u/0/#drafts/{draft_id}"
_GMAIL_INBOX_URL_TEMPLATE = "https://mail.google.com/mail/u/0/#all/{message_id}"


# ---------------------------------------------------------------------------
# Internal async implementation
# ---------------------------------------------------------------------------

async def _create_draft_via_mcp(
    subject: str,
    body: str,
    recipient: str,
    attachments: list[str],
) -> str:
    """Connect to the Gmail MCP server and create a draft. Returns the draft ID.

    Parameters
    ----------
    subject : str
        Email subject line.
    body : str
        Email body (plain text).
    recipient : str
        Recipient email address. May be empty for manual-route applications.
    attachments : list[str]
        List of file paths to attach (PDFs). Empty list means no attachments.

    Returns
    -------
    tuple[str, str | None]
        (draft_id, message_id) — the MCP draft ID and the real Gmail message ID
        (used for web URLs). message_id may be None if lookup fails.

    Raises
    ------
    RuntimeError
        If the MCP server returns an error or the draft ID cannot be extracted.
    ConnectionError
        If the MCP server is unreachable.
    """
    # Build the tool arguments dict — user_google_email is always required
    tool_args: dict = {
        "subject": subject,
        "body": body,
        "user_google_email": APPLICANT_EMAIL,
    }

    # Only include recipient if non-empty (manual-route may have no recipient)
    if recipient:
        tool_args["to"] = recipient

    # Attach files — encode each as base64
    # For single attachment, use the flat keys the MCP server expects.
    # For multiple, use an attachments list if the server supports it.
    valid_attachments: list[dict] = []
    for file_path in attachments:
        pdf_file = Path(file_path)
        if pdf_file.exists() and pdf_file.is_file():
            pdf_bytes = pdf_file.read_bytes()
            valid_attachments.append({
                "content": base64.b64encode(pdf_bytes).decode("utf-8"),
                "name": pdf_file.name,
                "mime_type": "application/pdf",
            })
            logger.info("Attaching file: %s (%d bytes)", pdf_file.name, len(pdf_bytes))
        else:
            logger.warning(
                "Attachment path provided but file not found: %s — skipping",
                file_path,
            )

    if valid_attachments:
        # MCP server expects "attachments" as a list of dicts with keys:
        #   "content" (base64), "filename", and optionally "mime_type".
        # The server schema has additionalProperties=false, so we must
        # use exactly these key names (not "name", not flat keys).
        tool_args["attachments"] = [
            {
                "content": att["content"],
                "filename": att["name"],
                "mime_type": att["mime_type"],
            }
            for att in valid_attachments
        ]

    logger.info(
        "Connecting to Gmail MCP server at %s to create draft (to=%s, subject=%s, attachments=%d)",
        GMAIL_MCP_URL,
        recipient or "(blank)",
        subject,
        len(valid_attachments),
    )

    message_id: str | None = None

    try:
        async with streamablehttp_client(GMAIL_MCP_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                logger.debug("MCP session initialised — calling draft_gmail_message tool")

                result = await session.call_tool("draft_gmail_message", tool_args)

                logger.debug("MCP draft_gmail_message raw result: %s", result)

                # Extract draft_id from the result.
                draft_id = _extract_draft_id(result)

                if not draft_id:
                    raise RuntimeError(
                        f"Gmail MCP server did not return a usable draft ID. Raw result: {result}"
                    )

                logger.info("Gmail draft created — draft_id=%s", draft_id)

                # The MCP server returns a draft resource ID (e.g. r-123456)
                # which does NOT work in Gmail web URLs.  Gmail web URLs need
                # the hex message ID.  Look it up by searching for the draft
                # we just created (reuse the same session).
                message_id = await _lookup_draft_message_id(session, subject)
                if message_id:
                    logger.info("Resolved draft message_id=%s for web URL", message_id)
                else:
                    logger.warning(
                        "Could not resolve message_id for draft (draft_id=%s). "
                        "Falling back to drafts folder URL.",
                        draft_id,
                    )

    except OSError as exc:
        raise ConnectionError(
            f"Cannot reach Gmail MCP server at {GMAIL_MCP_URL}: {exc}"
        ) from exc

    return draft_id, message_id


async def _lookup_draft_message_id(session: ClientSession, subject: str) -> str | None:
    """Search for a recently created draft by subject to get the real Gmail message ID.

    The MCP server's ``draft_gmail_message`` returns a draft resource ID
    (``r-...``) that cannot be used in Gmail web URLs.  Gmail web URLs
    require the hex message ID.  This helper searches for the draft we
    just created and extracts that ID.
    """
    import re as _re

    try:
        # Use a narrow search: exact subject, in drafts, very recent
        search_result = await session.call_tool("search_gmail_messages", {
            "query": f'subject:"{subject}" in:drafts newer_than:1d',
            "user_google_email": APPLICANT_EMAIL,
        })

        if not hasattr(search_result, "content") or not search_result.content:
            return None

        for block in search_result.content:
            if hasattr(block, "text") and block.text:
                # The search result contains lines like:
                #   1. Message ID: 19d768e1dce66005
                match = _re.search(r"Message ID:\s*([a-f0-9]+)", block.text)
                if match:
                    return match.group(1)

    except Exception as exc:
        logger.debug("Draft message_id lookup failed (non-fatal): %s", exc)

    return None



async def _fetch_draft_content_via_mcp(
    original_subject: str,
    stale_message_id: str | None = None,
) -> dict | None:
    """Fetch the current content of a Gmail draft, picking up user edits.

    Gmail changes a draft's internal message ID every time the user edits it
    in the web UI.  The message_id we stored at creation time becomes stale,
    and ``get_gmail_message_content`` returns 404.

    This function **always searches** for the draft by subject to find the
    current message ID, then fetches its content.  The search adds ~1s but
    guarantees we get the user's edited version.

    Parameters
    ----------
    original_subject : str
        The email subject used when creating the draft.  Used to search
        for the draft in Gmail.
    stale_message_id : str or None
        Unused — kept for API compatibility.  We always search by subject
        because Gmail changes the message ID on every draft edit.

    Returns a dict with keys: subject, body, to, cc, bcc (all str or None),
    or None if the fetch fails entirely.
    """
    import re as _re

    logger.info(
        "Fetching edited draft from Gmail (searching by subject=%r)",
        original_subject[:80],
    )

    try:
        async with streamablehttp_client(GMAIL_MCP_URL) as (rs, ws, _):
            async with ClientSession(rs, ws) as sess:
                await sess.initialize()

                # Search for the draft by subject
                search_result = await sess.call_tool("search_gmail_messages", {
                    "query": f'subject:"{original_subject}" in:drafts newer_than:1d',
                    "user_google_email": APPLICANT_EMAIL,
                    "page_size": 3,
                })

                # Extract message ID from search results
                found_id = None
                if hasattr(search_result, "content") and search_result.content:
                    for block in search_result.content:
                        if hasattr(block, "text") and block.text:
                            m = _re.search(r"Message ID:\s*([a-f0-9]+)", block.text)
                            if m:
                                found_id = m.group(1)
                                break

                if not found_id:
                    logger.warning(
                        "Could not find draft in Gmail (subject=%r) — "
                        "draft may have been sent, deleted, or subject changed",
                        original_subject[:60],
                    )
                    return None

                logger.info("Found draft message_id=%s via search", found_id)

                # Fetch the content with the current ID
                result = await sess.call_tool("get_gmail_message_content", {
                    "message_id": found_id,
                    "user_google_email": APPLICANT_EMAIL,
                    "body_format": "text",
                })

                return _parse_mcp_message_content(result, _re)

    except Exception as exc:
        logger.warning(
            "Failed to fetch draft content (subject=%r): %s — will fall back to original",
            original_subject[:60], exc,
        )
        return None


def _parse_mcp_message_content(result: object, _re) -> dict | None:
    """Parse a get_gmail_message_content MCP result into a dict.

    Returns dict with keys: subject, body, to, cc, bcc (all str or None),
    or None if parsing fails.
    """
    if not hasattr(result, "content") or not result.content:
        return None

    full_text = ""
    for block in result.content:
        if hasattr(block, "text") and block.text:
            full_text += block.text

    if not full_text:
        return None

    logger.debug("Draft content fetched (%d chars)", len(full_text))

    # google_workspace_mcp returns:
    #   Subject: <subject>
    #   From: <from>
    #   To: <to>
    #   ...
    #   --- BODY ---
    #   <body text>
    draft: dict = {}

    subj_match = _re.search(r"^Subject:\s*(.+?)$", full_text, _re.MULTILINE)
    draft["subject"] = subj_match.group(1).strip() if subj_match else None

    to_match = _re.search(r"^To:\s*(.+?)$", full_text, _re.MULTILINE)
    draft["to"] = to_match.group(1).strip() if to_match else None

    cc_match = _re.search(r"^CC:\s*(.+?)$", full_text, _re.MULTILINE)
    draft["cc"] = cc_match.group(1).strip() if cc_match else None

    bcc_match = _re.search(r"^BCC:\s*(.+?)$", full_text, _re.MULTILINE)
    draft["bcc"] = bcc_match.group(1).strip() if bcc_match else None

    # Body: everything after "--- BODY ---" or "Body:" line
    body_match = _re.search(
        r"^(?:---\s*BODY\s*---|Body:)\s*\n(.*)",
        full_text, _re.MULTILINE | _re.DOTALL,
    )
    if body_match:
        draft["body"] = body_match.group(1).strip()
    else:
        # Fallback: everything after the first blank line (end of headers)
        parts = full_text.split("\n\n", 1)
        draft["body"] = parts[1].strip() if len(parts) > 1 else None

    # Strip attachment metadata appended by the MCP server.
    # get_gmail_message_content returns attachment info after the body like:
    #   --- ATTACHMENTS ---
    #   1. Resume.pdf (application/pdf, 120.0 KB)
    #      Attachment ID: ANGjdJ...
    #      Use get_gmail_attachment_content(...) to download
    # This must NOT be included in the email body when re-sending.
    if draft.get("body"):
        att_marker = _re.search(
            r"\n---\s*ATTACHMENTS?\s*---",
            draft["body"],
            _re.IGNORECASE,
        )
        if att_marker:
            draft["body"] = draft["body"][:att_marker.start()].strip()
            logger.debug(
                "Stripped attachment metadata from body (removed %d chars)",
                len(draft["body"]) - att_marker.start(),
            )

    logger.info(
        "Draft content parsed — subject=%s, to=%s, body_len=%d",
        draft.get("subject", "(none)"),
        draft.get("to", "(none)"),
        len(draft.get("body") or ""),
    )
    return draft


async def _send_email_via_mcp(
    subject: str,
    body: str,
    recipient: str,
    attachments: list[str],
) -> str | None:
    """Connect to the Gmail MCP server and send an email.

    The MCP server's send_gmail_message tool sends a new email (it does not
    support sending an existing draft by ID).  We reconstruct the email from
    the same details that were used to create the draft.

    Parameters
    ----------
    subject : str
        Email subject line.
    body : str
        Email body text.
    recipient : str
        Recipient email address.
    attachments : list[str]
        List of file paths to attach (PDFs). Empty list means no attachments.

    Returns
    -------
    str or None
        The sent message ID if successful, None if the send failed.

    Raises
    ------
    RuntimeError
        If the MCP server returns an error.
    ConnectionError
        If the MCP server is unreachable.
    """
    logger.info(
        "Connecting to Gmail MCP server at %s to send email (to=%s, subject=%s)",
        GMAIL_MCP_URL,
        recipient,
        subject,
    )

    tool_args: dict = {
        "to": recipient,
        "subject": subject,
        "body": body,
        "user_google_email": APPLICANT_EMAIL,
    }

    # Build attachments in the MCP server's expected format
    valid_attachments: list[dict] = []
    for file_path in attachments:
        p = Path(file_path)
        if p.exists():
            valid_attachments.append({
                "content": base64.b64encode(p.read_bytes()).decode("ascii"),
                "filename": p.name,
                "mime_type": "application/pdf",
            })

    if valid_attachments:
        tool_args["attachments"] = [
            {
                "content": att["content"],
                "filename": att["filename"],
                "mime_type": att["mime_type"],
            }
            for att in valid_attachments
        ]

    result = None
    try:
        async with streamablehttp_client(GMAIL_MCP_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                logger.debug("MCP session initialised — calling send_gmail_message tool")

                result = await session.call_tool("send_gmail_message", tool_args)
                logger.debug("MCP send_gmail_message raw result: %s", result)

                # Check for errors in the response
                if hasattr(result, "isError") and result.isError:
                    error_text = ""
                    if hasattr(result, "content"):
                        for block in result.content:
                            if hasattr(block, "text"):
                                error_text += block.text
                    raise RuntimeError(f"MCP send_gmail_message failed: {error_text}")

    except OSError as exc:
        raise ConnectionError(
            f"Cannot reach Gmail MCP server at {GMAIL_MCP_URL}: {exc}"
        ) from exc

    # Extract message ID from the MCP response
    sent_message_id: str | None = None
    if hasattr(result, "content") and result.content:
        for block in result.content:
            if hasattr(block, "text") and block.text:
                text = block.text.strip()
                # google_workspace_mcp format: "Email sent! Message ID: <id>"
                if "Message ID:" in text:
                    sent_message_id = text.split("Message ID:")[-1].strip()
                    break

    logger.info(
        "Email sent successfully via MCP (to=%s, subject=%s, message_id=%s)",
        recipient, subject, sent_message_id,
    )
    return sent_message_id


def _extract_draft_id(result: object) -> str | None:
    """Best-effort extraction of the draft ID from an MCP tool result.

    The google_workspace_mcp server returns results like:
        TextContent(text='Draft created! Draft ID: r5100756162228124')

    This function handles that format plus JSON and plain-string fallbacks.
    """
    # Case 1: result has a .content attribute (list of content blocks)
    if hasattr(result, "content") and result.content:
        for block in result.content:
            # TextContent block — the text itself might contain the draft ID
            if hasattr(block, "text") and block.text:
                text = block.text.strip()

                # Check for MCP error responses first — do NOT treat errors as IDs
                if "validation error" in text.lower() or "error" in text.lower()[:30]:
                    logger.error("MCP server returned an error: %s", text[:500])
                    return None

                # google_workspace_mcp format: "Draft created! Draft ID: <id>"
                if "Draft ID:" in text:
                    return text.split("Draft ID:")[-1].strip()

                # Some servers return JSON like {"draftId": "..."}
                if text.startswith("{"):
                    try:
                        data = json.loads(text)
                        for key in ("draftId", "draft_id", "id"):
                            if key in data:
                                return str(data[key]).strip()
                    except (json.JSONDecodeError, KeyError):
                        pass
                # Otherwise treat the raw text as the draft ID
                if text:
                    return text

    # Case 2: result is a plain string
    if isinstance(result, str) and result.strip():
        return result.strip()

    # Case 3: result is a dict
    if isinstance(result, dict):
        for key in ("draftId", "draft_id", "id"):
            if key in result:
                return str(result[key]).strip()

    return None


# ---------------------------------------------------------------------------
# Sync bridge helper
# ---------------------------------------------------------------------------

def _run_async(coro):
    """Run an async coroutine from sync context, safely isolated from any outer loop.

    Always executes in a dedicated thread with a fresh event loop.  This avoids
    conflicts when the caller is a background thread spawned by the Telegram bot
    (which runs its own asyncio loop in the main thread).  The MCP SDK's
    streamablehttp_client + anyio internals require a clean loop lifecycle.
    """
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result(timeout=60)


# ---------------------------------------------------------------------------
# Public sync interface (called by the pipeline)
# ---------------------------------------------------------------------------

def create_draft(
    subject: str,
    body: str,
    recipient_email: str,
    attachments: list[str] | None = None,
    pdf_path: str | None = None,
) -> tuple[str, str]:
    """Create a Gmail draft via the MCP server. Returns (draft_url, draft_id).

    This is synchronous — the async MCP calls are wrapped internally.

    Parameters
    ----------
    subject : str
        Email subject line.
    body : str
        Email body text.
    recipient_email : str
        Recipient email address. Pass empty string for manual-route applications.
    attachments : list[str] or None
        List of file paths to attach. Preferred parameter for multiple files.
    pdf_path : str or None
        Legacy single-file parameter. Used if attachments is None.

    Returns
    -------
    tuple[str, str]
        (draft_url, draft_id) — Gmail deeplink URL and the raw draft ID.

    Raises
    ------
    ConnectionError
        If the Gmail MCP server is unreachable.
    RuntimeError
        If the MCP server returns an error or no draft ID.
    """
    # Build attachments list from either parameter
    attachment_list: list[str] = []
    if attachments is not None:
        attachment_list = [a for a in attachments if a]
    elif pdf_path is not None:
        attachment_list = [pdf_path]

    draft_id, message_id = _run_async(
        _create_draft_via_mcp(subject, body, recipient_email, attachment_list)
    )

    # Prefer the real message ID for the web URL (works in Gmail).
    # Fall back to the drafts folder if message_id lookup failed.
    if message_id:
        draft_url = _GMAIL_INBOX_URL_TEMPLATE.format(message_id=message_id)
    else:
        # Fallback: link to the drafts folder (user finds it manually)
        draft_url = "https://mail.google.com/mail/u/0/#drafts"
    logger.info("Draft URL: %s", draft_url)
    return draft_url, draft_id


def send_email(
    subject: str,
    body: str,
    recipient_email: str,
    attachments: list[str] | None = None,
    draft_message_id: str | None = None,
) -> str | None:
    """Send an email via the Gmail MCP server.

    Called by the pipeline after Gate 2 approval.  Before sending, if a
    ``draft_message_id`` is provided, the function fetches the **current**
    draft content from Gmail.  This ensures that any edits the user made
    in the Gmail UI (subject, body, recipients) are respected.

    If the draft fetch fails or ``draft_message_id`` is not provided,
    falls back to sending with the original ``subject``/``body``/
    ``recipient_email`` from pipeline state.

    Parameters
    ----------
    subject : str
        Original email subject line (fallback).
    body : str
        Original email body text (fallback).
    recipient_email : str
        Original recipient email address (fallback).
    attachments : list[str] or None
        List of file paths to attach (PDFs).
    draft_message_id : str or None
        The Gmail message ID of the draft (hex ID used in web URLs).
        If provided, the draft is fetched from Gmail to pick up user edits.

    Returns
    -------
    str or None
        The sent message ID if successful, None if the send failed.

    Raises
    ------
    ConnectionError
        If the Gmail MCP server is unreachable.
    RuntimeError
        If the MCP server returns an error.
    """
    # Try to fetch the edited draft content from Gmail
    final_subject = subject
    final_body = body
    final_recipient = recipient_email

    if draft_message_id:
        logger.info(
            "Fetching draft from Gmail before sending (message_id=%s) "
            "to pick up any user edits...",
            draft_message_id,
        )
        try:
            draft_content = _run_async(
                _fetch_draft_content_via_mcp(
                    original_subject=subject,
                    stale_message_id=draft_message_id,
                )
            )
            if draft_content:
                # Use fetched values, falling back to originals if any field is missing
                if draft_content.get("subject"):
                    final_subject = draft_content["subject"]
                    if final_subject != subject:
                        logger.info("User edited subject: %r -> %r", subject, final_subject)
                if draft_content.get("body"):
                    final_body = draft_content["body"]
                    if final_body != body:
                        logger.info("User edited email body (original %d chars -> edited %d chars)",
                                    len(body), len(final_body))
                if draft_content.get("to"):
                    # Extract just the email from "Name <email>" format if needed
                    fetched_to = draft_content["to"]
                    if "<" in fetched_to and ">" in fetched_to:
                        import re as _re
                        email_match = _re.search(r"<([^>]+)>", fetched_to)
                        if email_match:
                            fetched_to = email_match.group(1)
                    final_recipient = fetched_to
                    if final_recipient != recipient_email:
                        logger.info("User edited recipient: %r -> %r", recipient_email, final_recipient)
                logger.info("Using edited draft content for sending")
            else:
                logger.warning("Draft fetch returned None — sending with original content")
        except Exception as exc:
            logger.warning("Failed to fetch edited draft: %s — sending with original content", exc)

    attachment_list = [a for a in (attachments or []) if a]
    return _run_async(_send_email_via_mcp(final_subject, final_body, final_recipient, attachment_list))


def send_draft(draft_id: str) -> bool:
    """Deprecated: MCP server does not support sending drafts by ID.

    Use send_email() instead. This stub exists for backward compatibility
    and always raises RuntimeError.
    """
    raise RuntimeError(
        "send_draft() is no longer supported. The Gmail MCP server does not "
        "have a send-by-draft-ID tool. Use send_email() with full email details."
    )
