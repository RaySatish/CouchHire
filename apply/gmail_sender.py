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
) -> str | None:
    """Send an email via the Gmail MCP server.

    Called by the pipeline after Gate 2 approval.  The MCP server does not
    support sending an existing draft by ID, so we re-compose the email from
    the same details used to create the draft.

    Parameters
    ----------
    subject : str
        Email subject line.
    body : str
        Email body text.
    recipient_email : str
        Recipient email address.
    attachments : list[str] or None
        List of file paths to attach (PDFs).

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
    attachment_list = [a for a in (attachments or []) if a]
    return _run_async(_send_email_via_mcp(subject, body, recipient_email, attachment_list))


def send_draft(draft_id: str) -> bool:
    """Deprecated: MCP server does not support sending drafts by ID.

    Use send_email() instead. This stub exists for backward compatibility
    and always raises RuntimeError.
    """
    raise RuntimeError(
        "send_draft() is no longer supported. The Gmail MCP server does not "
        "have a send-by-draft-ID tool. Use send_email() with full email details."
    )
