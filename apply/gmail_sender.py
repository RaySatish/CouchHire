"""Gmail draft creation and sending via MCP server.

CouchHire is an MCP *client*. This module connects to an external Gmail MCP
server (configured via GMAIL_MCP_URL), calls its ``create_draft`` tool, and
returns a Gmail deeplink URL so the user can review and send manually.

After Gate 2 approval, ``send_draft()`` sends the draft via the MCP server's
``send_draft`` (or ``drafts/send``) tool.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from config import GMAIL_MCP_URL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gmail deeplink template
# ---------------------------------------------------------------------------
_GMAIL_DRAFT_URL_TEMPLATE = "https://mail.google.com/mail/u/0/#drafts/{draft_id}"


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
    str
        The Gmail draft ID returned by the MCP server.

    Raises
    ------
    RuntimeError
        If the MCP server returns an error or the draft ID cannot be extracted.
    ConnectionError
        If the MCP server is unreachable.
    """
    # Build the tool arguments dict
    tool_args: dict = {
        "subject": subject,
        "body": body,
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

    if len(valid_attachments) == 1:
        # Single attachment — use flat keys for broader MCP server compatibility
        tool_args["attachment"] = valid_attachments[0]["content"]
        tool_args["attachment_name"] = valid_attachments[0]["name"]
        tool_args["attachment_mime_type"] = valid_attachments[0]["mime_type"]
    elif len(valid_attachments) > 1:
        # Multiple attachments — try list format first, fall back to first-only
        tool_args["attachments"] = valid_attachments

    logger.info(
        "Connecting to Gmail MCP server at %s to create draft (to=%s, subject=%s, attachments=%d)",
        GMAIL_MCP_URL,
        recipient or "(blank)",
        subject,
        len(valid_attachments),
    )

    try:
        async with streamablehttp_client(GMAIL_MCP_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                logger.debug("MCP session initialised — calling create_draft tool")

                result = await session.call_tool("create_draft", tool_args)

                logger.debug("MCP create_draft raw result: %s", result)
    except OSError as exc:
        raise ConnectionError(
            f"Cannot reach Gmail MCP server at {GMAIL_MCP_URL}: {exc}"
        ) from exc

    # Extract draft_id from the result.
    draft_id = _extract_draft_id(result)

    if not draft_id:
        raise RuntimeError(
            f"Gmail MCP server did not return a usable draft ID. Raw result: {result}"
        )

    logger.info("Gmail draft created — draft_id=%s", draft_id)
    return draft_id


async def _send_draft_via_mcp(draft_id: str) -> bool:
    """Connect to the Gmail MCP server and send an existing draft.

    Parameters
    ----------
    draft_id : str
        The Gmail draft ID to send.

    Returns
    -------
    bool
        True if the draft was sent successfully.

    Raises
    ------
    RuntimeError
        If the MCP server returns an error.
    ConnectionError
        If the MCP server is unreachable.
    """
    logger.info(
        "Connecting to Gmail MCP server at %s to send draft (draft_id=%s)",
        GMAIL_MCP_URL,
        draft_id,
    )

    try:
        async with streamablehttp_client(GMAIL_MCP_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                logger.debug("MCP session initialised — calling send_draft tool")

                # Try common MCP server tool names for sending a draft
                result = None
                for tool_name in ("send_draft", "drafts_send", "gmail_send_draft"):
                    try:
                        result = await session.call_tool(
                            tool_name, {"draft_id": draft_id}
                        )
                        logger.debug("MCP %s raw result: %s", tool_name, result)
                        break
                    except Exception as exc:
                        logger.debug("Tool %s not available: %s", tool_name, exc)
                        continue

                if result is None:
                    raise RuntimeError(
                        f"No send_draft tool found on Gmail MCP server at {GMAIL_MCP_URL}. "
                        "Tried: send_draft, drafts_send, gmail_send_draft"
                    )

    except OSError as exc:
        raise ConnectionError(
            f"Cannot reach Gmail MCP server at {GMAIL_MCP_URL}: {exc}"
        ) from exc

    logger.info("Gmail draft sent — draft_id=%s", draft_id)
    return True


def _extract_draft_id(result: object) -> str | None:
    """Best-effort extraction of the draft ID from an MCP tool result.

    Different Gmail MCP server implementations return the draft ID in
    varying formats. This function tries several common patterns.
    """
    # Case 1: result has a .content attribute (list of content blocks)
    if hasattr(result, "content") and result.content:
        for block in result.content:
            # TextContent block — the text itself might be the draft ID
            if hasattr(block, "text") and block.text:
                text = block.text.strip()
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
    """Run an async coroutine from sync context, handling existing event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        # Already inside an async context — run in a thread pool
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


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

    draft_id = _run_async(
        _create_draft_via_mcp(subject, body, recipient_email, attachment_list)
    )

    draft_url = _GMAIL_DRAFT_URL_TEMPLATE.format(draft_id=draft_id)
    logger.info("Draft URL: %s", draft_url)
    return draft_url, draft_id


def send_draft(draft_id: str) -> bool:
    """Send an existing Gmail draft via the MCP server.

    Called by the pipeline after Gate 2 approval.

    Parameters
    ----------
    draft_id : str
        The Gmail draft ID (returned by create_draft()).

    Returns
    -------
    bool
        True if the draft was sent successfully.

    Raises
    ------
    ConnectionError
        If the Gmail MCP server is unreachable.
    RuntimeError
        If the MCP server returns an error or no send tool is found.
    """
    return _run_async(_send_draft_via_mcp(draft_id))
