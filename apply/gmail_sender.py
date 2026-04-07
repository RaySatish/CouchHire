"""Gmail draft creation via MCP server.

CouchHire is an MCP *client*. This module connects to an external Gmail MCP
server (configured via GMAIL_MCP_URL), calls its ``create_draft`` tool, and
returns a Gmail deeplink URL so the user can review and send manually.

**Draft-only policy:** There is no ``send()`` function in this module or
anywhere in the codebase. Every code path creates a draft — never sends.
"""

import asyncio
import base64
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
    pdf_path: str | None,
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
    pdf_path : str or None
        Absolute path to the resume PDF to attach. ``None`` means no attachment.

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

    # Attach the PDF if a path was provided and the file exists
    if pdf_path is not None:
        pdf_file = Path(pdf_path)
        if pdf_file.exists() and pdf_file.is_file():
            pdf_bytes = pdf_file.read_bytes()
            tool_args["attachment"] = base64.b64encode(pdf_bytes).decode("utf-8")
            tool_args["attachment_name"] = pdf_file.name
            tool_args["attachment_mime_type"] = "application/pdf"
            logger.info("Attaching PDF: %s (%d bytes)", pdf_file.name, len(pdf_bytes))
        else:
            logger.warning(
                "PDF path provided but file not found: %s — creating draft without attachment",
                pdf_path,
            )

    logger.info(
        "Connecting to Gmail MCP server at %s to create draft (to=%s, subject=%s)",
        GMAIL_MCP_URL,
        recipient or "(blank)",
        subject,
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
    # The MCP server may return the ID in different shapes depending on the
    # implementation (plain string, dict with "draft_id", or result.content).
    draft_id = _extract_draft_id(result)

    if not draft_id:
        raise RuntimeError(
            f"Gmail MCP server did not return a usable draft ID. Raw result: {result}"
        )

    logger.info("Gmail draft created — draft_id=%s", draft_id)
    return draft_id


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
                        import json
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
# Public sync interface (called by the pipeline)
# ---------------------------------------------------------------------------

def create_draft(
    subject: str,
    body: str,
    pdf_path: str | None,
    recipient_email: str,
) -> str:
    """Create a Gmail draft via the MCP server. Returns the Gmail draft deeplink URL.

    This is the only public function in this module. It is synchronous —
    the async MCP calls are wrapped internally.

    Parameters
    ----------
    subject : str
        Email subject line.
    body : str
        Email body text.
    pdf_path : str or None
        Path to the resume PDF to attach, or None for no attachment.
    recipient_email : str
        Recipient email address. Pass empty string for manual-route applications.

    Returns
    -------
    str
        Gmail draft deeplink URL (``https://mail.google.com/mail/u/0/#drafts/<id>``).

    Raises
    ------
    ConnectionError
        If the Gmail MCP server is unreachable.
    RuntimeError
        If the MCP server returns an error or no draft ID.
    """
    # Use asyncio.run() to bridge sync → async.
    # This is safe because gmail_sender is only called from the synchronous
    # LangGraph pipeline, which does not have a running event loop.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        # We're already inside an async context (shouldn't happen in normal
        # pipeline flow, but handle gracefully for tests / notebooks).
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            draft_id = pool.submit(
                asyncio.run,
                _create_draft_via_mcp(subject, body, recipient_email, pdf_path),
            ).result()
    else:
        draft_id = asyncio.run(
            _create_draft_via_mcp(subject, body, recipient_email, pdf_path)
        )

    draft_url = _GMAIL_DRAFT_URL_TEMPLATE.format(draft_id=draft_id)
    logger.info("Draft URL: %s", draft_url)
    return draft_url
