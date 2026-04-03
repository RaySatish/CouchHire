"""CouchHire — Supabase client for the applications table.

Provides CRUD functions for logging and querying job applications.
All functions are synchronous and use the Supabase Python client.
"""

import logging
from typing import Any

from postgrest.exceptions import APIError
from supabase import Client, create_client

from config import SUPABASE_KEY, SUPABASE_URL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Table name — single source of truth
# ---------------------------------------------------------------------------
_TABLE = "applications"


# ---------------------------------------------------------------------------
# Client factory (lazy singleton)
# ---------------------------------------------------------------------------
_client: Client | None = None


def _get_client() -> Client:
    """Return a cached Supabase client, creating it on first call."""
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.debug("Supabase client initialised for %s", SUPABASE_URL)
    return _client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def insert_application(data: dict[str, Any]) -> dict[str, Any]:
    """Insert a new application row and return the created record.

    Args:
        data: Column-value mapping. At minimum one of ``jd_text``, ``jd_url``,
              or ``role_input`` must be present (enforced by the DB constraint).

    Returns:
        The full row as a dict (includes server-generated ``id``, ``created_at``, etc.).

    Raises:
        APIError: If the Supabase insert fails (e.g. constraint violation).
    """
    client = _get_client()
    try:
        response = client.table(_TABLE).insert(data).execute()
        row = response.data[0]
        logger.info(
            "Inserted application %s — company=%s role=%s",
            row.get("id"),
            row.get("company"),
            row.get("role"),
        )
        return row
    except APIError as exc:
        logger.error("Failed to insert application: %s | data=%s", exc, data)
        raise


def update_application(
    application_id: str,
    updates: dict[str, Any],
) -> dict[str, Any]:
    """Update an existing application row by its UUID.

    Args:
        application_id: The UUID primary key of the row.
        updates: Column-value mapping of fields to update.

    Returns:
        The updated row as a dict.

    Raises:
        APIError: If the Supabase update fails.
        ValueError: If no matching row is found.
    """
    client = _get_client()
    try:
        response = (
            client.table(_TABLE)
            .update(updates)
            .eq("id", application_id)
            .execute()
        )
        if not response.data:
            raise ValueError(
                f"No application found with id={application_id}"
            )
        row = response.data[0]
        logger.info(
            "Updated application %s — fields=%s",
            application_id,
            list(updates.keys()),
        )
        return row
    except APIError as exc:
        logger.error(
            "Failed to update application %s: %s | updates=%s",
            application_id,
            exc,
            updates,
        )
        raise


def update_status(application_id: str, status: str) -> dict[str, Any]:
    """Convenience wrapper: update only the status column.

    Args:
        application_id: The UUID primary key.
        status: One of the CHECK-constrained status values
                (pending, scraping, parsing, scoring, below_threshold,
                tailoring, drafting, awaiting_review, applied, error).

    Returns:
        The updated row as a dict.
    """
    return update_application(application_id, {"status": status})


def update_outcome(application_id: str, outcome: str) -> dict[str, Any]:
    """Convenience wrapper: set the outcome label for retraining.

    Args:
        application_id: The UUID primary key.
        outcome: One of the CHECK-constrained outcome values
                 (interview, rejected, no_response, offer, withdrawn).

    Returns:
        The updated row as a dict.
    """
    return update_application(application_id, {"outcome": outcome})


def get_all_applications(
    *,
    limit: int = 100,
    offset: int = 0,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch applications ordered by created_at descending.

    Args:
        limit: Maximum rows to return (default 100).
        offset: Number of rows to skip (for pagination).
        status: If provided, filter to only this status value.

    Returns:
        A list of application dicts, newest first.
    """
    client = _get_client()
    try:
        query = (
            client.table(_TABLE)
            .select("*")
            .order("created_at", desc=True)
        )
        if status is not None:
            query = query.eq("status", status)
        query = query.range(offset, offset + limit - 1)

        response = query.execute()
        logger.debug(
            "Fetched %d applications (limit=%d, offset=%d, status=%s)",
            len(response.data),
            limit,
            offset,
            status,
        )
        return response.data
    except APIError as exc:
        logger.error("Failed to fetch applications: %s", exc)
        raise


def get_labeled_outcomes() -> list[dict[str, Any]]:
    """Fetch all applications that have an outcome label set.

    Used by ``nlp/retrain.py`` to retrain the match scorer with real
    application results.

    Returns:
        A list of application dicts where ``outcome IS NOT NULL``,
        ordered by created_at descending.
    """
    client = _get_client()
    try:
        response = (
            client.table(_TABLE)
            .select("*")
            .not_.is_("outcome", "null")
            .order("created_at", desc=True)
            .execute()
        )
        logger.debug(
            "Fetched %d labeled outcomes for retraining", len(response.data)
        )
        return response.data
    except APIError as exc:
        logger.error("Failed to fetch labeled outcomes: %s", exc)
        raise


def get_application_by_id(application_id: str) -> dict[str, Any] | None:
    """Fetch a single application by its UUID.

    Args:
        application_id: The UUID primary key.

    Returns:
        The application dict, or None if not found.
    """
    client = _get_client()
    try:
        response = (
            client.table(_TABLE)
            .select("*")
            .eq("id", application_id)
            .maybe_single()
            .execute()
        )
        return response.data
    except APIError as exc:
        logger.error(
            "Failed to fetch application %s: %s", application_id, exc
        )
        raise
