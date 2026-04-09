"""CouchHire — Job search via JobSpy (python-jobspy).

Searches multiple job boards (Indeed, LinkedIn, Glassdoor, Google, ZipRecruiter)
concurrently using the JobSpy library. No API keys or OAuth required.

Public API:
    search_jobs(query, location, job_type, remote, country, results_wanted, hours_old) -> list[dict]
    get_job_details(job_url) -> dict  (returns what we already have — JobSpy gives full descriptions)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def search_jobs(
    query: str,
    location: str = "",
    job_type: str = "",
    remote: bool = False,
    country: str = "USA",
    results_wanted: int = 25,
    hours_old: int | None = 72,
    site_names: list[str] | None = None,
) -> list[dict]:
    """Search multiple job boards via JobSpy and return normalised results."""
    from jobspy import scrape_jobs  # lazy import — avoids error if not installed
    import pandas as pd

    if site_names is None:
        try:
            from config import JOBSPY_SITES
            site_names = JOBSPY_SITES
        except ImportError:
            site_names = ["indeed", "linkedin", "google"]

    # Map job_type to JobSpy expected values
    _valid_types = {"fulltime", "parttime", "internship", "contract"}
    mapped_type = job_type.lower().strip() if job_type else ""
    if mapped_type and mapped_type not in _valid_types:
        logger.warning("Unknown job_type '%s', ignoring. Valid: %s", job_type, _valid_types)
        mapped_type = ""

    # Build kwargs — only pass optional params when they have values
    kwargs: dict[str, Any] = {
        "site_name": site_names,
        "search_term": query,
        "results_wanted": results_wanted,
        "country_indeed": country,
        "description_format": "markdown",
        "verbose": 0,
    }
    if location:
        kwargs["location"] = location
    if mapped_type:
        kwargs["job_type"] = mapped_type
    if remote:
        kwargs["is_remote"] = True
    if hours_old is not None:
        kwargs["hours_old"] = hours_old

    # Inject proxies from config if available
    try:
        from config import JOBSPY_PROXIES
        if JOBSPY_PROXIES:
            kwargs["proxies"] = JOBSPY_PROXIES
    except ImportError:
        pass

    try:
        df = scrape_jobs(**kwargs)
    except Exception as exc:
        logger.error("JobSpy scrape_jobs() failed: %s", exc, exc_info=True)
        return []

    if df is None or df.empty:
        logger.info("JobSpy returned 0 results from %s for query '%s'", site_names, query)
        return []

    results: list[dict] = []
    for _, row in df.iterrows():
        results.append(_row_to_dict(row, pd))

    logger.info("JobSpy returned %d results from %s for query '%s'", len(results), site_names, query)
    return results


def _row_to_dict(row: Any, pd: Any) -> dict:
    """Convert a single DataFrame row to a normalised job dict."""

    def _safe(val: Any) -> str:
        """Return empty string for NaN/None, else str(val)."""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ""
        try:
            if pd.isna(val):
                return ""
        except (TypeError, ValueError):
            pass
        return str(val)

    # Build location from city, state, country
    loc_parts = [_safe(row.get("city")), _safe(row.get("state")), _safe(row.get("country"))]
    location = ", ".join(p for p in loc_parts if p)

    # Build salary string
    min_amt = row.get("min_amount")
    max_amt = row.get("max_amount")
    interval = _safe(row.get("interval"))
    salary = ""
    min_s = _safe(min_amt)
    max_s = _safe(max_amt)
    if min_s and max_s:
        salary = f"${min_s} - ${max_s}"
        if interval:
            salary += f" {interval}"
    elif min_s:
        salary = f"${min_s}"
        if interval:
            salary += f" {interval}"
    elif max_s:
        salary = f"${max_s}"
        if interval:
            salary += f" {interval}"

    # is_remote — coerce to bool
    is_remote_raw = row.get("is_remote")
    is_remote = bool(is_remote_raw) if is_remote_raw is not None else False

    return {
        "title": _safe(row.get("title")),
        "company": _safe(row.get("company")),
        "url": _safe(row.get("job_url")),
        "location": location,
        "description": _safe(row.get("description")),
        "salary": salary,
        "is_remote": is_remote,
        "date_posted": _safe(row.get("date_posted")),
        "site": _safe(row.get("site")),
        "job_type": _safe(row.get("job_type")),
    }


def get_job_details(job_url: str) -> dict:
    """Return a stub dict for the given job URL.

    JobSpy already provides full descriptions in search_jobs(), so this
    function exists only as a compatibility passthrough.  The Telegram
    callback handler should use the cached description from search results
    instead of calling this.
    """
    logger.debug(
        "get_job_details called for %s — full description should already "
        "be in the search results cache.",
        job_url,
    )
    return {"url": job_url, "description": ""}
