"""CouchHire — Score and filter job search results against the user's CV.

Takes raw job search results from JobSpy, scores each against the user's
CV using the match scorer, filters to those above threshold, and formats
them for Telegram display.

Scoring pipeline:
    For each job → cv_rag.retrieve_cv_sections() → match_scorer.score() → filter → sort

This is CPU-heavy (runs sentence-transformer inference per job). Call from
a background thread when invoked from the Telegram bot.
"""

from __future__ import annotations

import html
import logging

logger = logging.getLogger(__name__)


def filter_and_score(jobs: list[dict]) -> list[dict]:
    """Score and filter job results against the user's CV.

    For each job:
    1. Use the job's snippet/description as a mini-JD
    2. Retrieve relevant CV sections via cv_rag
    3. Score the match
    4. Keep only jobs above MIN_MATCH_SCORE

    Args:
        jobs: list of job dicts from job_search.search_jobs().

    Returns:
        list of job dicts, each augmented with:
            - "match_score": float (0-100)
            - "cv_sections": list[str] — the CV chunks used for scoring
        Sorted by match_score descending.
        Limited to MAX_SEARCH_RESULTS items.
    """
    from config import MIN_MATCH_SCORE, MAX_SEARCH_RESULTS
    from agents.cv_rag import retrieve_cv_sections
    from agents.match_scorer import score

    scored_jobs = []
    for job in jobs:
        # Use snippet or description as the JD text for scoring
        jd_text = job.get("description") or job.get("snippet") or ""
        if not jd_text:
            logger.warning("Skipping job with no description: %s", job.get("title"))
            continue

        # Build a minimal requirements dict for cv_rag.retrieve_cv_sections
        # It expects {"role": str|None, "skills": list[str]}
        # We use the job title as role and extract no skills (let the embedder
        # match on the full text via the role field)
        requirements = {
            "role": job.get("title", ""),
            "skills": [],
        }

        # Retrieve relevant CV sections
        try:
            cv_sections = retrieve_cv_sections(requirements)
        except FileNotFoundError:
            logger.error(
                "ChromaDB store not found. Run 'python cv/embed_cv.py' first. "
                "Cannot score jobs without embedded CV."
            )
            return []
        except Exception as exc:
            logger.warning(
                "Failed to retrieve CV sections for '%s': %s",
                job.get("title"),
                exc,
            )
            continue

        # Score the match
        match_score = score(jd_text, cv_sections)

        if match_score >= MIN_MATCH_SCORE:
            job["match_score"] = match_score
            job["cv_sections"] = cv_sections
            scored_jobs.append(job)
        else:
            logger.debug(
                "Filtered out: %s at %s (score=%.1f < %.1f)",
                job.get("title"),
                job.get("company"),
                match_score,
                MIN_MATCH_SCORE,
            )

    # Sort by score descending, limit results
    scored_jobs.sort(key=lambda j: j["match_score"], reverse=True)
    scored_jobs = scored_jobs[:MAX_SEARCH_RESULTS]

    logger.info(
        "Filtered %d jobs → %d above %.0f%% threshold",
        len(jobs),
        len(scored_jobs),
        MIN_MATCH_SCORE,
    )

    return scored_jobs



def format_unscored_list(jobs: list[dict]) -> str:
    """Format unscored jobs (no descriptions) into a Telegram HTML message.

    Used as a fallback when JobSpy returns jobs without description text,
    making CV-based scoring impossible. Shows titles, companies, and links
    so the user can still browse results manually.

    Args:
        jobs: list of raw job dicts from job_search.search_jobs().

    Returns:
        Formatted HTML string for Telegram.
    """
    if not jobs:
        return "😕 No matching jobs found."

    lines = [
        f"🔍 <b>Found {len(jobs)} job{'s' if len(jobs) != 1 else ''}</b> "
        f"(unscored — descriptions unavailable):\n"
    ]

    for i, job in enumerate(jobs[:10], 1):
        title = html.escape(job.get("title", "Unknown"))
        company = html.escape(job.get("company", "Unknown"))
        location = job.get("location", "")
        url = job.get("url", "")

        line = f"{i}. <b>{title}</b> — {company}"
        if location:
            line += f"\n   📍 {html.escape(location)}"
        if url:
            line += f"\n   🔗 <a href=\"{url}\">View Job</a>"
        lines.append(line)

    lines.append(
        "\n⚠️ <i>Job boards returned no descriptions, so match scoring "
        "was skipped. Browse manually or try different keywords.</i>"
    )

    return "\n\n".join(lines)


def format_job_list(scored_jobs: list[dict]) -> str:
    """Format scored jobs into a Telegram HTML message.

    Returns a formatted string like:
        🔍 <b>Found 5 matching jobs:</b>

        1. <b>Senior Backend Engineer</b> — Google
           📍 Remote | 💰 $150k-$200k | 📊 87.5%

        2. <b>Platform Engineer</b> — Stripe
           📍 San Francisco | 📊 82.0%
    """
    if not scored_jobs:
        return "😕 No matching jobs found."

    lines = [f"🔍 <b>Found {len(scored_jobs)} matching job{'s' if len(scored_jobs) != 1 else ''}:</b>\n"]

    for i, job in enumerate(scored_jobs, 1):
        title = html.escape(job.get("title", "Unknown"))
        company = html.escape(job.get("company", "Unknown"))
        score_val = job.get("match_score", 0)
        location = job.get("location", "")
        salary = job.get("salary", "")

        line = f"{i}. <b>{title}</b> — {company}\n   "

        details = []
        if location:
            details.append(f"📍 {html.escape(location)}")
        if salary:
            details.append(f"💰 {html.escape(str(salary))}")
        details.append(f"📊 {score_val:.1f}%")

        line += " | ".join(details)
        lines.append(line)

    return "\n\n".join(lines)
