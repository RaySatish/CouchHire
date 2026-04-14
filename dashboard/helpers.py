"""CouchHire Dashboard — helper utilities for data transformation and display.

Keeps dashboard/app.py focused on layout and interaction. All heavy data
munging, formatting, and config I/O lives here.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATUS_OPTIONS: list[str] = [
    "pending",
    "scraping",
    "parsing",
    "scoring",
    "below_threshold",
    "tailoring",
    "drafting",
    "awaiting_review",
    "applied",
    "error",
]

OUTCOME_OPTIONS: list[str] = [
    "interview",
    "rejected",
    "no_response",
    "offer",
    "withdrawn",
]

ROUTE_OPTIONS: list[str] = ["email", "form", "manual"]

# Status → emoji mapping for visual clarity
STATUS_EMOJI: dict[str, str] = {
    "pending": "⏳",
    "scraping": "🔍",
    "parsing": "📝",
    "scoring": "📊",
    "below_threshold": "📉",
    "tailoring": "✂️",
    "drafting": "📧",
    "awaiting_review": "👀",
    "applied": "✅",
    "error": "❌",
}

OUTCOME_EMOJI: dict[str, str] = {
    "interview": "🎯",
    "rejected": "🚫",
    "no_response": "🤷",
    "offer": "🎉",
    "withdrawn": "↩️",
}

OUTCOME_COLORS: dict[str, str] = {
    "interview": "#4CAF50",
    "rejected": "#F44336",
    "no_response": "#9E9E9E",
    "offer": "#2196F3",
    "withdrawn": "#FF9800",
}

ROUTE_COLORS: dict[str, str] = {
    "email": "#2196F3",
    "form": "#4CAF50",
    "manual": "#FF9800",
}


# ---------------------------------------------------------------------------
# Data transformation
# ---------------------------------------------------------------------------


def applications_to_dataframe(apps: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of application dicts to a cleaned DataFrame.

    Args:
        apps: Raw application dicts from Supabase.

    Returns:
        A DataFrame with parsed dates and filled NaN values.
    """
    if not apps:
        return pd.DataFrame()

    df = pd.DataFrame(apps)

    # Parse timestamps
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        df["date"] = df["created_at"].dt.date

    if "updated_at" in df.columns:
        df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True)

    # Fill display columns with sensible defaults
    for col in ("company", "role", "status", "outcome", "route"):
        if col in df.columns:
            df[col] = df[col].fillna("—")

    if "match_score" in df.columns:
        df["match_score"] = pd.to_numeric(df["match_score"], errors="coerce")

    return df


def compute_summary_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute summary statistics from the applications DataFrame.

    Returns:
        Dict with keys: total, avg_score, interview_rate, offer_rate,
        applied_count, active_count.
    """
    if df.empty:
        return {
            "total": 0,
            "avg_score": 0.0,
            "interview_rate": 0.0,
            "offer_rate": 0.0,
            "applied_count": 0,
            "active_count": 0,
        }

    total = len(df)

    # Average match score (exclude NaN)
    avg_score = df["match_score"].mean() if "match_score" in df.columns else 0.0
    avg_score = round(avg_score, 1) if pd.notna(avg_score) else 0.0

    # Outcome-based rates (only among labeled applications)
    labeled = df[df["outcome"].isin(OUTCOME_OPTIONS)] if "outcome" in df.columns else pd.DataFrame()
    labeled_count = len(labeled)

    if labeled_count > 0:
        interview_count = len(labeled[labeled["outcome"].isin(["interview", "offer"])])
        offer_count = len(labeled[labeled["outcome"] == "offer"])
        interview_rate = round(interview_count / labeled_count * 100, 1)
        offer_rate = round(offer_count / labeled_count * 100, 1)
    else:
        interview_rate = 0.0
        offer_rate = 0.0

    applied_count = len(df[df["status"] == "applied"]) if "status" in df.columns else 0
    active_count = len(df[~df["status"].isin(["error", "below_threshold", "applied"])]) if "status" in df.columns else 0

    return {
        "total": total,
        "avg_score": avg_score,
        "interview_rate": interview_rate,
        "offer_rate": offer_rate,
        "applied_count": applied_count,
        "active_count": active_count,
        "labeled_count": labeled_count,
    }


def format_score_badge(score: float | None) -> str:
    """Return a colored score string for display."""
    if score is None or pd.isna(score):
        return "—"
    score = float(score)
    if score >= 80:
        return f"🟢 {score:.0f}"
    elif score >= 60:
        return f"🟡 {score:.0f}"
    else:
        return f"🔴 {score:.0f}"


def format_status(status: str) -> str:
    """Return status with emoji prefix."""
    emoji = STATUS_EMOJI.get(status, "")
    return f"{emoji} {status}" if emoji else status


def format_outcome(outcome: str) -> str:
    """Return outcome with emoji prefix."""
    if outcome in ("—", "", None):
        return "—"
    emoji = OUTCOME_EMOJI.get(outcome, "")
    return f"{emoji} {outcome}" if emoji else outcome


# ---------------------------------------------------------------------------
# Form answers I/O
# ---------------------------------------------------------------------------


def load_form_answers(path: Path) -> dict[str, Any]:
    """Load form_answers.json, returning empty dict if missing or invalid.

    Args:
        path: Path to the form_answers.json file.

    Returns:
        The parsed JSON as a dict.
    """
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load form_answers.json: %s", exc)
    return {}


def save_form_answers(path: Path, data: dict[str, Any]) -> bool:
    """Save form answers dict back to disk.

    Args:
        path: Path to the form_answers.json file.
        data: The dict to serialize.

    Returns:
        True if saved successfully, False otherwise.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return True
    except OSError as exc:
        logger.error("Failed to save form_answers.json: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Settings I/O (config.yaml)
# ---------------------------------------------------------------------------

# The prompt specifies config.yaml but the project uses .env.
# We create a lightweight config.yaml for dashboard-editable settings
# that don't require a restart (match threshold, LLM prefs, etc.).
# The actual .env is never read/displayed (security rules).

_DEFAULT_SETTINGS: dict[str, Any] = {
    "match_threshold": 60,
    "min_retrain_labels": 10,
    "retrain_every": 10,
    "min_match_score": 60,
    "max_search_results": 10,
    "jobspy_sites": "indeed,linkedin,google",
    "jobspy_country": "USA",
    "jobspy_hours_old": 72,
    "browser_headless": False,
}


def load_settings(path: Path) -> dict[str, Any]:
    """Load config.yaml settings, returning defaults if missing.

    Args:
        path: Path to config.yaml.

    Returns:
        Settings dict.
    """
    try:
        if path.exists():
            import yaml
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                # Merge with defaults so new keys are always present
                merged = {**_DEFAULT_SETTINGS, **data}
                return merged
    except Exception as exc:
        logger.warning("Failed to load config.yaml: %s", exc)
    return dict(_DEFAULT_SETTINGS)


def save_settings(path: Path, data: dict[str, Any]) -> bool:
    """Save settings dict to config.yaml.

    Args:
        path: Path to config.yaml.
        data: Settings dict to save.

    Returns:
        True if saved successfully.
    """
    try:
        import yaml
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.dump(data, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        return True
    except Exception as exc:
        logger.error("Failed to save config.yaml: %s", exc)
        return False
