"""Offline tests for jobs/job_filter.py."""

from __future__ import annotations

from jobs import job_filter


def test_format_job_list_returns_html() -> None:
    jobs = [{"title": "ML Engineer", "company": "Acme", "match_score": 88.0}]
    text = job_filter.format_job_list(jobs)
    assert "Found 1 matching job" in text
    assert "88.0%" in text


def test_format_unscored_list_handles_empty() -> None:
    assert "No matching jobs" in job_filter.format_unscored_list([])
