"""Offline tests for jobs/job_search.py."""

from __future__ import annotations

from types import SimpleNamespace

from jobs import job_search


def test_row_to_dict_contains_required_keys() -> None:
    row = {
        "title": "ML Engineer",
        "company": "Acme",
        "job_url": "https://example.com/job/1",
        "job_url_direct": "https://apply.example.com/1",
        "city": "Bengaluru",
        "state": "KA",
        "country": "India",
        "description": "desc",
        "min_amount": 10,
        "max_amount": 20,
        "interval": "hour",
        "is_remote": True,
        "date_posted": "2026-01-01",
        "site": "linkedin",
        "job_type": "fulltime",
    }
    pd = SimpleNamespace(isna=lambda x: False)
    out = job_search._row_to_dict(row, pd)
    for key in ("title", "company", "url", "description", "site", "job_type"):
        assert key in out


def test_get_job_details_passthrough() -> None:
    url = "https://example.com/job/1"
    out = job_search.get_job_details(url)
    assert out["url"] == url
