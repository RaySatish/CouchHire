"""Offline tests for dashboard/helpers.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dashboard import helpers


def test_applications_to_dataframe_basic() -> None:
    df = helpers.applications_to_dataframe(
        [{"company": "Acme", "role": "ML", "created_at": "2026-01-01T00:00:00Z"}]
    )
    assert isinstance(df, pd.DataFrame)
    assert "date" in df.columns


def test_compute_summary_stats_empty() -> None:
    stats = helpers.compute_summary_stats(pd.DataFrame())
    assert stats["total"] == 0


def test_form_answers_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "form_answers.json"
    assert helpers.save_form_answers(path, {"x": 1}) is True
    assert helpers.load_form_answers(path)["x"] == 1
