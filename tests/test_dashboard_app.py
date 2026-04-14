"""Static structure checks for dashboard/app.py."""

from __future__ import annotations

from pathlib import Path


def test_page_config_present() -> None:
    source = Path("dashboard/app.py").read_text(encoding="utf-8")
    assert "st.set_page_config(" in source
    assert "page_title=\"CouchHire Dashboard\"" in source


def test_all_four_tabs_exist() -> None:
    source = Path("dashboard/app.py").read_text(encoding="utf-8")
    assert "Tracker" in source
    assert "Analytics" in source
    assert "Retrain" in source
    assert "Settings" in source
