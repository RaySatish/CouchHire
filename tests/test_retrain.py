"""Offline tests for nlp/retrain.py."""

from __future__ import annotations

import pytest

from nlp import retrain


def test_calculate_epochs_ranges() -> None:
    assert retrain._calculate_epochs(10) == 8
    assert retrain._calculate_epochs(40) == 5
    assert retrain._calculate_epochs(90) == 3
    assert retrain._calculate_epochs(150) == 2


def test_should_retrain_false_when_fetch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(__import__("os").environ, "RETRAIN_EVERY", "10")
    monkeypatch.setitem(__import__("os").environ, "MIN_RETRAIN_LABELS", "10")
    monkeypatch.setattr("db.supabase_client.get_labeled_outcomes", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert retrain.should_retrain() is False
