"""Offline tests for db/supabase_client.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from db import supabase_client


def test_insert_application_returns_row(mock_supabase_client: MagicMock) -> None:
    row = supabase_client.insert_application({"company": "X"})
    assert isinstance(row, dict)
    assert row["id"] == "test-uuid-1234"


def test_update_application_raises_when_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    response = MagicMock()
    response.data = []
    client.table.return_value.update.return_value.eq.return_value.execute.return_value = response
    monkeypatch.setattr(supabase_client, "_get_client", lambda: client)
    with pytest.raises(ValueError):
        supabase_client.update_application("missing", {"status": "applied"})


def test_get_all_applications_returns_list(mock_supabase_client: MagicMock) -> None:
    rows = supabase_client.get_all_applications(limit=10)
    assert isinstance(rows, list)
