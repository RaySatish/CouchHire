"""Offline tests for apply/gmail_sender.py."""

from __future__ import annotations

import types

import pytest

from apply import gmail_sender


def test_extract_draft_id_from_text_block() -> None:
    result = types.SimpleNamespace(
        content=[types.SimpleNamespace(text="Draft created! Draft ID: r12345")]
    )
    assert gmail_sender._extract_draft_id(result) == "r12345"


def test_extract_draft_id_from_dict() -> None:
    assert gmail_sender._extract_draft_id({"draft_id": "abc"}) == "abc"


def test_create_draft_returns_tuple(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gmail_sender,
        "_run_async",
        lambda _coro: ("r123", "19d768e1dce66005"),
    )
    draft_url, draft_id = gmail_sender.create_draft(
        subject="Subject",
        body="Body",
        recipient_email="hr@example.com",
        attachments=["/tmp/a.pdf"],
    )
    assert isinstance(draft_url, str)
    assert draft_url.endswith("#all/19d768e1dce66005")
    assert draft_id == "r123"


def test_send_draft_raises_runtime_error() -> None:
    with pytest.raises(RuntimeError):
        gmail_sender.send_draft("r123")
