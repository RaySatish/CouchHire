"""Offline tests for apply/session_handoff.py."""

from __future__ import annotations

import types

import pytest

from apply import session_handoff


def test_get_cdp_url_raises_for_missing_session() -> None:
    with pytest.raises(RuntimeError):
        session_handoff.get_cdp_url("missing-session")


def test_get_takeover_instructions_headless_false() -> None:
    session_handoff._SESSIONS["s1"] = {
        "process": types.SimpleNamespace(poll=lambda: None),
        "cdp_port": 9222,
        "headless": False,
    }
    try:
        text = session_handoff.get_takeover_instructions("s1")
        assert "Session: s1" in text
        assert "browser window is visible" in text
    finally:
        session_handoff._SESSIONS.pop("s1", None)


def test_close_browser_cleans_up_session() -> None:
    proc = types.SimpleNamespace(
        poll=lambda: 0,
        terminate=lambda: None,
        wait=lambda timeout=None: None,
    )
    session_handoff._SESSIONS["s2"] = {
        "process": proc,
        "cdp_port": 9222,
        "headless": True,
    }
    session_handoff.close_browser("s2")
    assert "s2" not in session_handoff._SESSIONS
