"""Offline tests for bot/telegram_bot.py."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from bot import telegram_bot


def test_escape_html_escapes_special_chars() -> None:
    assert telegram_bot._escape_html("<x&y>") == "&lt;x&amp;y&gt;"


def test_command_handlers_are_callable() -> None:
    assert callable(telegram_bot._handle_apply)
    assert callable(telegram_bot._handle_outcome)
    assert callable(telegram_bot._handle_search)
    assert callable(telegram_bot._handle_callback)


def test_callback_gate_sets_pending_interrupt() -> None:
    event = SimpleNamespace(set=lambda: None)
    telegram_bot._pending_interrupt = {"event": event, "response": None}

    answered: list[str] = []

    async def _answer(msg: str) -> None:
        answered.append(msg)

    async def _edit_message_text(*_args, **_kwargs) -> None:
        return None

    query = SimpleNamespace(
        data="gate1_approve",
        answer=_answer,
        edit_message_text=_edit_message_text,
        message=SimpleNamespace(text_html="msg"),
    )
    update = SimpleNamespace(callback_query=query)

    try:
        asyncio.run(telegram_bot._handle_callback(update, context=None))
        assert answered
    finally:
        telegram_bot._pending_interrupt = None
