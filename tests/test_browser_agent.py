"""Tests for apply/browser_agent.py — form field matching and helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from apply.browser_agent import (
    _fuzzy_match_option,
    _load_form_answers,
    _save_form_answer,
    _parse_json_response,
)


# ── _fuzzy_match_option ───────────────────────────────────────────────────

class TestFuzzyMatchOption:
    def test_exact_match(self):
        options = [
            {"value": "IN", "text": "India"},
            {"value": "US", "text": "United States"},
        ]
        result = _fuzzy_match_option("India", options)
        assert result is not None
        assert result["value"] == "IN"

    def test_case_insensitive(self):
        options = [{"value": "IN", "text": "India"}]
        result = _fuzzy_match_option("india", options)
        assert result is not None
        assert result["value"] == "IN"

    def test_substring_match(self):
        options = [
            {"value": "IN", "text": "India (+91)"},
            {"value": "US", "text": "United States (+1)"},
        ]
        result = _fuzzy_match_option("India", options)
        assert result is not None
        assert result["value"] == "IN"

    def test_starts_with_match(self):
        options = [
            {"value": "IN", "text": "India - Republic of India"},
        ]
        result = _fuzzy_match_option("India", options)
        assert result is not None

    def test_no_match_returns_none(self):
        options = [{"value": "US", "text": "United States"}]
        result = _fuzzy_match_option("Narnia", options)
        assert result is None

    def test_empty_options(self):
        result = _fuzzy_match_option("India", [])
        assert result is None

    def test_word_overlap_scoring(self):
        options = [
            {"value": "1", "text": "New York City"},
            {"value": "2", "text": "Los Angeles"},
        ]
        result = _fuzzy_match_option("New York", options)
        assert result is not None
        assert result["value"] == "1"


# ── _load_form_answers ────────────────────────────────────────────────────

class TestLoadFormAnswers:
    def test_missing_file_returns_empty(self, tmp_path):
        with patch("config.FORM_ANSWERS_PATH", tmp_path / "nonexistent.json"):
            result = _load_form_answers()
            assert result == {}

    def test_valid_file_returns_dict(self, tmp_path):
        answers_file = tmp_path / "form_answers.json"
        answers_file.write_text(json.dumps({"q1": "a1"}))
        with patch("config.FORM_ANSWERS_PATH", answers_file):
            result = _load_form_answers()
            assert result == {"q1": "a1"}

    def test_invalid_json_returns_empty(self, tmp_path):
        answers_file = tmp_path / "form_answers.json"
        answers_file.write_text("not json {{{")
        with patch("config.FORM_ANSWERS_PATH", answers_file):
            result = _load_form_answers()
            assert result == {}


# ── _parse_json_response ──────────────────────────────────────────────────

class TestParseJsonResponse:
    def test_valid_json_array(self):
        raw = '[{"selector": "input", "value": "test"}]'
        result = _parse_json_response(raw)
        assert result is not None
        assert len(result) == 1

    def test_json_in_code_fences(self):
        raw = '```json\n[{"selector": "input", "value": "test"}]\n```'
        result = _parse_json_response(raw)
        assert result is not None

    def test_invalid_json_returns_none(self):
        result = _parse_json_response("not json at all")
        assert result is None

    def test_empty_string_returns_none(self):
        result = _parse_json_response("")
        assert result is None
