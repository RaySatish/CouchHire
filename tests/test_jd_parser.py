"""Tests for agents/jd_parser.py — JD parsing, validation, and skill merging."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from agents.jd_parser import (
    _DEFAULT_REQUIREMENTS,
    _merge_skills,
    _strip_markdown_fences,
    _validate_and_normalise,
    _filter_noise_skills,
    parse_jd,
)


# ── _strip_markdown_fences ────────────────────────────────────────────────

class TestStripMarkdownFences:
    def test_plain_json(self):
        raw = '{"company": "Acme"}'
        assert _strip_markdown_fences(raw) == raw

    def test_json_code_fence(self):
        raw = '```json\n{"company": "Acme"}\n```'
        assert _strip_markdown_fences(raw) == '{"company": "Acme"}'

    def test_bare_code_fence(self):
        raw = '```\n{"company": "Acme"}\n```'
        assert _strip_markdown_fences(raw) == '{"company": "Acme"}'

    def test_think_tags_removed(self):
        raw = '<think>reasoning here</think>\n{"company": "Acme"}'
        result = _strip_markdown_fences(raw)
        assert "think" not in result.lower()
        assert '"company"' in result

    def test_unclosed_think_tag(self):
        raw = '<think>long reasoning without closing tag {"company": "Acme"}'
        result = _strip_markdown_fences(raw)
        assert "{" in result

    def test_whitespace_stripped(self):
        raw = '  \n  {"company": "Acme"}  \n  '
        assert _strip_markdown_fences(raw).strip() == '{"company": "Acme"}'


# ── _merge_skills ─────────────────────────────────────────────────────────

class TestMergeSkills:
    def test_deduplication_case_insensitive(self):
        result = _merge_skills(["Python", "Docker"], ["python", "React"])
        assert len(result) == 3
        assert "Python" in result
        assert "React" in result

    def test_empty_inputs(self):
        assert _merge_skills([], []) == []

    def test_preserves_order(self):
        result = _merge_skills(["A", "B"], ["C"])
        assert result == ["A", "B", "C"]

    def test_strips_whitespace(self):
        result = _merge_skills(["  Python  "], ["  Docker  "])
        assert result == ["Python", "Docker"]

    def test_skips_empty_strings(self):
        result = _merge_skills(["Python", "", "  "], ["Docker"])
        assert result == ["Python", "Docker"]


# ── _validate_and_normalise ───────────────────────────────────────────────

class TestValidateAndNormalise:
    def test_all_keys_present(self):
        result = _validate_and_normalise({})
        for key in _DEFAULT_REQUIREMENTS:
            assert key in result

    def test_invalid_apply_method_defaults_to_unknown(self):
        result = _validate_and_normalise({"apply_method": "carrier_pigeon"})
        assert result["apply_method"] == "unknown"

    def test_valid_apply_methods(self):
        for method in ("email", "url", "unknown"):
            result = _validate_and_normalise({"apply_method": method})
            assert result["apply_method"] == method

    def test_booleans_coerced(self):
        result = _validate_and_normalise({"cover_letter_required": 1, "github_requested": 0})
        assert result["cover_letter_required"] is True
        assert result["github_requested"] is False

    def test_skills_list_of_strings(self):
        result = _validate_and_normalise({"skills": ["Python", 42, None]})
        assert "Python" in result["skills"]
        assert "42" in result["skills"]

    def test_skills_non_list_becomes_empty(self):
        result = _validate_and_normalise({"skills": "Python"})
        assert result["skills"] == []

    def test_string_fields_converted(self):
        result = _validate_and_normalise({"company": 123, "role": None})
        assert result["company"] == "123"
        assert result["role"] is None


# ── _filter_noise_skills ──────────────────────────────────────────────────

class TestFilterNoiseSkills:
    def test_removes_company_name(self):
        result = _filter_noise_skills(["Python", "Acme"], "Acme", "Engineer")
        assert "Python" in result
        assert "Acme" not in result

    def test_removes_generic_words(self):
        result = _filter_noise_skills(["Python", "data", "resume"], None, None)
        assert "Python" in result
        assert "resume" not in result

    def test_removes_single_char(self):
        result = _filter_noise_skills(["a", "Python"], None, None)
        assert "a" not in result

    def test_removes_substrings_of_multiword(self):
        result = _filter_noise_skills(["machine", "machine learning"], None, None)
        assert "machine learning" in result
        assert "machine" not in result


# ── parse_jd (integration with mocked LLM) ───────────────────────────────

class TestParseJd:
    def test_empty_input_returns_defaults(self):
        result = parse_jd("")
        assert result == _DEFAULT_REQUIREMENTS

    def test_whitespace_only_returns_defaults(self):
        result = parse_jd("   \n\t  ")
        assert result == _DEFAULT_REQUIREMENTS

    @patch("agents.jd_parser.extract_skills", return_value=["Docker"])
    @patch("agents.jd_parser.complete")
    def test_successful_parse(self, mock_complete, mock_ner, sample_jd_text):
        mock_complete.return_value = json.dumps({
            "company": "QuantumLeap AI",
            "role": "ML Engineer",
            "skills": ["Python", "PyTorch"],
            "apply_method": "email",
            "apply_target": "hr@quantumleap.ai",
            "cover_letter_required": False,
            "subject_line_format": None,
            "email_instructions": None,
            "github_requested": False,
            "form_fields": [],
        })
        result = parse_jd(sample_jd_text)
        assert result["company"] == "QuantumLeap AI"
        assert result["role"] == "ML Engineer"
        assert result["apply_method"] == "email"
        assert "Python" in result["skills"]
        mock_complete.assert_called_once()

    @patch("agents.jd_parser.extract_skills", return_value=[])
    @patch("agents.jd_parser.complete", return_value="not valid json at all")
    def test_invalid_json_returns_defaults(self, mock_complete, mock_ner):
        result = parse_jd("Some JD text")
        assert result["apply_method"] == "unknown"
        assert result["company"] is None

    @patch("agents.jd_parser.extract_skills", return_value=[])
    @patch("agents.jd_parser.complete", side_effect=RuntimeError("LLM down"))
    def test_llm_failure_raises(self, mock_complete, mock_ner):
        with pytest.raises(RuntimeError, match="LLM down"):
            parse_jd("Some JD text")

    @patch("agents.jd_parser.extract_skills", return_value=[])
    @patch("agents.jd_parser.complete")
    def test_result_has_all_required_keys(self, mock_complete, mock_ner, sample_jd_text):
        mock_complete.return_value = json.dumps({"company": "Test"})
        result = parse_jd(sample_jd_text)
        for key in _DEFAULT_REQUIREMENTS:
            assert key in result

    @patch("agents.jd_parser.extract_skills", return_value=[])
    @patch("agents.jd_parser.complete")
    def test_jd_text_stored_in_result(self, mock_complete, mock_ner, sample_jd_text):
        mock_complete.return_value = json.dumps({"company": "Test"})
        result = parse_jd(sample_jd_text)
        assert result["_jd_text"] == sample_jd_text
