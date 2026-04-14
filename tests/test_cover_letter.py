"""Tests for agents/cover_letter.py — cover letter generation with mocked LLM."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agents.cover_letter import _detect_tone, _tone_instruction, generate


# ── _detect_tone ──────────────────────────────────────────────────────────

class TestDetectTone:
    @pytest.mark.parametrize("role,expected", [
        ("Quantitative Analyst", "quant"),
        ("Trading Engineer", "quant"),
        ("ML Engineer", "ml"),
        ("Machine Learning Researcher", "ml"),
        ("AI Research Scientist", "ml"),
        ("LLM Engineer", "ml"),
        ("Backend Developer", "professional"),
        ("Product Manager", "professional"),
    ])
    def test_tone_detection(self, role, expected):
        assert _detect_tone(role) == expected


class TestToneInstruction:
    @pytest.mark.parametrize("tone", ["quant", "ml", "professional"])
    def test_returns_non_empty_string(self, tone):
        result = _tone_instruction(tone)
        assert isinstance(result, str)
        assert len(result) > 20


# ── generate ──────────────────────────────────────────────────────────────

class TestGenerate:
    def test_skips_when_not_required(self, sample_requirements, sample_cv_sections, sample_resume_content):
        sample_requirements["cover_letter_required"] = False
        result = generate(sample_requirements, sample_cv_sections, sample_resume_content)
        assert result == ""

    @patch("llm.client.complete")
    def test_generates_when_required(self, mock_complete, sample_requirements, sample_cv_sections, sample_resume_content):
        sample_requirements["cover_letter_required"] = True
        mock_complete.return_value = (
            "Paragraph 1 about the company.\n\n"
            "Paragraph 2 about fit.\n\n"
            "Paragraph 3 closing."
        )
        result = generate(sample_requirements, sample_cv_sections, sample_resume_content)
        assert isinstance(result, str)
        assert len(result) > 0
        mock_complete.assert_called_once()

    @patch("llm.client.complete")
    def test_strips_code_fences(self, mock_complete, sample_requirements, sample_cv_sections, sample_resume_content):
        sample_requirements["cover_letter_required"] = True
        mock_complete.return_value = "```\nParagraph text here.\n```"
        result = generate(sample_requirements, sample_cv_sections, sample_resume_content)
        assert "```" not in result

    @patch("llm.client.complete", side_effect=RuntimeError("LLM down"))
    def test_llm_failure_raises(self, mock_complete, sample_requirements, sample_cv_sections, sample_resume_content):
        sample_requirements["cover_letter_required"] = True
        with pytest.raises(RuntimeError, match="LLM down"):
            generate(sample_requirements, sample_cv_sections, sample_resume_content)
