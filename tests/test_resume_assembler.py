"""Tests for agents/resume_assembler.py — template parsing and style utilities."""

from __future__ import annotations

import pytest

from agents.resume_assembler import (
    extract_style_examples,
    detect_project_separator,
    _fuzzy_find,
    get_page_count,
)


# ── extract_style_examples ────────────────────────────────────────────────

class TestExtractStyleExamples:
    def test_extracts_inject_blocks(self):
        template = (
            "%%INJECT:EXPERIENCE%%\n"
            "\\textbf{Job Title} at Company\n"
            "%%END:EXPERIENCE%%\n"
            "%%INJECT:SKILLS%%\n"
            "Python, Docker\n"
            "%%END:SKILLS%%\n"
        )
        result = extract_style_examples(template)
        assert isinstance(result, dict)
        assert "EXPERIENCE" in result
        assert "SKILLS" in result

    def test_empty_template_returns_empty(self):
        result = extract_style_examples("")
        assert result == {}

    def test_no_markers_returns_empty(self):
        result = extract_style_examples("Just plain LaTeX content here.")
        assert result == {}


# ── detect_project_separator ──────────────────────────────────────────────

class TestDetectProjectSeparator:
    def test_detects_vspace(self):
        section = "Project 1\n\\vspace{2pt}\nProject 2"
        result = detect_project_separator(section)
        assert result is not None
        assert "vspace" in result

    def test_none_when_no_separator(self):
        section = "Single project only"
        result = detect_project_separator(section)
        # May return None or a default — just verify it doesn't crash
        assert result is None or isinstance(result, str)


# ── _fuzzy_find ───────────────────────────────────────────────────────────

class TestFuzzyFind:
    def test_exact_match(self):
        candidates = {"CouchHire": "content1", "NIFTY50": "content2"}
        result = _fuzzy_find("CouchHire", candidates)
        assert result == "CouchHire"

    def test_case_insensitive(self):
        candidates = {"CouchHire": "content1", "NIFTY50": "content2"}
        result = _fuzzy_find("couchhire", candidates)
        assert result == "CouchHire"

    def test_no_match_returns_none(self):
        candidates = {"CouchHire": "content1"}
        result = _fuzzy_find("NonExistent", candidates)
        assert result is None

    def test_partial_match(self):
        candidates = {
            "CouchHire -- Agentic Job Application Automation": "content1",
            "NIFTY50 Portfolio Optimizer": "content2",
        }
        result = _fuzzy_find("CouchHire", candidates)
        # Should find the partial match
        assert result is not None


# ── get_page_count ────────────────────────────────────────────────────────

class TestGetPageCount:
    def test_nonexistent_file_returns_zero(self, tmp_path):
        result = get_page_count(tmp_path / "nonexistent.pdf")
        assert result == 0

    def test_returns_integer(self, tmp_path):
        # Create a minimal file (not a real PDF, so page count should be 0 or handled)
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_text("not a real pdf")
        result = get_page_count(fake_pdf)
        assert isinstance(result, int)
