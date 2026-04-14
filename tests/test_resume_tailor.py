"""Tests for agents/resume_tailor.py — resume tailoring with mocked dependencies."""

from __future__ import annotations

from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest


# ── Import helpers (these don't require LLM/ChromaDB) ─────────────────────

class TestResumeTailorImports:
    """Verify the module is importable and key functions exist."""

    def test_module_importable(self):
        import agents.resume_tailor as rt
        assert hasattr(rt, "tailor")

    def test_generate_resume_content_exists(self):
        from agents.resume_tailor import _generate_resume_content_summary
        assert callable(_generate_resume_content_summary)


class TestGenerateResumeContent:
    """Test the deterministic resume_content generation."""

    def test_returns_string(self):
        from agents.resume_tailor import _generate_resume_content_summary
        result = _generate_resume_content_summary(
            selection={"sections_to_include": {}, "project_order": []},
            requirements={"role": "ML Engineer", "company": "Acme"},
            instructions="",
            raw_sections={"Skills": "Python, Docker, Kubernetes"},
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_no_latex_commands_in_output(self):
        from agents.resume_tailor import _generate_resume_content_summary
        result = _generate_resume_content_summary(
            selection={"sections_to_include": {"Experience": True}, "project_order": []},
            requirements={"role": "ML Engineer", "company": "Acme"},
            instructions="",
            raw_sections={"Experience": r"\textbf{Software Engineer} \item Built ML pipeline"},
        )
        # Should not contain raw LaTeX commands
        assert "\\textbf" not in result
        assert "\\begin{" not in result
        assert "\\item" not in result
