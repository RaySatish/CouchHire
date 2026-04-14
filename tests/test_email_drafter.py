"""Tests for agents/email_drafter.py — email subject and body generation."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agents.email_drafter import _build_subject, draft


# ── _build_subject ────────────────────────────────────────────────────────

class TestBuildSubject:
    def test_uses_jd_format_when_present(self):
        reqs = {
            "subject_line_format": "ML Engineer Application - [Your Name]",
            "role": "ML Engineer",
        }
        result = _build_subject(reqs)
        assert "Test User" in result
        assert "[Your Name]" not in result

    def test_default_format_when_no_jd_format(self):
        reqs = {"subject_line_format": None, "role": "Backend Developer"}
        result = _build_subject(reqs)
        assert "Backend Developer" in result
        assert "Test User" in result
        assert "Application for" in result

    def test_default_role_when_missing(self):
        reqs = {"subject_line_format": None}
        result = _build_subject(reqs)
        assert "the role" in result


# ── draft ─────────────────────────────────────────────────────────────────

class TestDraft:
    @patch("llm.client.complete")
    def test_returns_subject_and_body_tuple(self, mock_complete, sample_requirements, sample_resume_content):
        mock_complete.return_value = "Hi,\n\nI'm applying for the ML Engineer role.\n\nTest User | test@example.com | +1-555-0100"
        subject, body = draft(
            sample_requirements,
            cover_letter_text="",
            resume_pdf_path="/tmp/resume.pdf",
            resume_content=sample_resume_content,
        )
        assert isinstance(subject, str)
        assert isinstance(body, str)
        assert len(subject) > 0
        assert len(body) > 0

    @patch("llm.client.complete")
    def test_subject_uses_jd_format(self, mock_complete, sample_requirements, sample_resume_content):
        mock_complete.return_value = "Hi,\n\nEmail body."
        subject, _ = draft(
            sample_requirements,
            cover_letter_text="",
            resume_pdf_path="/tmp/resume.pdf",
            resume_content=sample_resume_content,
        )
        # sample_requirements has subject_line_format set
        assert "Test User" in subject

    @patch("llm.client.complete")
    def test_signature_appended_if_missing(self, mock_complete, sample_requirements, sample_resume_content):
        mock_complete.return_value = "Hi,\n\nBody without signature."
        _, body = draft(
            sample_requirements,
            cover_letter_text="",
            resume_pdf_path="/tmp/resume.pdf",
            resume_content=sample_resume_content,
        )
        assert "Test User" in body

    @patch("llm.client.complete")
    def test_asterisks_stripped(self, mock_complete, sample_requirements, sample_resume_content):
        mock_complete.return_value = "Hi,\n\n**Bold text** and *italic*."
        _, body = draft(
            sample_requirements,
            cover_letter_text="",
            resume_pdf_path="/tmp/resume.pdf",
            resume_content=sample_resume_content,
        )
        assert "**" not in body
        assert "*" not in body

    @patch("llm.client.complete", side_effect=RuntimeError("LLM down"))
    def test_llm_failure_raises(self, mock_complete, sample_requirements, sample_resume_content):
        with pytest.raises(RuntimeError, match="LLM down"):
            draft(
                sample_requirements,
                cover_letter_text="",
                resume_pdf_path="/tmp/resume.pdf",
                resume_content=sample_resume_content,
            )
