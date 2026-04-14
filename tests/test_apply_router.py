"""Tests for agents/apply_router.py — deterministic routing logic."""

from __future__ import annotations

import pytest

from agents.apply_router import route, _is_valid_email, _is_valid_url


# ── Validation helpers ────────────────────────────────────────────────────

class TestIsValidEmail:
    @pytest.mark.parametrize("email", [
        "user@example.com",
        "hr@quantumleap.ai",
        "jobs+apply@company.co.uk",
    ])
    def test_valid_emails(self, email):
        assert _is_valid_email(email) is True

    @pytest.mark.parametrize("email", [
        "",
        "not-an-email",
        "@missing.user",
        "user@",
        "user @example.com",
    ])
    def test_invalid_emails(self, email):
        assert _is_valid_email(email) is False


class TestIsValidUrl:
    @pytest.mark.parametrize("url", [
        "https://jobs.lever.co/widget/12345",
        "http://example.com/apply",
        "https://boards.greenhouse.io/company/jobs/123",
    ])
    def test_valid_urls(self, url):
        assert _is_valid_url(url) is True

    @pytest.mark.parametrize("url", [
        "",
        "ftp://example.com",
        "jobs.lever.co/widget",
        "not a url",
    ])
    def test_invalid_urls(self, url):
        assert _is_valid_url(url) is False


# ── route() ───────────────────────────────────────────────────────────────

class TestRoute:
    def test_email_route(self, sample_email_requirements):
        assert route(sample_email_requirements) == "email"

    def test_form_route(self, sample_url_requirements):
        assert route(sample_url_requirements) == "form"

    def test_manual_route(self, sample_manual_requirements):
        assert route(sample_manual_requirements) == "manual"

    def test_email_method_but_invalid_target(self):
        reqs = {"apply_method": "email", "apply_target": "not-an-email"}
        assert route(reqs) == "manual"

    def test_url_method_but_invalid_target(self):
        reqs = {"apply_method": "url", "apply_target": "not-a-url"}
        assert route(reqs) == "manual"

    def test_email_method_no_target(self):
        reqs = {"apply_method": "email", "apply_target": None}
        assert route(reqs) == "manual"

    def test_unknown_method(self):
        reqs = {"apply_method": "unknown", "apply_target": "user@example.com"}
        assert route(reqs) == "manual"

    def test_empty_requirements(self):
        assert route({}) == "manual"

    def test_return_type_is_string(self, sample_requirements):
        result = route(sample_requirements)
        assert isinstance(result, str)
        assert result in {"email", "form", "manual"}
