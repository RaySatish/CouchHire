"""Tests for agents/llm_selector.py — validation and cleaning logic.

Does NOT test the actual LLM call (that requires API keys).
Tests the guardrails that prevent hallucinated content.
"""

import json
import pytest

from agents.llm_selector import (
    _clean_json_response,
    _fuzzy_match,
    _validate_selection,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def sample_inventory():
    """A minimal inventory matching the real master_cv structure."""
    return {
        "PROJECTS": {
            "names": [
                "CouchHire -- Agentic Job Application Automation",
                "Market Surveillance System",
                "NIFTY50 Portfolio Optimizer",
                "Loan Risk Analysis",
                "Maritime Situational Awareness System",
                "AI-Powered Soil Classification System",
                "Conversational AI Assistant",
            ],
            "items": ["..."] * 7,
            "count": 7,
        },
        "EXPERIENCE": {
            "names": [
                "Research Intern at Indian Institute of Information Technology, Allahabad"
            ],
            "items": ["..."],
            "count": 1,
        },
        "CERTIFICATIONS": {
            "names": [
                "AWS Certified Cloud Practitioner",
                "Oracle Cloud Infrastructure 2024 Generalist",
                "Machine Learning Specialization",
                "Generative AI with Large Language Models",
            ],
            "items": ["..."] * 4,
            "count": 4,
        },
        "SKILLS": {
            "names": ["Languages", "ML/AI", "Data", "Cloud", "Dev Tools"],
            "items": ["..."] * 5,
            "count": 5,
        },
        "EDUCATION": {
            "names": ["IIIT Allahabad"],
            "items": ["..."],
            "count": 1,
        },
        "LEADERSHIP": {
            "names": [
                "Programme Representative, Office of Student Welfare",
                "Core Member, Finance and Economics Club",
            ],
            "items": ["..."] * 2,
            "count": 2,
        },
    }


# ── Fuzzy matching ────────────────────────────────────────────────────

class TestFuzzyMatch:
    def test_exact_match(self):
        names = {"Market Surveillance System", "NIFTY50 Portfolio Optimizer"}
        assert _fuzzy_match("Market Surveillance System", names) == "Market Surveillance System"

    def test_case_insensitive(self):
        names = {"NIFTY50 Portfolio Optimizer"}
        assert _fuzzy_match("nifty50 portfolio optimizer", names) == "NIFTY50 Portfolio Optimizer"

    def test_substring_short_to_long(self):
        names = {"CouchHire -- Agentic Job Application Automation"}
        assert _fuzzy_match("CouchHire", names) == "CouchHire -- Agentic Job Application Automation"

    def test_substring_long_to_short(self):
        names = {"CouchHire"}
        assert _fuzzy_match("CouchHire -- Agentic Job Application Automation", names) == "CouchHire"

    def test_no_match_returns_none(self):
        names = {"Market Surveillance System", "NIFTY50 Portfolio Optimizer"}
        assert _fuzzy_match("Totally Fake Project", names) is None

    def test_empty_valid_names(self):
        assert _fuzzy_match("anything", set()) is None


# ── JSON cleaning ─────────────────────────────────────────────────────

class TestCleanJsonResponse:
    def test_clean_json(self):
        raw = '{"key": "value"}'
        assert json.loads(_clean_json_response(raw)) == {"key": "value"}

    def test_json_fenced(self):
        raw = '```json\n{"key": "value"}\n```'
        assert json.loads(_clean_json_response(raw)) == {"key": "value"}

    def test_plain_fenced(self):
        raw = '```\n{"key": "value"}\n```'
        assert json.loads(_clean_json_response(raw)) == {"key": "value"}

    def test_whitespace(self):
        raw = '  \n{"key": "value"}\n  '
        assert json.loads(_clean_json_response(raw)) == {"key": "value"}


# ── Validation ────────────────────────────────────────────────────────

class TestValidateSelection:
    def test_forces_required_sections(self, sample_inventory):
        """HEADER, EDUCATION, SKILLS must always be true."""
        selection = {
            "sections_to_include": {
                "PROJECTS": ["Market Surveillance System"],
            },
        }
        result = _validate_selection(selection, sample_inventory)
        assert result["sections_to_include"]["HEADER"] is True
        assert result["sections_to_include"]["EDUCATION"] is True
        assert result["sections_to_include"]["SKILLS"] is True

    def test_removes_hallucinated_projects(self, sample_inventory):
        """Projects not in inventory must be stripped out."""
        selection = {
            "sections_to_include": {
                "HEADER": True,
                "EDUCATION": True,
                "SKILLS": True,
                "PROJECTS": [
                    "Market Surveillance System",
                    "Fake AI Project That Doesnt Exist",
                    "NIFTY50 Portfolio Optimizer",
                ],
            },
            "project_order": [
                "Market Surveillance System",
                "Fake AI Project That Doesnt Exist",
                "NIFTY50 Portfolio Optimizer",
            ],
        }
        result = _validate_selection(selection, sample_inventory)
        projects = result["sections_to_include"]["PROJECTS"]
        assert "Market Surveillance System" in projects
        assert "NIFTY50 Portfolio Optimizer" in projects
        assert "Fake AI Project That Doesnt Exist" not in projects
        assert len(projects) == 2
        # project_order should also be cleaned
        assert "Fake AI Project That Doesnt Exist" not in result["project_order"]

    def test_fuzzy_corrects_short_names(self, sample_inventory):
        """LLM using 'CouchHire' should be corrected to full name."""
        selection = {
            "sections_to_include": {
                "HEADER": True,
                "EDUCATION": True,
                "SKILLS": True,
                "PROJECTS": ["CouchHire", "NIFTY50 Portfolio Optimizer"],
            },
        }
        result = _validate_selection(selection, sample_inventory)
        projects = result["sections_to_include"]["PROJECTS"]
        assert "CouchHire -- Agentic Job Application Automation" in projects

    def test_true_means_include_all(self, sample_inventory):
        """Setting a section to True should pass through unchanged."""
        selection = {
            "sections_to_include": {
                "HEADER": True,
                "EDUCATION": True,
                "SKILLS": True,
                "PROJECTS": True,
                "EXPERIENCE": True,
            },
        }
        result = _validate_selection(selection, sample_inventory)
        assert result["sections_to_include"]["PROJECTS"] is True
        assert result["sections_to_include"]["EXPERIENCE"] is True

    def test_false_means_exclude(self, sample_inventory):
        """Setting a section to False should pass through unchanged."""
        selection = {
            "sections_to_include": {
                "HEADER": True,
                "EDUCATION": True,
                "SKILLS": True,
                "LEADERSHIP": False,
            },
        }
        result = _validate_selection(selection, sample_inventory)
        assert result["sections_to_include"]["LEADERSHIP"] is False

    def test_auto_populates_project_order(self, sample_inventory):
        """If project_order is missing, it should be built from sections_to_include."""
        selection = {
            "sections_to_include": {
                "HEADER": True,
                "EDUCATION": True,
                "SKILLS": True,
                "PROJECTS": ["Market Surveillance System", "NIFTY50 Portfolio Optimizer"],
            },
        }
        result = _validate_selection(selection, sample_inventory)
        assert result["project_order"] == [
            "Market Surveillance System",
            "NIFTY50 Portfolio Optimizer",
        ]

    def test_project_order_adds_missing_items(self, sample_inventory):
        """If project_order is incomplete, missing selected projects are appended."""
        selection = {
            "sections_to_include": {
                "HEADER": True,
                "EDUCATION": True,
                "SKILLS": True,
                "PROJECTS": [
                    "Market Surveillance System",
                    "NIFTY50 Portfolio Optimizer",
                    "Loan Risk Analysis",
                ],
            },
            "project_order": ["NIFTY50 Portfolio Optimizer"],
        }
        result = _validate_selection(selection, sample_inventory)
        order = result["project_order"]
        assert order[0] == "NIFTY50 Portfolio Optimizer"
        assert "Market Surveillance System" in order
        assert "Loan Risk Analysis" in order
        assert len(order) == 3

    def test_removes_hallucinated_certifications(self, sample_inventory):
        """Certifications not in inventory must be stripped out."""
        selection = {
            "sections_to_include": {
                "HEADER": True,
                "EDUCATION": True,
                "SKILLS": True,
                "CERTIFICATIONS": [
                    "AWS Certified Cloud Practitioner",
                    "Google Cloud Professional Architect",  # hallucinated
                ],
            },
        }
        result = _validate_selection(selection, sample_inventory)
        certs = result["sections_to_include"]["CERTIFICATIONS"]
        assert "AWS Certified Cloud Practitioner" in certs
        assert "Google Cloud Professional Architect" not in certs
        assert len(certs) == 1
