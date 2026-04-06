"""Tests for agents/llm_selector.py — validation, cleaning, and enforcement logic.

Does NOT test the actual LLM call (that requires API keys).
Tests the guardrails that prevent hallucinated content and enforce
skill category selection, ordering, and instruction-based rules.
"""

import json
import pytest

from agents.llm_selector import (
    _clean_json_response,
    _enforce_instructions,
    _fuzzy_match,
    _validate_selection,
)


# ── Fixtures ──────────────────────────────────────────────────────

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
            "names": [
                "Programming",
                "ML/AI",
                "Data",
                "Cloud",
                "Dev Tools",
                "Soft Skills",
                "Quant/Finance",
                "Embedded/IoT",
            ],
            "items": ["..."] * 8,
            "count": 8,
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


# ── Fuzzy matching ────────────────────────────────────────────────

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


# ── JSON cleaning ─────────────────────────────────────────────────

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


# ── Validation ────────────────────────────────────────────────────

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


# ── Skill category validation ────────────────────────────────────

class TestSkillCategoryValidation:
    """Tests for skill_categories_to_include validation within _validate_selection."""

    def test_valid_categories_preserved(self, sample_inventory):
        """Valid skill categories pass through unchanged."""
        selection = {
            "sections_to_include": {
                "HEADER": True,
                "EDUCATION": True,
                "SKILLS": True,
            },
            "skill_categories_to_include": [
                "ML/AI", "Cloud", "Programming", "Soft Skills",
            ],
        }
        result = _validate_selection(selection, sample_inventory)
        cats = result["skill_categories_to_include"]
        assert cats == ["ML/AI", "Cloud", "Programming", "Soft Skills"]

    def test_hallucinated_categories_removed(self, sample_inventory):
        """Skill categories not in inventory are stripped out."""
        selection = {
            "sections_to_include": {
                "HEADER": True,
                "EDUCATION": True,
                "SKILLS": True,
            },
            "skill_categories_to_include": [
                "Programming",
                "Blockchain",  # hallucinated
                "ML/AI",
                "Quantum Computing",  # hallucinated
                "Soft Skills",
            ],
        }
        result = _validate_selection(selection, sample_inventory)
        cats = result["skill_categories_to_include"]
        assert "Blockchain" not in cats
        assert "Quantum Computing" not in cats
        assert "Programming" in cats
        assert "ML/AI" in cats
        assert "Soft Skills" in cats

    def test_fuzzy_corrects_category_names(self, sample_inventory):
        """LLM using approximate names should be fuzzy-corrected."""
        selection = {
            "sections_to_include": {
                "HEADER": True,
                "EDUCATION": True,
                "SKILLS": True,
            },
            "skill_categories_to_include": [
                "programming",  # lowercase
                "ml/ai",  # lowercase
                "dev tools",  # lowercase
                "Soft Skills",
            ],
        }
        result = _validate_selection(selection, sample_inventory)
        cats = result["skill_categories_to_include"]
        assert "Programming" in cats
        assert "ML/AI" in cats
        assert "Dev Tools" in cats

    def test_missing_categories_defaults_to_all(self, sample_inventory):
        """If skill_categories_to_include is missing, default to all inventory categories."""
        selection = {
            "sections_to_include": {
                "HEADER": True,
                "EDUCATION": True,
                "SKILLS": True,
            },
            # No skill_categories_to_include at all
        }
        result = _validate_selection(selection, sample_inventory)
        cats = result["skill_categories_to_include"]
        # Should have all 8 categories from inventory
        assert len(cats) == len(sample_inventory["SKILLS"]["names"])
        for name in sample_inventory["SKILLS"]["names"]:
            assert name in cats

    def test_empty_list_defaults_to_all(self, sample_inventory):
        """An empty skill_categories_to_include should default to all."""
        selection = {
            "sections_to_include": {
                "HEADER": True,
                "EDUCATION": True,
                "SKILLS": True,
            },
            "skill_categories_to_include": [],
        }
        result = _validate_selection(selection, sample_inventory)
        cats = result["skill_categories_to_include"]
        assert len(cats) == len(sample_inventory["SKILLS"]["names"])

    def test_required_categories_enforced(self, sample_inventory):
        """Programming and Soft Skills must always be present if in inventory."""
        selection = {
            "sections_to_include": {
                "HEADER": True,
                "EDUCATION": True,
                "SKILLS": True,
            },
            "skill_categories_to_include": ["ML/AI", "Cloud"],
            # Missing Programming and Soft Skills
        }
        result = _validate_selection(selection, sample_inventory)
        cats = result["skill_categories_to_include"]
        assert "Programming" in cats
        assert "Soft Skills" in cats
        # Original selections preserved
        assert "ML/AI" in cats
        assert "Cloud" in cats

    def test_required_categories_not_duplicated(self, sample_inventory):
        """If Programming is already present, it should not be added again."""
        selection = {
            "sections_to_include": {
                "HEADER": True,
                "EDUCATION": True,
                "SKILLS": True,
            },
            "skill_categories_to_include": [
                "Programming", "ML/AI", "Soft Skills",
            ],
        }
        result = _validate_selection(selection, sample_inventory)
        cats = result["skill_categories_to_include"]
        assert cats.count("Programming") == 1
        assert cats.count("Soft Skills") == 1

    def test_ordering_preserved(self, sample_inventory):
        """The LLM's chosen order should be preserved after validation."""
        selection = {
            "sections_to_include": {
                "HEADER": True,
                "EDUCATION": True,
                "SKILLS": True,
            },
            "skill_categories_to_include": [
                "Cloud", "ML/AI", "Data", "Programming", "Soft Skills",
            ],
        }
        result = _validate_selection(selection, sample_inventory)
        cats = result["skill_categories_to_include"]
        # Order should be exactly as LLM specified (all are valid)
        assert cats[:5] == ["Cloud", "ML/AI", "Data", "Programming", "Soft Skills"]

    def test_deduplication_preserves_first(self, sample_inventory):
        """Duplicate categories should be deduplicated, keeping first occurrence."""
        selection = {
            "sections_to_include": {
                "HEADER": True,
                "EDUCATION": True,
                "SKILLS": True,
            },
            "skill_categories_to_include": [
                "ML/AI", "Cloud", "ML/AI", "Programming", "Cloud", "Soft Skills",
            ],
        }
        result = _validate_selection(selection, sample_inventory)
        cats = result["skill_categories_to_include"]
        assert cats.count("ML/AI") == 1
        assert cats.count("Cloud") == 1
        # First-occurrence order preserved
        assert cats.index("ML/AI") < cats.index("Cloud")
        assert cats.index("Cloud") < cats.index("Programming")

    def test_categories_inside_sections_to_include(self, sample_inventory):
        """skill_categories_to_include inside sections_to_include should also work."""
        selection = {
            "sections_to_include": {
                "HEADER": True,
                "EDUCATION": True,
                "SKILLS": True,
                "skill_categories_to_include": ["Data", "Dev Tools", "Programming", "Soft Skills"],
            },
        }
        result = _validate_selection(selection, sample_inventory)
        cats = result["skill_categories_to_include"]
        assert "Data" in cats
        assert "Dev Tools" in cats

    def test_categories_synced_to_sections(self, sample_inventory):
        """After validation, skill_categories should be in both top-level and sections."""
        selection = {
            "sections_to_include": {
                "HEADER": True,
                "EDUCATION": True,
                "SKILLS": True,
            },
            "skill_categories_to_include": ["ML/AI", "Programming", "Soft Skills"],
        }
        result = _validate_selection(selection, sample_inventory)
        top_level = result["skill_categories_to_include"]
        in_sections = result["sections_to_include"]["skill_categories_to_include"]
        assert top_level == in_sections


# ── Instruction enforcement ───────────────────────────────────────

class TestEnforceInstructions:
    """Tests for _enforce_instructions — skill category and item-level rules."""

    def test_conditional_exclude_when_condition_met(self, sample_inventory):
        """'do not include Quant/Finance in skills if not a quant role' should remove it."""
        instructions = (
            "do not include Quant/Finance section in skills if not a quant-based role\n"
        )
        selection = {
            "sections_to_include": {
                "HEADER": True,
                "EDUCATION": True,
                "SKILLS": True,
                "PROJECTS": ["CouchHire -- Agentic Job Application Automation"],
            },
            "skill_categories_to_include": [
                "Programming", "ML/AI", "Quant/Finance", "Soft Skills",
            ],
            "project_order": ["CouchHire -- Agentic Job Application Automation"],
        }
        requirements = {"role": "Software Engineer", "company": "Google"}
        result = _enforce_instructions(
            selection, instructions, sample_inventory, requirements
        )
        cats = result["skill_categories_to_include"]
        # Role is "Software Engineer" (not quant), so Quant/Finance should be removed
        assert "Quant/Finance" not in cats
        # Other categories untouched
        assert "Programming" in cats
        assert "ML/AI" in cats

    def test_conditional_exclude_when_condition_not_met(self, sample_inventory):
        """'do not include Quant/Finance if not quant role' should keep it for quant roles."""
        instructions = (
            "do not include Quant/Finance section in skills if not a quant-based role\n"
        )
        selection = {
            "sections_to_include": {
                "HEADER": True,
                "EDUCATION": True,
                "SKILLS": True,
            },
            "skill_categories_to_include": [
                "Programming", "ML/AI", "Quant/Finance", "Soft Skills",
            ],
        }
        requirements = {"role": "Quantitative Developer", "company": "Jane Street"}
        result = _enforce_instructions(
            selection, instructions, sample_inventory, requirements
        )
        cats = result["skill_categories_to_include"]
        # Role contains "quant", so Quant/Finance should be kept
        assert "Quant/Finance" in cats

    def test_for_role_use_category(self, sample_inventory):
        """'for quant roles use Quant/Finance' should add the category."""
        instructions = "for quant/trading roles use Quant/Finance\n"
        selection = {
            "sections_to_include": {
                "HEADER": True,
                "EDUCATION": True,
                "SKILLS": True,
            },
            "skill_categories_to_include": [
                "Programming", "ML/AI", "Soft Skills",
            ],
        }
        requirements = {"role": "Quantitative Analyst", "company": "Citadel"}
        result = _enforce_instructions(
            selection, instructions, sample_inventory, requirements
        )
        cats = result["skill_categories_to_include"]
        # Role contains "quant", so Quant/Finance should be added
        assert "Quant/Finance" in cats

    def test_for_role_use_category_no_match(self, sample_inventory):
        """'for quant roles use Quant/Finance' should not add it for non-quant roles."""
        instructions = "for quant/trading roles use Quant/Finance\n"
        selection = {
            "sections_to_include": {
                "HEADER": True,
                "EDUCATION": True,
                "SKILLS": True,
            },
            "skill_categories_to_include": [
                "Programming", "ML/AI", "Soft Skills",
            ],
        }
        requirements = {"role": "Frontend Developer", "company": "Spotify"}
        result = _enforce_instructions(
            selection, instructions, sample_inventory, requirements
        )
        cats = result["skill_categories_to_include"]
        # Role is Frontend Developer — no quant/trading match
        assert "Quant/Finance" not in cats

    def test_for_role_use_category_already_present(self, sample_inventory):
        """'for X roles use Y' should not duplicate if Y is already present."""
        instructions = "for ai/ml roles use ML/AI\n"
        selection = {
            "sections_to_include": {
                "HEADER": True,
                "EDUCATION": True,
                "SKILLS": True,
            },
            "skill_categories_to_include": [
                "Programming", "ML/AI", "Soft Skills",
            ],
        }
        requirements = {"role": "ML Engineer", "company": "OpenAI"}
        result = _enforce_instructions(
            selection, instructions, sample_inventory, requirements
        )
        cats = result["skill_categories_to_include"]
        assert cats.count("ML/AI") == 1

    def test_exclude_item_from_section(self, sample_inventory):
        """'do not include X in Projects' should remove matching items."""
        instructions = "do not include Loan Risk Analysis in projects\n"
        selection = {
            "sections_to_include": {
                "HEADER": True,
                "EDUCATION": True,
                "SKILLS": True,
                "PROJECTS": [
                    "Market Surveillance System",
                    "Loan Risk Analysis",
                    "NIFTY50 Portfolio Optimizer",
                ],
            },
            "skill_categories_to_include": ["Programming", "Soft Skills"],
            "project_order": [
                "Market Surveillance System",
                "Loan Risk Analysis",
                "NIFTY50 Portfolio Optimizer",
            ],
        }
        requirements = {"role": "Software Engineer", "company": "Google"}
        result = _enforce_instructions(
            selection, instructions, sample_inventory, requirements
        )
        projects = result["sections_to_include"]["PROJECTS"]
        assert "Loan Risk Analysis" not in projects
        assert "Market Surveillance System" in projects
        assert "NIFTY50 Portfolio Optimizer" in projects

    def test_no_instructions_is_noop(self, sample_inventory):
        """Empty instructions should not modify the selection."""
        selection = {
            "sections_to_include": {
                "HEADER": True,
                "EDUCATION": True,
                "SKILLS": True,
                "PROJECTS": ["Market Surveillance System"],
            },
            "skill_categories_to_include": [
                "Programming", "ML/AI", "Soft Skills",
            ],
            "project_order": ["Market Surveillance System"],
        }
        original_cats = list(selection["skill_categories_to_include"])
        result = _enforce_instructions(
            selection, "", sample_inventory, {"role": "Engineer", "company": "X"}
        )
        assert result["skill_categories_to_include"] == original_cats

    def test_multiple_rules_applied(self, sample_inventory):
        """Multiple instruction rules should all be applied."""
        instructions = (
            "do not include Embedded/IoT section in skills if not an embedded role\n"
            "for quant/trading roles use Quant/Finance\n"
        )
        selection = {
            "sections_to_include": {
                "HEADER": True,
                "EDUCATION": True,
                "SKILLS": True,
            },
            "skill_categories_to_include": [
                "Programming", "ML/AI", "Embedded/IoT", "Soft Skills",
            ],
        }
        requirements = {"role": "Quantitative Developer", "company": "Two Sigma"}
        result = _enforce_instructions(
            selection, instructions, sample_inventory, requirements
        )
        cats = result["skill_categories_to_include"]
        # Not an embedded role → Embedded/IoT removed
        assert "Embedded/IoT" not in cats
        # Is a quant role → Quant/Finance added
        assert "Quant/Finance" in cats
        # Others untouched
        assert "Programming" in cats
        assert "ML/AI" in cats
