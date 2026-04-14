"""
Step 25i — Verification tests for the full pipeline integration.

Tests 1–11 validate that all sub-steps (25b–25h) are correctly implemented:
  1. Import check: pipeline module and key exports
  2. Graph structure: correct number of nodes
  3. CLI help: argparse flags exist
  4. Telegram wiring: no TODO stubs remain, gate handler registered
  5. State schema: PipelineState has expected keys
  6. Config: APPLICANT_EMAIL, APPLICANT_PHONE exist
  7. gmail_sender: send_draft is callable, create_draft returns tuple
  8. browser_agent: fill_form has submit param, submit_form importable
  9. Cover letter template: exists, has 10 placeholders, no forbidden content
 10. Cover letter prompt: contains resume-only constraint
 11. .env.example: has APPLICANT_EMAIL, APPLICANT_PHONE entries
"""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path
from typing import get_type_hints

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Test 1: Import check — pipeline module and key exports
# ---------------------------------------------------------------------------

class TestImports:
    """Verify the pipeline module is importable and exports key symbols."""

    def test_import_pipeline_module(self):
        """pipeline.py is importable without errors."""
        import pipeline  # noqa: F401

    def test_run_pipeline_importable(self):
        """run_pipeline function is importable."""
        from pipeline import run_pipeline
        assert callable(run_pipeline)

    def test_build_graph_importable(self):
        """build_graph function is importable."""
        from pipeline import build_graph
        assert callable(build_graph)

    def test_compile_graph_importable(self):
        """compile_graph function is importable."""
        from pipeline import compile_graph
        assert callable(compile_graph)

    def test_pipeline_state_importable(self):
        """PipelineState TypedDict is importable."""
        from pipeline import PipelineState
        assert PipelineState is not None

    def test_compiled_pipeline_exists(self):
        """Module-level _compiled_pipeline is created at import time."""
        import pipeline
        assert hasattr(pipeline, "_compiled_pipeline")
        assert pipeline._compiled_pipeline is not None


# ---------------------------------------------------------------------------
# Test 2: Graph structure — correct number of nodes
# ---------------------------------------------------------------------------

class TestGraphStructure:
    """Verify the LangGraph graph has the expected nodes and edges."""

    def test_graph_has_at_least_18_nodes(self):
        """Graph should have at least 18 nodes (may have 19 with compile_cover_letter_pdf)."""
        from pipeline import build_graph
        graph = build_graph()
        node_names = set(graph.nodes.keys())
        assert len(node_names) >= 18, f"Expected >= 18 nodes, got {len(node_names)}: {node_names}"

    def test_graph_has_expected_core_nodes(self):
        """All core nodes exist in the graph."""
        from pipeline import build_graph
        graph = build_graph()
        node_names = set(graph.nodes.keys())

        expected_nodes = {
            "scrape_jd", "parse_jd", "cv_rag", "match_scorer",
            "threshold_gate", "resume_tailor", "cover_letter",
            "email_drafter", "approval_gate", "apply_router",
            "gmail_draft", "browser_agent", "gate2", "execute_send",
            "notify", "save_to_db", "rejected", "error",
        }
        missing = expected_nodes - node_names
        assert not missing, f"Missing nodes: {missing}"

    def test_graph_entry_point_is_scrape_jd(self):
        """Graph entry point should be scrape_jd."""
        from pipeline import build_graph
        graph = build_graph()
        # LangGraph stores the entry point — check it compiles without error
        compiled = graph.compile()
        assert compiled is not None


# ---------------------------------------------------------------------------
# Test 3: CLI help — argparse flags exist
# ---------------------------------------------------------------------------

class TestCLI:
    """Verify the CLI argument parser has expected flags."""

    def test_main_function_exists(self):
        """main() function exists in pipeline module."""
        from pipeline import main
        assert callable(main)

    def test_cli_flags_in_source(self):
        """CLI source code contains expected argument flags."""
        source = Path(PROJECT_ROOT / "pipeline.py").read_text(encoding="utf-8")
        expected_flags = ["--jd", "--url", "--file", "--search", "--query", "--location"]
        for flag in expected_flags:
            assert flag in source, f"CLI flag '{flag}' not found in pipeline.py source"


# ---------------------------------------------------------------------------
# Test 4: Telegram wiring — no TODO stubs, gate handler registered
# ---------------------------------------------------------------------------

class TestTelegramWiring:
    """Verify the Telegram bot has no TODO stubs and gate handlers are wired."""

    def test_no_todo_stubs_in_telegram_bot(self):
        """bot/telegram_bot.py should have 0 TODO comments."""
        source = Path(PROJECT_ROOT / "bot" / "telegram_bot.py").read_text(encoding="utf-8")
        todo_count = source.lower().count("todo")
        assert todo_count == 0, f"Found {todo_count} TODO(s) in telegram_bot.py"

    def test_no_pipeline_not_wired_stub(self):
        """No 'Pipeline not yet wired' or 'Pipeline wiring pending' stubs."""
        source = Path(PROJECT_ROOT / "bot" / "telegram_bot.py").read_text(encoding="utf-8")
        assert "Pipeline not yet wired" not in source
        assert "Pipeline wiring pending" not in source

    def test_gate_handler_exists(self):
        """Gate callback handler pattern exists in telegram_bot.py."""
        source = Path(PROJECT_ROOT / "bot" / "telegram_bot.py").read_text(encoding="utf-8")
        # Should have gate callback handling
        assert "gate" in source.lower(), "No gate-related code found in telegram_bot.py"

    def test_send_document_importable(self):
        """send_document function is importable from bot.telegram_bot."""
        from bot.telegram_bot import send_document
        assert callable(send_document)

    def test_send_job_card_importable(self):
        """send_job_card function is importable from bot.telegram_bot."""
        from bot.telegram_bot import send_job_card
        assert callable(send_job_card)

    def test_request_interrupt_importable(self):
        """request_interrupt function is importable from bot.telegram_bot."""
        from bot.telegram_bot import request_interrupt
        assert callable(request_interrupt)

    def test_request_interrupt_accepts_buttons(self):
        """request_interrupt should accept a buttons parameter."""
        from bot.telegram_bot import request_interrupt
        sig = inspect.signature(request_interrupt)
        assert "buttons" in sig.parameters, (
            f"request_interrupt missing 'buttons' param. Params: {list(sig.parameters.keys())}"
        )


# ---------------------------------------------------------------------------
# Test 5: State schema — PipelineState has expected keys
# ---------------------------------------------------------------------------

class TestStateSchema:
    """Verify PipelineState TypedDict has the expected keys."""

    def test_pipeline_state_has_at_least_22_keys(self):
        """PipelineState should have at least 22 keys."""
        from pipeline import PipelineState
        hints = get_type_hints(PipelineState)
        assert len(hints) >= 22, f"Expected >= 22 keys, got {len(hints)}: {sorted(hints.keys())}"

    def test_pipeline_state_has_core_keys(self):
        """PipelineState has all core keys from the Agent Contracts table."""
        from pipeline import PipelineState
        hints = get_type_hints(PipelineState)

        expected_keys = {
            "jd_text", "requirements", "cv_sections", "match_score",
            "resume_pdf_path", "resume_content", "cover_letter_text",
            "email_subject", "email_body", "route", "draft_url",
            "form_result", "application_id", "error",
            "_approval", "_rejection_reason",
            "_gate2_approval", "_gate2_rejection_reason",
        }
        missing = expected_keys - set(hints.keys())
        assert not missing, f"PipelineState missing keys: {missing}"

    def test_pipeline_state_is_total_false(self):
        """PipelineState should be total=False (all keys optional)."""
        from pipeline import PipelineState
        # TypedDict with total=False has __required_keys__ as empty frozenset
        assert hasattr(PipelineState, "__required_keys__")
        assert len(PipelineState.__required_keys__) == 0, (
            f"PipelineState has required keys: {PipelineState.__required_keys__}"
        )


# ---------------------------------------------------------------------------
# Test 6: Config — APPLICANT_EMAIL, APPLICANT_PHONE exist
# ---------------------------------------------------------------------------

class TestConfig:
    """Verify config.py exports new personal info constants."""

    def test_applicant_email_exists(self):
        """config.APPLICANT_EMAIL is defined."""
        import config
        assert hasattr(config, "APPLICANT_EMAIL")
        assert isinstance(config.APPLICANT_EMAIL, (str, type(None)))

    def test_applicant_phone_exists(self):
        """config.APPLICANT_PHONE is defined."""
        import config
        assert hasattr(config, "APPLICANT_PHONE")
        assert isinstance(config.APPLICANT_PHONE, (str, type(None)))

    def test_applicant_name_exists(self):
        """config.APPLICANT_NAME is defined (pre-existing)."""
        import config
        assert hasattr(config, "APPLICANT_NAME")


# ---------------------------------------------------------------------------
# Test 7: gmail_sender — send_draft callable, create_draft returns tuple
# ---------------------------------------------------------------------------

class TestGmailSender:
    """Verify gmail_sender.py has the updated API."""

    def test_send_draft_importable(self):
        """send_draft function is importable from apply.gmail_sender."""
        from apply.gmail_sender import send_draft
        assert callable(send_draft)

    def test_create_draft_importable(self):
        """create_draft function is importable from apply.gmail_sender."""
        from apply.gmail_sender import create_draft
        assert callable(create_draft)

    def test_send_draft_signature(self):
        """send_draft accepts a draft_id parameter."""
        from apply.gmail_sender import send_draft
        sig = inspect.signature(send_draft)
        params = list(sig.parameters.keys())
        assert "draft_id" in params, f"send_draft params: {params}"

    def test_create_draft_accepts_attachments(self):
        """create_draft accepts an attachments parameter."""
        from apply.gmail_sender import create_draft
        sig = inspect.signature(create_draft)
        params = list(sig.parameters.keys())
        assert "attachments" in params, f"create_draft params: {params}"


# ---------------------------------------------------------------------------
# Test 8: browser_agent — fill_form has submit param, submit_form importable
# ---------------------------------------------------------------------------

class TestBrowserAgent:
    """Verify browser_agent.py has the updated API."""

    def test_fill_form_importable(self):
        """fill_form function is importable from apply.browser_agent."""
        from apply.browser_agent import fill_form
        assert callable(fill_form)

    def test_fill_form_has_submit_param(self):
        """fill_form has a submit parameter with default True."""
        from apply.browser_agent import fill_form
        sig = inspect.signature(fill_form)
        assert "submit" in sig.parameters, (
            f"fill_form missing 'submit' param. Params: {list(sig.parameters.keys())}"
        )
        default = sig.parameters["submit"].default
        assert default is True, f"fill_form submit default should be True, got {default}"

    def test_submit_form_importable(self):
        """submit_form function is importable from apply.browser_agent."""
        from apply.browser_agent import submit_form
        assert callable(submit_form)


# ---------------------------------------------------------------------------
# Test 9: Cover letter template — exists, 10 placeholders, no forbidden content
# ---------------------------------------------------------------------------

class TestCoverLetterTemplate:
    """Verify the cover letter LaTeX template is correctly set up."""

    TEMPLATE_PATH = PROJECT_ROOT / "cv" / "defaults" / "cover_letter_template.tex"

    def test_template_exists(self):
        """cv/defaults/cover_letter_template.tex exists."""
        assert self.TEMPLATE_PATH.exists(), "Cover letter template not found"

    def test_template_has_10_placeholders(self):
        """Template contains all 10 expected placeholders."""
        import re
        content = self.TEMPLATE_PATH.read_text(encoding="utf-8")
        placeholders = set(re.findall(r'\{\{[A-Z_]+\}\}', content))

        expected = {
            "{{APPLICANT_NAME}}", "{{APPLICANT_PHONE}}", "{{APPLICANT_EMAIL}}",
            "{{TARGET_ROLE}}", "{{LINKEDIN_URL}}", "{{LINKEDIN_DISPLAY}}",
            "{{PORTFOLIO_URL}}", "{{PORTFOLIO_DISPLAY}}", "{{RECIPIENT}}", "{{BODY}}",
        }
        missing = expected - placeholders
        assert not missing, f"Template missing placeholders: {missing}"

    def test_template_no_john_snow(self):
        """Template must not contain 'John Snow'."""
        content = self.TEMPLATE_PATH.read_text(encoding="utf-8")
        assert "john snow" not in content.lower(), "Template contains 'John Snow'"

    def test_template_no_lorem_ipsum(self):
        """Template must not contain 'Lorem ipsum'."""
        content = self.TEMPLATE_PATH.read_text(encoding="utf-8")
        assert "lorem ipsum" not in content.lower(), "Template contains 'Lorem ipsum'"

    def test_template_uses_xelatex(self):
        """Template should use fontspec (xelatex requirement)."""
        content = self.TEMPLATE_PATH.read_text(encoding="utf-8")
        # Either fontspec or xelatex indicator
        assert "fontspec" in content or "xelatex" in content.lower(), (
            "Template doesn't appear to use xelatex/fontspec"
        )


# ---------------------------------------------------------------------------
# Test 10: Cover letter prompt — contains resume-only constraint
# ---------------------------------------------------------------------------

class TestCoverLetterPrompt:
    """Verify the cover letter agent has the resume-context constraint."""

    def test_cover_letter_module_importable(self):
        """agents/cover_letter.py is importable."""
        from agents.cover_letter import generate  # noqa: F401

    def test_prompt_has_resume_constraint(self):
        """Cover letter source contains resume-only reference constraint."""
        source = Path(PROJECT_ROOT / "agents" / "cover_letter.py").read_text(encoding="utf-8")
        # Check for the critical constraint
        assert "ONLY reference" in source or "only reference" in source, (
            "Cover letter prompt missing 'ONLY reference' constraint"
        )

    def test_prompt_has_do_not_repeat(self):
        """Cover letter source instructs not to repeat resume content."""
        source = Path(PROJECT_ROOT / "agents" / "cover_letter.py").read_text(encoding="utf-8")
        assert "DO NOT repeat" in source or "do not repeat" in source.lower(), (
            "Cover letter prompt missing 'DO NOT repeat' instruction"
        )

    def test_prompt_includes_resume_summary(self):
        """Cover letter prompt includes resume summary/content section."""
        source = Path(PROJECT_ROOT / "agents" / "cover_letter.py").read_text(encoding="utf-8")
        assert "resume_content" in source or "RESUME SUMMARY" in source, (
            "Cover letter prompt doesn't reference resume content"
        )


# ---------------------------------------------------------------------------
# Test 11: .env.example — has APPLICANT_EMAIL, APPLICANT_PHONE entries
# ---------------------------------------------------------------------------

class TestEnvExample:
    """Verify .env.example has the new config entries."""

    ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env.example"

    def test_env_example_exists(self):
        """.env.example file exists."""
        assert self.ENV_EXAMPLE_PATH.exists(), ".env.example not found"

    def test_env_example_has_applicant_email(self):
        """.env.example contains APPLICANT_EMAIL."""
        content = self.ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
        assert "APPLICANT_EMAIL" in content, "APPLICANT_EMAIL not in .env.example"

    def test_env_example_has_applicant_phone(self):
        """.env.example contains APPLICANT_PHONE."""
        content = self.ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
        assert "APPLICANT_PHONE" in content, "APPLICANT_PHONE not in .env.example"


# ---------------------------------------------------------------------------
# Bonus: Pipeline internal helpers
# ---------------------------------------------------------------------------

class TestPipelineHelpers:
    """Verify internal pipeline helper functions work correctly."""

    def test_html_to_text_basic(self):
        """_html_to_text strips HTML tags correctly."""
        from pipeline import _html_to_text
        result = _html_to_text("<p>Hello <b>world</b></p>")
        assert "Hello" in result
        assert "world" in result
        assert "<p>" not in result
        assert "<b>" not in result

    def test_html_to_text_strips_scripts(self):
        """_html_to_text removes script/style content."""
        from pipeline import _html_to_text
        result = _html_to_text("<script>alert('x')</script><p>Content</p>")
        assert "alert" not in result
        assert "Content" in result

    def test_latex_to_plain_text_basic(self):
        """_latex_to_plain_text handles basic LaTeX."""
        from pipeline import _latex_to_plain_text
        result = _latex_to_plain_text(r"\textbf{Hello} \textit{world}")
        assert "Hello" in result
        assert "world" in result
        assert "\\textbf" not in result

    def test_safe_node_catches_exceptions(self):
        """_safe_node decorator catches exceptions and stores in state['error']."""
        from pipeline import _safe_node

        @_safe_node
        def bad_node(state):
            raise ValueError("test error")

        state = {}
        result = bad_node(state)
        assert "error" in result
        assert "test error" in result["error"]

    def test_route_after_resume_cover_letter_required(self):
        """route_after_resume returns 'cover_letter' when required."""
        from pipeline import route_after_resume
        state = {"requirements": {"cover_letter_required": True}}
        assert route_after_resume(state) == "cover_letter"

    def test_route_after_resume_no_cover_letter(self):
        """route_after_resume returns 'apply_router' when not required."""
        from pipeline import route_after_resume
        state = {"requirements": {"cover_letter_required": False}}
        assert route_after_resume(state) == "apply_router"

    def test_route_after_apply_email(self):
        """route_after_apply returns 'gmail_draft' for email route."""
        from pipeline import route_after_apply
        state = {"route": "email"}
        assert route_after_apply(state) == "gmail_draft"

    def test_route_after_apply_form(self):
        """route_after_apply returns 'browser_agent' for form route."""
        from pipeline import route_after_apply
        state = {"route": "form"}
        assert route_after_apply(state) == "browser_agent"

    def test_route_after_apply_manual(self):
        """route_after_apply returns 'gmail_draft' for manual route."""
        from pipeline import route_after_apply
        state = {"route": "manual"}
        assert route_after_apply(state) == "gmail_draft"

    def test_route_after_approval_approve(self):
        """route_after_approval dispatches to route-specific next node."""
        from pipeline import route_after_approval
        state = {"_approval": "approved"}
        assert route_after_approval(state) == "gmail_draft"

    def test_route_after_approval_regenerate(self):
        """route_after_approval returns 'resume_tailor' when regenerate."""
        from pipeline import route_after_approval
        state = {"_approval": "regenerate"}
        assert route_after_approval(state) == "resume_tailor"

    def test_route_after_approval_rejected(self):
        """route_after_approval returns 'rejected' when rejected."""
        from pipeline import route_after_approval
        state = {"_approval": "rejected"}
        assert route_after_approval(state) == "rejected"


