"""
CouchHire — LangGraph pipeline orchestrator.

Wires all agents into a state-machine graph with conditional edges and
human-in-the-loop approval gates via Telegram.

Usage:
    python pipeline.py --jd "paste JD here"
    python pipeline.py --url "https://jobs.example.com/job/12345"
    python pipeline.py --search --query "role" --location "city"
"""

from __future__ import annotations

import argparse
import json
import functools
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

import config  # validates env vars on import

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline state schema
# ---------------------------------------------------------------------------
class PipelineState(TypedDict, total=False):
    """Shared state dict passed through every LangGraph node."""

    # Input
    jd_text: str
    jd_url: str | None
    source: str  # "cli" | "telegram" | "dashboard"

    # Parsed
    requirements: dict

    # CV retrieval
    cv_sections: list[str]

    # Scoring
    match_score: float

    # Resume generation
    resume_pdf_path: str
    resume_content: str

    # Cover letter
    cover_letter_text: str | None
    cover_letter_pdf_path: str | None

    # Email draft
    email_subject: str
    email_body: str

    # Routing
    route: str  # "email" | "form" | "manual"

    # Gmail draft
    draft_url: str | None
    draft_id: str | None

    # Browser agent
    form_result: dict | None

    # DB tracking
    application_id: str | None

    # Gate 1: resume/cover-letter review (approve | regenerate | cancel)
    _approval: str | None
    _rejection_reason: str | None

    # Gate 2: send/submit review (approve | cancel)
    _gate2_approval: str | None
    _gate2_rejection_reason: str | None

    # Send status
    email_sent: bool
    sent_url: str | None

    # Error
    error: str | None


# ---------------------------------------------------------------------------
# _safe_node decorator
#
# Wraps every node function so that unhandled exceptions are caught, logged,
# and stored in state["error"] instead of crashing the graph.  The graph's
# conditional edges can then route to the error terminal node.
# ---------------------------------------------------------------------------

def _safe_node(fn):
    """Decorator: catch exceptions in a node, store in state['error']."""
    @functools.wraps(fn)
    def wrapper(state: PipelineState) -> PipelineState:
        try:
            return fn(state)
        except Exception as exc:
            logger.error(
                "Node %s failed: %s\n%s",
                fn.__name__,
                exc,
                traceback.format_exc(),
            )
            state["error"] = f"{fn.__name__}: {exc}"
            return state
    return wrapper


# ---------------------------------------------------------------------------
# Node functions
#
# Each node reads from / writes to `state` per the Agent Contracts table
# in CLAUDE.md.  Exceptions are caught, logged, and stored in state["error"]
# so the graph can route to the error-handling terminal node.
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Shared HTML-to-text helper (stdlib only — no BeautifulSoup)
# ---------------------------------------------------------------------------

from html.parser import HTMLParser as _StdlibHTMLParser


class _HTMLToText(_StdlibHTMLParser):
    """Lightweight HTML stripper using only stdlib html.parser."""

    _SKIP_TAGS = frozenset({"script", "style", "noscript", "head", "meta", "link",
                            "nav", "footer", "header"})

    def __init__(self) -> None:
        super().__init__()
        self._pieces: list[str] = []
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag in ("br", "p", "div", "li", "h1", "h2", "h3", "h4", "tr"):
            self._pieces.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in ("p", "div", "li", "h1", "h2", "h3", "h4", "tr"):
            self._pieces.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._pieces.append(data)

    def get_text(self) -> str:
        import re as _re
        raw = "".join(self._pieces)
        raw = _re.sub(r"\n{3,}", "\n\n", raw)
        raw = _re.sub(r"[ \t]+", " ", raw)
        return raw.strip()


def _html_to_text(html: str) -> str:
    """Convert HTML to plain text using stdlib html.parser."""
    parser = _HTMLToText()
    parser.feed(html)
    return parser.get_text()


# ── LaTeX-to-plain-text converter ──────────────────────────────────────
# Used to convert a tailored .tex resume into readable plain text when
# the LLM-generated resume_content summary is unavailable. This gives
# the cover letter agent rich context about what the resume contains.

def _latex_to_plain_text(latex: str) -> str:
    """Convert LaTeX source to readable plain text.

    Handles standard LaTeX commands, Jake's Resume template macros,
    moderncv macros, and fontawesome icons. Returns clean multi-line text
    suitable for LLM consumption.
    """
    import re as _re

    text = latex

    # ── Remove preamble (everything before \begin{document}) ──
    doc_start = text.find(r"\begin{document}")
    if doc_start != -1:
        text = text[doc_start + len(r"\begin{document}"):]
    doc_end = text.find(r"\end{document}")
    if doc_end != -1:
        text = text[:doc_end]

    # ── Remove comments ──
    text = _re.sub(r"(?<!\\)%.*$", "", text, flags=_re.MULTILINE)

    # ── Section headings → plain headers ──
    text = _re.sub(r"\\section\{([^}]*)\}", r"\n\n== \1 ==\n", text)
    text = _re.sub(r"\\subsection\{([^}]*)\}", r"\n— \1 —\n", text)

    # ── Formatting commands → content only ──
    text = _re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    text = _re.sub(r"\\textit\{([^}]*)\}", r"\1", text)
    text = _re.sub(r"\\underline\{([^}]*)\}", r"\1", text)
    text = _re.sub(r"\\emph\{([^}]*)\}", r"\1", text)
    text = _re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", text)
    text = _re.sub(r"\\url\{([^}]*)\}", r"\1", text)

    # ── Font size commands ──
    for cmd in ("tiny", "scriptsize", "footnotesize", "small", "normalsize",
                "large", "Large", "LARGE", "huge", "Huge"):
        text = _re.sub(rf"\\{cmd}\b\s*", "", text)

    # ── Fontawesome icons ──
    text = _re.sub(r"\\fa[A-Z][a-zA-Z]*\s*", "", text)

    # ── Layout commands ──
    text = _re.sub(r"\\hfill\s*", " | ", text)
    text = _re.sub(r"\\\\", "\n", text)
    text = _re.sub(r"\\newline\b", "\n", text)
    text = _re.sub(r"\\item\s*", "• ", text)
    text = _re.sub(r"\\vspace\{[^}]*\}", "", text)
    text = _re.sub(r"\\hspace\{[^}]*\}", " ", text)
    text = _re.sub(r"\\[hv]space\*\{[^}]*\}", "", text)
    text = _re.sub(r"\\noindent\b", "", text)
    text = _re.sub(r"\\centering\b", "", text)
    text = _re.sub(r"\\raggedright\b", "", text)

    # ── Environment begin/end ──
    text = _re.sub(r"\\begin\{[^}]*\}(\[[^\]]*\])?", "", text)
    text = _re.sub(r"\\end\{[^}]*\}", "", text)

    # ── Spacing / rules ──
    text = _re.sub(r"\\hrule\b", "---", text)
    text = _re.sub(r"\\rule\{[^}]*\}\{[^}]*\}", "---", text)
    text = _re.sub(r"\\titlerule\b", "---", text)

    # ── Jake's Resume template macros ──
    text = _re.sub(r"\\resumeItem\{([^}]*)\}", r"• \1", text)
    text = _re.sub(
        r"\\resumeProjectHeading\{([^}]*)\}\{([^}]*)\}", r"\1 | \2", text
    )
    text = _re.sub(
        r"\\resumeSubheading\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}",
        r"\1 | \2\n  \3 | \4", text,
    )
    text = _re.sub(
        r"\\resumeSubHeadingListStart|\\resumeSubHeadingListEnd", "", text
    )
    text = _re.sub(
        r"\\resumeItemListStart|\\resumeItemListEnd", "", text
    )

    # ── moderncv macros ──
    text = _re.sub(
        r"\\cventry\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}",
        r"\2 | \3 | \1 | \4", text,
    )
    text = _re.sub(r"\\cvitem\{([^}]*)\}\{([^}]*)\}", r"\1: \2", text)

    # ── Generic remaining commands with one arg → content ──
    text = _re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)

    # ── Math mode ──
    text = _re.sub(r"\$([^$]*)\$", r"\1", text)

    # ── LaTeX special chars ──
    text = text.replace(r"\&", "&")
    text = text.replace(r"\%", "%")
    text = text.replace(r"\#", "#")
    text = text.replace(r"\_", "_")
    text = text.replace(r"\$", "$")
    text = text.replace("~", " ")
    text = text.replace(r"\,", " ")
    text = text.replace(r"\;", " ")
    text = text.replace(r"\!", "")
    text = text.replace(r"\LaTeX", "LaTeX")
    text = text.replace(r"\TeX", "TeX")

    # ── Collapse whitespace ──
    text = _re.sub(r"[ \t]+", " ", text)
    text = _re.sub(r"\n{3,}", "\n\n", text)
    text = _re.sub(r"^\s+$", "", text, flags=_re.MULTILINE)

    return text.strip()


def _read_tex_as_plain_text(tex_path: str) -> str:
    """Read a .tex file and return its content as plain text.

    Convenience wrapper: reads the file then runs _latex_to_plain_text().
    Returns empty string if the file doesn't exist or can't be read.
    """
    from pathlib import Path

    p = Path(tex_path)
    if not p.exists() or not p.suffix == ".tex":
        return ""
    try:
        return _latex_to_plain_text(p.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read/convert %s: %s", tex_path, exc)
        return ""

@_safe_node
def node_scrape_jd(state: PipelineState) -> PipelineState:
    """Scrape JD text from a URL when jd_text is empty. Uses _html_to_text()."""
    import urllib.request

    logger.info("▶ node_scrape_jd")

    # If we already have JD text, nothing to do
    if state.get("jd_text", "").strip():
        logger.info("  jd_text already provided — skipping scrape")
        return state

    url = state.get("jd_url", "").strip()
    if not url:
        logger.warning("  No jd_text and no jd_url — nothing to scrape")
        state["error"] = "No JD text or URL provided."
        return state

    logger.info("  Scraping JD from URL: %s", url)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CouchHire/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html_bytes = resp.read()
            charset = resp.headers.get_content_charset() or "utf-8"
            html_text = html_bytes.decode(charset, errors="replace")

        scraped = _html_to_text(html_text)

        if len(scraped) < 50:
            logger.warning("  Scraped text too short (%d chars) — may not be a JD", len(scraped))
            state["error"] = f"Scraped page too short ({len(scraped)} chars). Provide JD text directly."
            return state

        state["jd_text"] = scraped
        logger.info("  Scraped %d chars of JD text", len(scraped))

    except Exception as exc:
        logger.error("  Failed to scrape %s: %s", url, exc)
        state["error"] = f"Failed to scrape JD URL: {exc}"

    return state

@_safe_node
def node_parse_jd(state: PipelineState) -> PipelineState:
    """Parse the job description into structured requirements."""
    from agents.jd_parser import parse_jd

    logger.info("▶ node_parse_jd")
    _update_db_status(state, "parsing")

    requirements = parse_jd(state["jd_text"])
    state["requirements"] = requirements

    company = requirements.get("company", "Unknown")
    role = requirements.get("role", "Unknown")
    logger.info(
        "Parsed JD — company=%s role=%s apply_method=%s cover_letter=%s",
        company,
        role,
        requirements.get("apply_method"),
        requirements.get("cover_letter_required"),
    )
    return state


@_safe_node
def node_cv_rag(state: PipelineState) -> PipelineState:
    """Retrieve relevant CV sections from ChromaDB."""
    from agents.cv_rag import retrieve_cv_sections

    logger.info("▶ node_cv_rag")
    cv_sections = retrieve_cv_sections(state["requirements"])
    state["cv_sections"] = cv_sections
    logger.info("Retrieved %d CV sections", len(cv_sections))
    return state


@_safe_node
def node_match_scorer(state: PipelineState) -> PipelineState:
    """Compute match score between JD and CV."""
    from agents.match_scorer import score

    logger.info("▶ node_match_scorer")
    _update_db_status(state, "scoring")

    match_score = score(state["jd_text"], state["cv_sections"])
    state["match_score"] = match_score
    logger.info("Match score: %.1f", match_score)
    return state


@_safe_node
def node_threshold_gate(state: PipelineState) -> PipelineState:
    """Check if match score meets the threshold.  Low scores are flagged."""
    logger.info("▶ node_threshold_gate — score=%.1f threshold=%d",
                state["match_score"], config.MATCH_THRESHOLD)

    if state["match_score"] < config.MATCH_THRESHOLD:
        _update_db_status(state, "below_threshold")
    return state


@_safe_node
def node_resume_tailor(state: PipelineState) -> PipelineState:
    """Generate a tailored resume PDF."""
    from agents.resume_tailor import tailor

    logger.info("▶ node_resume_tailor")
    _update_db_status(state, "tailoring")

    pdf_path, resume_content = tailor(state["cv_sections"], state["requirements"])
    state["resume_pdf_path"] = pdf_path
    state["resume_content"] = resume_content
    logger.info("Resume PDF: %s", pdf_path)
    return state


@_safe_node
def node_cover_letter(state: PipelineState) -> PipelineState:
    """Generate a cover letter that complements the resume."""
    from agents.cover_letter import generate

    logger.info("▶ node_cover_letter")

    # Use LLM-generated resume summary; fall back to LaTeX→plain text
    resume_context = state.get("resume_content", "")
    if not resume_context and state.get("resume_pdf_path"):
        tex_path = state["resume_pdf_path"].replace(".pdf", ".tex")
        resume_context = _read_tex_as_plain_text(tex_path)
        if resume_context:
            logger.info(
                "  resume_content empty — fell back to LaTeX→plain text "
                "(%d chars)", len(resume_context),
            )

    text = generate(
        state["requirements"],
        state["cv_sections"],
        resume_context,
    )
    state["cover_letter_text"] = text or None
    logger.info(
        "Cover letter: %s",
        f"{len(text)} chars" if text else "skipped",
    )
    return state



@_safe_node
def node_compile_cover_letter_pdf(state: PipelineState) -> PipelineState:
    """Compile cover letter text into a PDF using xelatex + template substitution."""
    import re as _re
    import subprocess
    import tempfile
    import shutil

    logger.info("▶ node_compile_cover_letter_pdf")
    print("\n\n=== DEBUG: node_compile_cover_letter_pdf ENTERED ===\n")

    cl_text = state.get("cover_letter_text")
    if not cl_text:
        logger.info("  No cover letter text — skipping PDF compilation")
        state["cover_letter_pdf_path"] = None
        return state

    # --- Locate cover letter template ---
    project_root = Path(__file__).resolve().parent
    user_tpl = project_root / "cv" / "uploads" / "cover_letter_template.tex"
    default_tpl = project_root / "cv" / "defaults" / "cover_letter_template.tex"
    tpl_path = user_tpl if user_tpl.exists() else default_tpl

    if not tpl_path.exists():
        logger.warning("  No cover letter template found — skipping PDF")
        state["cover_letter_pdf_path"] = None
        return state

    template_src = tpl_path.read_text(encoding="utf-8")

    # --- Font fallback ---
    # Handle %%FONT_PLACEHOLDER%% marker (user template) or inline \setmainfont
    # When fonts/ dir is missing, replace with a system font.
    fonts_dir = tpl_path.parent / "fonts"
    _font_fallback = r"\setmainfont{Helvetica}"

    if "%%FONT_PLACEHOLDER%%" in template_src:
        # User template uses a marker — check if fonts are available
        _font_src_dir = project_root / "cv" / "uploads" / "fonts"
        if _font_src_dir.is_dir():
            # Use SourceSansPro from uploads/fonts/
            template_src = template_src.replace(
                "%%FONT_PLACEHOLDER%%",
                "\\setmainfont[\n"
                "BoldFont=SourceSansPro-Semibold.otf,\n"
                "ItalicFont=SourceSansPro-RegularIt.otf\n"
                "]{SourceSansPro-Regular.otf}",
            )
        else:
            template_src = template_src.replace(
                "%%FONT_PLACEHOLDER%%", _font_fallback
            )
            logger.info("  fonts/ dir not found — using Helvetica fallback")
    elif not fonts_dir.is_dir() and "\\setmainfont" in template_src:
        # Default template or any template with inline \setmainfont
        # Pattern 1: \setmainfont{...}[...]  (options after name)
        template_src = _re.sub(
            r"\\setmainfont\{[^}]*\}\[.*?\]",
            _font_fallback, template_src, flags=_re.DOTALL,
        )
        # Pattern 2: \setmainfont[...]{...}  (options before name)
        template_src = _re.sub(
            r"\\setmainfont\[.*?\]\{[^}]*\}",
            _font_fallback, template_src, flags=_re.DOTALL,
        )
        # Pattern 3: \setmainfont{...}  (no options at all)
        template_src = _re.sub(
            r"\\setmainfont\{[^}]*\.otf\}",
            _font_fallback, template_src,
        )
        logger.info("  fonts/ dir not found — falling back to Helvetica")


    # --- Escape LaTeX special chars in plain text ---
    def _esc(text: str) -> str:
        for c, r in [("\\", r"\textbackslash{}"), ("&", r"\&"),
                      ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                      ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                      ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
            text = text.replace(c, r)
        text = _re.sub(r"\n\n+", "\n\n\\\\par\n", text)
        return text

    # --- Build substitution map ---
    import config as _cfg

    reqs = state.get("requirements", {})
    role = reqs.get("role") or "the position"
    company = reqs.get("company") or "your company"
    recipient = reqs.get("hiring_manager") or "Hiring Manager"

    # Strip protocol + trailing slash for display-friendly URL text
    def _display_url(url: str | None) -> str:
        """Strip protocol, trailing slash, and escape for LaTeX text mode."""
        if not url:
            return ""
        display = _re.sub(r"^https?://", "", url).rstrip("/")
        # Escape underscores for LaTeX text mode (inside \href display arg)
        display = display.replace("_", r"\_")
        return display

    subs = {
        "{{APPLICANT_NAME}}": _cfg.APPLICANT_NAME or "Applicant",
        "{{APPLICANT_EMAIL}}": _cfg.APPLICANT_EMAIL or "",
        "{{APPLICANT_PHONE}}": _cfg.APPLICANT_PHONE or "",
        "{{TARGET_ROLE}}": _esc(f"{role} - {company}"),
        "{{LINKEDIN_URL}}": _cfg.APPLICANT_LINKEDIN or "",
        "{{LINKEDIN_DISPLAY}}": _display_url(_cfg.APPLICANT_LINKEDIN),
        "{{PORTFOLIO_URL}}": _cfg.GITHUB_URL or "",
        "{{PORTFOLIO_DISPLAY}}": _display_url(_cfg.GITHUB_URL),
        "{{RECIPIENT}}": _esc(recipient),
        "{{BODY}}": _esc(cl_text),
    }

    latex_src = template_src
    for placeholder, value in subs.items():
        latex_src = latex_src.replace(placeholder, value)

    # --- Handle conditional blocks: %%IF_<KEY>%% ... %%ENDIF_<KEY>%% ---
    # When a value is empty, strip the entire block (markers + content).
    # When a value is present, keep the content but remove the markers.
    _conditionals = {
        "PHONE": _cfg.APPLICANT_PHONE,
        "LINKEDIN": _cfg.APPLICANT_LINKEDIN,
        "PORTFOLIO": _cfg.GITHUB_URL,
    }
    for key, val in _conditionals.items():
        open_tag = f"%%IF_{key}%%"
        close_tag = f"%%ENDIF_{key}%%"
        if val:
            # Value present — keep content, strip markers
            latex_src = latex_src.replace(open_tag, "")
            latex_src = latex_src.replace(close_tag, "")
        else:
            # Value empty — remove entire block (markers + content between)
            pattern = _re.escape(open_tag) + r".*?" + _re.escape(close_tag)
            latex_src = _re.sub(pattern, "", latex_src, flags=_re.DOTALL)


    # --- Compile with xelatex ---
    from config import get_output_dir
    output_dir = get_output_dir(str(company), str(role))
    pdf_name = "Cover Letter.pdf"

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_file = Path(tmpdir) / "cover_letter.tex"
        tex_file.write_text(latex_src, encoding="utf-8")

        # Copy fonts directory if template references local fonts
        fonts_src = tpl_path.parent / "fonts"
        if fonts_src.is_dir():
            shutil.copytree(fonts_src, Path(tmpdir) / "fonts")

        try:
            proc = subprocess.run(
                ["xelatex", "-interaction=nonstopmode",
                 "-halt-on-error", str(tex_file)],
                cwd=tmpdir, capture_output=True, text=True, timeout=60,
            )
            print(f"=== DEBUG: xelatex exit={proc.returncode}, pdf_exists={Path(tmpdir).joinpath(chr(99)+chr(111)+chr(118)+chr(101)+chr(114)+chr(95)+chr(108)+chr(101)+chr(116)+chr(116)+chr(101)+chr(114)+chr(46)+chr(112)+chr(100)+chr(102)).exists()}")
            compiled = Path(tmpdir) / "cover_letter.pdf"
            if compiled.exists():
                final = output_dir / pdf_name
                shutil.copy2(compiled, final)
                state["cover_letter_pdf_path"] = str(final)
                logger.info("  Cover letter PDF compiled: %s", final)
            else:
                # Read .log file for detailed error info
                log_file = Path(tmpdir) / "cover_letter.log"
                log_tail = ""
                if log_file.exists():
                    log_lines = log_file.read_text(errors="replace").split("\n")
                    err_lines = [l for l in log_lines if l.startswith("!")]
                    tail = log_lines[-15:]
                    log_tail = "\n".join(err_lines + ["---"] + tail)
                logger.error(
                    "  xelatex produced no PDF (exit %d).\nLog:\n%s",
                    proc.returncode, log_tail[-1000:],
                )
                state["cover_letter_pdf_path"] = None
        except FileNotFoundError:
            logger.warning("  xelatex not installed — cover letter PDF skipped")
            state["cover_letter_pdf_path"] = None
        except subprocess.TimeoutExpired:
            logger.warning("  xelatex timed out — cover letter PDF skipped")
            state["cover_letter_pdf_path"] = None

    return state

@_safe_node
def node_email_drafter(state: PipelineState) -> PipelineState:
    """Draft the application email subject and body."""
    from agents.email_drafter import draft

    logger.info("▶ node_email_drafter")
    _update_db_status(state, "drafting")

    subject, body = draft(
        state["requirements"],
        state.get("cover_letter_text") or "",
        state.get("resume_pdf_path", ""),
        resume_content=state.get("resume_content", ""),
    )
    state["email_subject"] = subject
    state["email_body"] = body
    logger.info("Email drafted — subject: %s", subject)
    return state


@_safe_node
def node_apply_router(state: PipelineState) -> PipelineState:
    """Determine the application route: email, form, or manual."""
    from agents.apply_router import route

    logger.info("▶ node_apply_router")
    apply_route = route(state["requirements"])
    state["route"] = apply_route
    logger.info("Route: %s", apply_route)
    return state


@_safe_node
def node_approval_gate(state: PipelineState) -> PipelineState:
    """Human-in-the-loop: send resume + draft to Telegram for review."""
    from bot.telegram_bot import (
        request_interrupt,
        send_document,
        send_job_card,
    )

    logger.info("▶ node_approval_gate")

    reqs = state["requirements"]
    company = reqs.get("company", "Unknown")
    role = reqs.get("role", "Unknown")

    # Send the job card summary
    send_job_card(
        company=company,
        role=role,
        score=state["match_score"],
        route=state["route"],
    )

    # Send the tailored resume PDF
    pdf_path = state.get("resume_pdf_path")
    if pdf_path and Path(pdf_path).exists():
        send_document(pdf_path, caption=f"📄 Tailored resume for {role} at {company}")

    # Send the cover letter PDF (if generated)
    cl_pdf_path = state.get("cover_letter_pdf_path")
    if cl_pdf_path and Path(cl_pdf_path).exists():
        send_document(cl_pdf_path, caption=f"📄 Cover letter for {role} at {company}")

    # Build review message
    parts = [
        f"<b>Review Application</b>\n",
        f"<b>Company:</b> {company}",
        f"<b>Role:</b> {role}",
        f"<b>Score:</b> {state['match_score']:.1f}",
        f"<b>Route:</b> {state['route']}",
        f"\n<b>Subject:</b> {state.get('email_subject', 'N/A')}",
    ]
    if state.get("cover_letter_text"):
        # Show first 200 chars of cover letter
        preview = state["cover_letter_text"][:200]
        parts.append(f"\n<b>Cover Letter Preview:</b>\n<i>{preview}...</i>")

    parts.append("\n\nApprove this application?")
    message = "\n".join(parts)

    buttons = [
        {"text": "✅ Approve", "callback_data": "approve"},
        {"text": "❌ Reject", "callback_data": "reject"},
        {"text": "🔄 Regenerate", "callback_data": "regenerate"},
    ]

    response = request_interrupt(
        interrupt_type="application_review",
        message=message,
        buttons=buttons,
        timeout=600,  # 10 minutes
    )

    if response == "approve":
        logger.info("Application APPROVED by user")
        state["_approval"] = "approved"
    elif response == "regenerate":
        logger.info("User requested REGENERATE — looping back to resume_tailor")
        state["_approval"] = "regenerate"
    else:
        # reject or timeout
        logger.info("Application REJECTED or timed out (response=%s)", response)
        state["_approval"] = "rejected"

    return state


@_safe_node
def node_gmail_draft(state: PipelineState) -> PipelineState:
    """Create a Gmail draft via MCP server."""
    from apply.gmail_sender import create_draft

    logger.info("▶ node_gmail_draft")

    reqs = state["requirements"]
    route = state["route"]

    # Determine recipient
    if route == "email":
        recipient = reqs.get("apply_target", "")
    else:
        # form or manual — still create a draft (recipient blank for manual)
        recipient = ""

    # Build attachments list (resume + cover letter if available)
    attachments = []
    if state.get("resume_pdf_path"):
        attachments.append(state["resume_pdf_path"])
    if state.get("cover_letter_pdf_path") and Path(state["cover_letter_pdf_path"]).exists():
        attachments.append(state["cover_letter_pdf_path"])

    try:
        draft_url, draft_id = create_draft(
            subject=state["email_subject"],
            body=state["email_body"],
            recipient_email=recipient,
            attachments=attachments if attachments else None,
        )
        state["draft_url"] = draft_url
        state["draft_id"] = draft_id
        logger.info("Gmail draft created: %s", draft_url)
    except Exception as exc:
        logger.error(
            "Failed to create Gmail draft: %s\n%s",
            exc,
            traceback.format_exc(),
        )
        state["draft_url"] = None
        state["draft_id"] = None
        state["error"] = f"Gmail draft creation failed: {exc}"
        # Non-fatal: we still notify user, but error is logged and visible

    return state


@_safe_node
def node_browser_agent(state: PipelineState) -> PipelineState:
    """Fill an ATS form using the browser agent."""
    from apply.browser_agent import fill_form

    logger.info("▶ node_browser_agent")

    reqs = state["requirements"]
    url = reqs.get("apply_target", "")

    if not url:
        logger.warning("No apply URL found — skipping browser agent")
        state["form_result"] = {
            "status": "failed",
            "url": "",
            "notes": "No application URL found in JD",
        }
        return state

    # Build documents dict for the browser agent
    documents = {
        "resume_pdf_path": state.get("resume_pdf_path", ""),
        "cover_letter_text": state.get("cover_letter_text"),
        "applicant_name": config.APPLICANT_NAME or "",
        "applicant_email": "",  # filled from form_answers or user prompt
        "github_url": config.GITHUB_URL,
        "linkedin_url": getattr(config, "APPLICANT_LINKEDIN", None),
        "phone": getattr(config, "APPLICANT_PHONE", None),
    }

    from bot.telegram_bot import send_form_started
    company = reqs.get("company", "Unknown")
    role = reqs.get("role", "Unknown")
    send_form_started(company=company, role=role)

    try:
        result = fill_form(url=url, documents=documents)
        state["form_result"] = result
        logger.info("Browser agent result: %s", result.get("status"))
    except Exception as exc:
        logger.error("Browser agent failed: %s", exc)
        state["form_result"] = {
            "status": "failed",
            "url": url,
            "notes": f"Browser agent error: {exc}",
        }

    return state



@_safe_node
def node_gate2(state: PipelineState) -> PipelineState:
    """Gate 2: human review of the draft/form before sending/submitting."""
    from bot.telegram_bot import request_interrupt, send_document

    logger.info("▶ node_gate2")

    reqs = state.get("requirements", {})
    company = reqs.get("company", "Unknown")
    role = reqs.get("role", "Unknown")
    route = state.get("route", "manual")

    # Send the draft email preview
    parts = [
        "<b>📨 Ready to Send</b>\n",
        f"<b>Company:</b> {company}",
        f"<b>Role:</b> {role}",
        f"<b>Route:</b> {route}",
        f"\n<b>Subject:</b> {state.get('email_subject', 'N/A')}",
    ]

    draft_url = state.get("draft_url", "")
    if draft_url:
        parts.append(f'\n<a href="{draft_url}">📝 View Gmail Draft</a>')

    if route == "form":
        form_result = state.get("form_result", {})
        form_status = form_result.get("status", "unknown")
        parts.append(f"\n<b>Form status:</b> {form_status}")
        if form_result.get("notes"):
            parts.append(f"<b>Notes:</b> {form_result['notes']}")

    parts.append("\n\nApprove sending/submitting?")
    message = "\n".join(parts)

    buttons = [
        {"text": "🚀 Send", "callback_data": "gate2_approve"},
        {"text": "❌ Cancel", "callback_data": "gate2_cancel"},
    ]

    response = request_interrupt(
        interrupt_type="gate2_review",
        message=message,
        buttons=buttons,
        timeout=600,
    )

    if response == "gate2_approve":
        logger.info("Gate 2: APPROVED — proceeding to send/submit")
        state["_gate2_approval"] = "approved"
    else:
        logger.info("Gate 2: CANCELLED (response=%s)", response)
        state["_gate2_approval"] = "cancelled"
        state["_gate2_rejection_reason"] = f"User cancelled at Gate 2 (response: {response})"

    return state


@_safe_node
def node_execute_send(state: PipelineState) -> PipelineState:
    """Execute the actual send/submit after Gate 2 approval."""
    logger.info("▶ node_execute_send")

    route = state.get("route", "manual")
    draft_id = state.get("draft_id", "")

    if route in ("email", "manual"):
        # Send the email via MCP (re-compose from state, since MCP has no send-draft-by-ID)
        reqs = state.get("requirements", {})
        recipient = reqs.get("apply_target", "") if route == "email" else ""
        subject = state.get("email_subject", "")
        body = state.get("email_body", "")

        if not recipient:
            logger.warning("  No recipient email — cannot send (route=%s)", route)
            return state
        if not subject:
            logger.warning("  No email subject — cannot send")
            return state

        attachments = []
        if state.get("resume_pdf_path"):
            attachments.append(state["resume_pdf_path"])
        if state.get("cover_letter_pdf_path") and Path(state["cover_letter_pdf_path"]).exists():
            attachments.append(state["cover_letter_pdf_path"])

        try:
            from apply.gmail_sender import send_email
            sent = send_email(
                subject=subject,
                body=body,
                recipient_email=recipient,
                attachments=attachments if attachments else None,
            )
            if sent:
                logger.info("  Email sent successfully (to=%s, subject=%s, message_id=%s)", recipient, subject, sent)
                state["email_sent"] = True
                # Build sent URL from message ID (sent is the message_id string)
                if isinstance(sent, str) and sent:
                    state["sent_url"] = f"https://mail.google.com/mail/u/0/#sent/{sent}"
                else:
                    state["sent_url"] = "https://mail.google.com/mail/u/0/#sent"
            else:
                logger.warning("  send_email returned False (to=%s)", recipient)
        except Exception as exc:
            logger.error("  Failed to send email: %s", exc, exc_info=True)
            state["error"] = f"send_email failed: {exc}"

    elif route == "form":
        # Submit the ATS form
        try:
            from apply.browser_agent import submit_form
            result = submit_form()
            state["form_result"] = result
            logger.info("  Form submitted: %s", result.get("status"))
        except Exception as exc:
            logger.error("  Failed to submit form: %s", exc)
            state["error"] = f"submit_form failed: {exc}"

    return state


@_safe_node
def node_notify(state: PipelineState) -> PipelineState:
    """Send final Telegram notification based on route."""
    from bot.telegram_bot import send_draft_ready, send_manual_notice, send_sent_confirmation

    logger.info("▶ node_notify")

    reqs = state["requirements"]
    company = reqs.get("company", "Unknown")
    role = reqs.get("role", "Unknown")
    route = state["route"]
    draft_url = state.get("draft_url")

    if route == "email":
        if state.get("email_sent"):
            # Email was sent after Gate 2 approval — use the actual sent message URL
            sent_url = state.get("sent_url") or "https://mail.google.com/mail/u/0/#sent"
            send_sent_confirmation(company=company, role=role, sent_url=sent_url)
        elif draft_url:
            send_draft_ready(company=company, role=role, draft_url=draft_url)
        else:
            send_manual_notice(company=company, role=role, draft_url=None)
    elif route == "form":
        form_result = state.get("form_result", {})
        form_status = form_result.get("status", "unknown")
        if draft_url:
            # Also created a draft as backup
            send_draft_ready(company=company, role=role, draft_url=draft_url)
        logger.info("Form application status: %s", form_status)
    else:
        # manual
        send_manual_notice(company=company, role=role, draft_url=draft_url)

    if state.get("email_sent"):
        _update_db_status(state, "sent")
    else:
        _update_db_status(state, "awaiting_review")
    return state


@_safe_node
def node_save_to_db(state: PipelineState) -> PipelineState:
    """Persist the application record to Supabase."""
    from db.supabase_client import insert_application, update_application

    logger.info("▶ node_save_to_db")

    reqs = state.get("requirements", {})
    app_id = state.get("application_id")

    data = {
        "jd_text": state.get("jd_text"),
        "jd_url": state.get("jd_url"),
        "requirements": json.dumps(reqs) if reqs else "{}",
        "company": reqs.get("company"),
        "role": reqs.get("role"),
        "match_score": state.get("match_score"),
        "route": state.get("route"),
        "resume_pdf_path": state.get("resume_pdf_path"),
        "resume_content": state.get("resume_content"),
        "cover_letter": state.get("cover_letter_text"),
        "email_subject": state.get("email_subject"),
        "email_body": state.get("email_body"),
        "draft_url": state.get("draft_url"),
        "status": "awaiting_review",
    }

    try:
        if app_id:
            update_application(app_id, data)
            logger.info("Updated application %s in DB", app_id)
        else:
            row = insert_application(data)
            state["application_id"] = row.get("id")
            logger.info("Inserted application %s in DB", row.get("id"))
    except Exception as exc:
        logger.error("DB save failed: %s", exc)
        # Non-fatal — pipeline still completes

    return state


@_safe_node
def node_rejected(state: PipelineState) -> PipelineState:
    """Handle rejected / below-threshold applications."""
    from bot.telegram_bot import send_notification

    logger.info("▶ node_rejected")

    reqs = state.get("requirements", {})
    company = reqs.get("company", "Unknown")
    role = reqs.get("role", "Unknown")
    score_val = state.get("match_score", 0)

    reason = state.get("_rejection_reason", "below_threshold")

    if reason == "user_rejected":
        msg = (
            f"❌ <b>Application rejected by you</b>\n"
            f"{role} at {company} (score: {score_val:.1f})"
        )
    else:
        msg = (
            f"⏭️ <b>Skipped — below threshold</b>\n"
            f"{role} at {company}\n"
            f"Score: {score_val:.1f} (threshold: {config.MATCH_THRESHOLD})"
        )

    send_notification(msg)
    _update_db_status(state, "below_threshold")
    return state


def node_error(state: PipelineState) -> PipelineState:
    """Handle pipeline errors — log to DB and notify user."""
    from bot.telegram_bot import send_notification

    logger.error("▶ node_error — %s", state.get("error"))

    reqs = state.get("requirements", {})
    company = reqs.get("company", "Unknown")
    role = reqs.get("role", "Unknown")

    msg = (
        f"🚨 <b>Pipeline error</b>\n"
        f"{role} at {company}\n"
        f"<code>{state.get('error', 'Unknown error')}</code>"
    )
    try:
        send_notification(msg)
    except Exception:
        logger.error("Failed to send error notification")

    _update_db_status(state, "error", error_message=state.get("error"))
    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _update_db_status(
    state: PipelineState,
    status: str,
    error_message: str | None = None,
) -> None:
    """Update the application status in DB if we have an application_id."""
    app_id = state.get("application_id")
    if not app_id:
        return

    try:
        from db.supabase_client import update_application

        updates: dict[str, Any] = {"status": status}
        if error_message:
            updates["error_message"] = error_message
        update_application(app_id, updates)
    except Exception as exc:
        logger.warning("Failed to update DB status to '%s': %s", status, exc)


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def route_after_threshold(state: PipelineState) -> str:
    """After threshold gate: proceed or reject."""
    score_val = state["match_score"]
    threshold = config.MATCH_THRESHOLD
    logger.info("route_after_threshold — score=%.1f threshold=%d", score_val, threshold)

    if score_val < threshold:
        # Ask user if they want to proceed anyway
        try:
            from bot.telegram_bot import request_interrupt

            reqs = state.get("requirements", {})
            company = reqs.get("company", "Unknown")
            role = reqs.get("role", "Unknown")

            msg = (
                f"⚠️ <b>Low match score</b>\n\n"
                f"<b>{role}</b> at <b>{company}</b>\n"
                f"Score: <b>{state['match_score']:.1f}</b> "
                f"(threshold: {threshold})\n\n"
                f"Proceed anyway?"
            )
            buttons = [
                {"text": "✅ Proceed", "callback_data": "proceed"},
                {"text": "⏭️ Skip", "callback_data": "skip"},
            ]
            response = request_interrupt(
                interrupt_type="threshold_gate",
                message=msg,
                buttons=buttons,
                timeout=300,
            )
            logger.info("Threshold gate user response: %r", response)
            if response == "proceed":
                logger.info("User chose to proceed despite low score")
                return "resume_tailor"
            logger.info("User chose to skip (response=%r)", response)
        except Exception as exc:
            logger.warning("Threshold gate interrupt failed: %s", exc, exc_info=True)

        return "rejected"

    logger.info("Score %.1f >= threshold %d — proceeding to resume_tailor", score_val, threshold)
    return "resume_tailor"


def route_after_resume(state: PipelineState) -> str:
    """After resume_tailor: error if failed, cover_letter if required, else email_drafter."""
    # If resume_tailor failed, halt the pipeline — downstream nodes need resume_pdf_path
    if state.get("error"):
        logger.error("route_after_resume — resume_tailor failed, routing to error node")
        return "error"
    reqs = state.get("requirements", {})
    if reqs.get("cover_letter_required", False):
        return "cover_letter"
    return "email_drafter"


def route_after_approval(state: PipelineState) -> str:
    """After Gate 1: approve, regenerate, or cancel (3-way)."""
    approval = state.get("_approval", "")
    if approval == "approved":
        # apply_router already ran — dispatch directly to the apply method
        route = state.get("route", "manual")
        if route == "email":
            return "gmail_draft"
        elif route == "form":
            return "browser_agent"
        else:
            return "gmail_draft"  # manual still gets a draft (no recipient)
    elif approval == "regenerate":
        return "resume_tailor"
    state["_rejection_reason"] = state.get("_rejection_reason") or "user_rejected"
    return "rejected"



def route_after_apply(state: PipelineState) -> str:
    """After apply_router: dispatch to the correct apply method."""
    route = state.get("route", "manual")
    if route == "email":
        return "gmail_draft"
    elif route == "form":
        return "browser_agent"
    else:
        return "gmail_draft"  # manual still gets a draft (no recipient)



def route_after_gate2(state: PipelineState) -> str:
    """After Gate 2: send/submit or cancel."""
    if state.get("_gate2_approval") == "approved":
        return "execute_send"
    return "rejected"


def route_after_cover_letter(state: PipelineState) -> str:
    """After cover_letter: compile PDF if text exists, else go to email_drafter."""
    if state.get("cover_letter_text"):
        return "compile_cover_letter_pdf"
    return "email_drafter"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Build the LangGraph pipeline with 18 nodes and 5 conditional edges."""
    graph = StateGraph(PipelineState)

    # --- Add all 18 nodes ---
    graph.add_node("scrape_jd", node_scrape_jd)
    graph.add_node("parse_jd", node_parse_jd)
    graph.add_node("cv_rag", node_cv_rag)
    graph.add_node("match_scorer", node_match_scorer)
    graph.add_node("threshold_gate", node_threshold_gate)
    graph.add_node("resume_tailor", node_resume_tailor)
    graph.add_node("cover_letter", node_cover_letter)
    graph.add_node("compile_cover_letter_pdf", node_compile_cover_letter_pdf)
    graph.add_node("email_drafter", node_email_drafter)
    graph.add_node("approval_gate", node_approval_gate)
    graph.add_node("apply_router", node_apply_router)
    graph.add_node("gmail_draft", node_gmail_draft)
    graph.add_node("browser_agent", node_browser_agent)
    graph.add_node("gate2", node_gate2)
    graph.add_node("execute_send", node_execute_send)
    graph.add_node("notify", node_notify)
    graph.add_node("save_to_db", node_save_to_db)
    graph.add_node("rejected", node_rejected)
    graph.add_node("error", node_error)

    # --- Entry point ---
    graph.set_entry_point("scrape_jd")

    # --- Linear edges ---
    graph.add_edge("scrape_jd", "parse_jd")
    graph.add_edge("parse_jd", "cv_rag")
    graph.add_edge("cv_rag", "match_scorer")
    graph.add_edge("match_scorer", "threshold_gate")

    # --- Conditional 1: threshold gate (pass / reject) ---
    graph.add_conditional_edges(
        "threshold_gate",
        route_after_threshold,
        {
            "resume_tailor": "resume_tailor",
            "rejected": "rejected",
        },
    )

    # --- Conditional 2: resume → cover letter or email_drafter ---
    graph.add_conditional_edges(
        "resume_tailor",
        route_after_resume,
        {
            "cover_letter": "cover_letter",
            "email_drafter": "email_drafter",
            "error": "error",
        },
    )

    # --- Conditional 3 (new): cover_letter → compile PDF or email_drafter ---
    graph.add_conditional_edges(
        "cover_letter",
        route_after_cover_letter,
        {
            "compile_cover_letter_pdf": "compile_cover_letter_pdf",
            "email_drafter": "email_drafter",
        },
    )

    # --- compile_cover_letter_pdf → email_drafter ---
    graph.add_edge("compile_cover_letter_pdf", "email_drafter")

    # --- email_drafter → apply_router → approval_gate ---
    graph.add_edge("email_drafter", "apply_router")
    graph.add_edge("apply_router", "approval_gate")

    # --- Conditional 4: Gate 1 (approve / regenerate / cancel) ---
    graph.add_conditional_edges(
        "approval_gate",
        route_after_approval,
        {
            "gmail_draft": "gmail_draft",
            "browser_agent": "browser_agent",
            "resume_tailor": "resume_tailor",
            "rejected": "rejected",
        },
    )

    # --- After gmail_draft → gate2 ---
    graph.add_edge("gmail_draft", "gate2")

    # --- After browser_agent → gmail_draft (backup draft, then gate2) ---
    graph.add_edge("browser_agent", "gmail_draft")

    # --- Conditional 6: Gate 2 (send / cancel) ---
    graph.add_conditional_edges(
        "gate2",
        route_after_gate2,
        {
            "execute_send": "execute_send",
            "rejected": "rejected",
        },
    )

    # --- execute_send → notify → save_to_db → END ---
    graph.add_edge("execute_send", "notify")
    graph.add_edge("notify", "save_to_db")

    # --- Terminal nodes ---
    graph.add_edge("save_to_db", END)
    graph.add_edge("rejected", END)
    graph.add_edge("error", END)

    return graph

def compile_graph():
    """Build and compile the graph, returning a runnable."""
    graph = build_graph()
    return graph.compile()



# Module-level compiled graph — compiled once at import time.
# Use this directly or call compile_graph() for a fresh instance.
_compiled_pipeline = compile_graph()

# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    jd_text: str = "",
    jd_url: str | None = None,
    application_id: str | None = None,
    source: str = "cli",
) -> PipelineState:

    """Run the full CouchHire pipeline for a single job description.

    Returns the final pipeline state dict.
    """
    logger.info("=" * 60)
    logger.info("PIPELINE START")
    logger.info("=" * 60)

    initial_state: PipelineState = {
        "jd_text": jd_text,
        "jd_url": jd_url,
        "application_id": application_id,
        "error": None,
        "source": source,
    }

    # Pre-create DB record if we don't have one
    if not application_id:
        try:
            from db.supabase_client import insert_application

            row = insert_application({
                "jd_text": jd_text,
                "jd_url": jd_url,
                "status": "pending",
            })
            initial_state["application_id"] = row.get("id")
            logger.info("Created application record: %s", row.get("id"))
        except Exception as exc:
            logger.warning("Failed to pre-create DB record: %s", exc)

    app = _compiled_pipeline

    try:
        final_state = app.invoke(initial_state)
    except Exception as exc:
        logger.error("Pipeline failed with unhandled exception: %s", exc)
        logger.error(traceback.format_exc())
        initial_state["error"] = str(exc)
        # Try to record the error
        try:
            node_error(initial_state)
        except Exception:
            pass
        return initial_state

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)

    return final_state


# ---------------------------------------------------------------------------
# JD scraping helper (for --url mode)
# ---------------------------------------------------------------------------

def _scrape_jd(url: str) -> str:
    """Scrape job description text from a URL. Uses stdlib html.parser."""
    import urllib.request

    logger.info("Scraping JD from: %s", url)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CouchHire/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            html_bytes = resp.read()
            charset = resp.headers.get_content_charset() or "utf-8"
            html_text = html_bytes.decode(charset, errors="replace")
    except Exception as exc:
        logger.error("Failed to fetch URL: %s", exc)
        sys.exit(1)

    text = _html_to_text(html_text)

    if not text.strip():
        logger.error("No text extracted from URL")
        sys.exit(1)

    logger.info("Scraped %d characters from URL", len(text))
    return text


# ---------------------------------------------------------------------------
# Job search helper (for --search mode)
# ---------------------------------------------------------------------------

def _search_jobs(query: str, location: str) -> list[dict]:
    """Search for jobs using JobSpy."""
    try:
        from jobspy import scrape_jobs
    except ImportError:
        logger.error("python-jobspy not installed — cannot search jobs")
        sys.exit(1)

    sites = getattr(config, "JOBSPY_SITES", ["indeed", "linkedin", "google"])
    country = getattr(config, "JOBSPY_COUNTRY", "India")
    hours_old = getattr(config, "JOBSPY_HOURS_OLD", 72)
    proxies = getattr(config, "JOBSPY_PROXIES", None)

    logger.info(
        "Searching jobs: query=%s location=%s sites=%s country=%s hours_old=%s",
        query, location, sites, country, hours_old,
    )

    kwargs: dict = {
        "site_name": sites,
        "search_term": query,
        "location": location,
        "country_indeed": country,
        "results_wanted": 20,
    }
    if proxies:
        kwargs["proxies"] = proxies
    if hours_old is not None:
        kwargs["hours_old"] = hours_old

    try:
        jobs_df = scrape_jobs(**kwargs)
    except Exception as exc:
        logger.error("Job search failed: %s", exc)
        return []

    jobs = jobs_df.to_dict("records") if hasattr(jobs_df, "to_dict") else []
    logger.info("Found %d jobs", len(jobs))
    return jobs


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for the CouchHire pipeline."""
    parser = argparse.ArgumentParser(
        description="CouchHire — agentic job application pipeline",
    )
    parser.add_argument("--jd", type=str, help="Job description text (inline)")
    parser.add_argument("--file", type=str, help="Path to a .txt file containing the JD")
    parser.add_argument("--url", type=str, help="URL to scrape the JD from")
    parser.add_argument("--search", action="store_true", help="Search for jobs")
    parser.add_argument("--query", type=str, help="Job search query (with --search)")
    parser.add_argument("--location", type=str, default="", help="Location (with --search)")

    args = parser.parse_args()

    # --- Search mode ---
    if args.search:
        if not args.query:
            logger.error("--search requires --query")
            sys.exit(1)

        jobs = _search_jobs(args.query, args.location)
        if not jobs:
            print("No jobs found.")
            sys.exit(0)

        print(f"\nFound {len(jobs)} jobs:\n")
        for i, job in enumerate(jobs, 1):
            title = job.get("title", "Unknown")
            company = job.get("company", "Unknown")
            loc = job.get("location", "")
            url = job.get("job_url", "")
            print(f"  {i}. {title} at {company} ({loc})")
            if url:
                print(f"     {url}")

        # Let user pick a job
        try:
            choice = input(f"\nPick a job (1-{len(jobs)}) or 'q' to quit: ").strip()
            if choice.lower() == "q":
                sys.exit(0)
            idx = int(choice) - 1
            if idx < 0 or idx >= len(jobs):
                logger.error("Invalid choice")
                sys.exit(1)
        except (ValueError, EOFError):
            sys.exit(0)

        selected = jobs[idx]
        jd_url = selected.get("job_url", "")
        jd_text = selected.get("description", "")

        if not jd_text and jd_url:
            jd_text = _scrape_jd(jd_url)
        elif not jd_text:
            logger.error("No description available for selected job")
            sys.exit(1)

        final = run_pipeline(jd_text=jd_text, jd_url=jd_url)
        _print_summary(final)
        return

    # --- URL mode ---
    if args.url:
        jd_text = _scrape_jd(args.url)
        final = run_pipeline(jd_text=jd_text, jd_url=args.url)
        _print_summary(final)
        return

    # --- File mode ---
    if args.file:
        path = Path(args.file)
        if not path.exists():
            logger.error("File not found: %s", args.file)
            sys.exit(1)
        jd_text = path.read_text(encoding="utf-8")
        final = run_pipeline(jd_text=jd_text)
        _print_summary(final)
        return

    # --- Inline JD mode ---
    if args.jd:
        final = run_pipeline(jd_text=args.jd)
        _print_summary(final)
        return

    # --- No args: print help ---
    parser.print_help()
    sys.exit(1)


def _print_summary(state: PipelineState) -> None:
    """Print a human-readable summary of the pipeline result."""
    reqs = state.get("requirements", {})
    company = reqs.get("company", "Unknown")
    role = reqs.get("role", "Unknown")

    print("\n" + "=" * 60)

    if state.get("error"):
        print("❌ PIPELINE FAILED")
        print("=" * 60)
        print(f"  Error: {state['error']}")
    elif state.get("_approval") == "rejected" or state.get("match_score", 100) < config.MATCH_THRESHOLD:
        print("⏭️  APPLICATION SKIPPED")
        print("=" * 60)
        print(f"  Company : {company}")
        print(f"  Role    : {role}")
        print(f"  Score   : {state.get('match_score', 'N/A')}")
    else:
        print("✅ PIPELINE COMPLETE")
        print("=" * 60)
        print(f"  Company    : {company}")
        print(f"  Role       : {role}")
        print(f"  Score      : {state.get('match_score', 'N/A')}")
        print(f"  Route      : {state.get('route', 'N/A')}")
        print(f"  Resume     : {state.get('resume_pdf_path', 'N/A')}")
        print(f"  Draft URL  : {state.get('draft_url', 'N/A')}")
        if state.get("form_result"):
            fr = state["form_result"]
            print(f"  Form Status: {fr.get('status', 'N/A')}")

    print("=" * 60)


if __name__ == "__main__":
    main()
