"""Tailor the master CV to a specific job description and compile to PDF.

SELECT-ONLY approach: the LLM never generates LaTeX. It selects which
content to include and in what order from the master CV. All output is
assembled verbatim from master_cv.tex — zero hallucinated content.

Pipeline:
  1. Load template + instructions (uploads/ → defaults/ fallback)
  2. Parse master_cv.tex → raw LaTeX sections
  3. Build plain-text inventory → LLM selects content (JSON only)
  4. Assemble verbatim LaTeX blocks based on LLM's selection
  5. Inject into template %%INJECT:SECTION%% markers
  6. Compile PDF with pdflatex
  7. Page enforcement loop — trim lowest-priority projects until within limit
  8. Generate resume_content summary for cover_letter.py
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
from pathlib import Path

from llm.client import complete
from agents.resume_assembler import (
    get_page_count as _get_page_count,
    _fuzzy_find,
    _fuzzy_find_in_template,
    build_template_block_registry,
    extract_style_examples,
    detect_project_separator,
    needs_reformatting,
    reformat_to_template_style,
)

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────
_CV_DIR = Path(__file__).resolve().parent.parent / "cv"
_UPLOADS_DIR = _CV_DIR / "uploads"
_DEFAULTS_DIR = _CV_DIR / "defaults"
_OUTPUT_DIR = _CV_DIR / "output"

# ── Regex ──────────────────────────────────────────────────────────────
# Matches %%INJECT:SECTION%% ... %%END:SECTION%% blocks in the template
_INJECT_PATTERN = re.compile(
    r"%%INJECT:(?P<section>[A-Z_]+)%%\n(?P<default>.*?)%%END:(?P=section)%%",
    re.DOTALL,
)

# Detects missing LaTeX packages from pdflatex output
_MISSING_PKG_PATTERN = re.compile(
    r"! LaTeX Error: File `(?P<package>[^']+)\.sty' not found\.",
)


# ═══════════════════════════════════════════════════════════════════════
# Step 1 — Load template and instructions from disk
# ═══════════════════════════════════════════════════════════════════════

def _load_template() -> str:
    """Load resume template — uploads/ first, defaults/ fallback."""
    uploads_path = _UPLOADS_DIR / "resume_template.tex"
    defaults_path = _DEFAULTS_DIR / "resume_template.tex"

    if uploads_path.exists():
        logger.info("Using custom template: %s", uploads_path)
        return uploads_path.read_text(encoding="utf-8")
    elif defaults_path.exists():
        logger.info("Using default template: %s", defaults_path)
        return defaults_path.read_text(encoding="utf-8")
    else:
        raise FileNotFoundError(
            "No resume template found. Place resume_template.tex in "
            f"{_UPLOADS_DIR} or {_DEFAULTS_DIR}"
        )


def _load_instructions() -> str:
    """Load tailoring instructions — uploads/ first, defaults/ fallback."""
    uploads_path = _UPLOADS_DIR / "instructions.md"
    defaults_path = _DEFAULTS_DIR / "instructions.md"

    if uploads_path.exists():
        logger.info("Using custom instructions: %s", uploads_path)
        return uploads_path.read_text(encoding="utf-8")
    elif defaults_path.exists():
        logger.info("Using default instructions: %s", defaults_path)
        return defaults_path.read_text(encoding="utf-8")
    else:
        logger.warning("No instructions file found — using empty instructions")
        return ""


def _parse_page_limit(instructions: str) -> int:
    """Extract page limit from instructions text. Defaults to 1 if not found."""
    # Match patterns like "1 page", "2 pages", "max 1 page", "Keep resume to 1 page"
    match = re.search(
        r"(?:keep\s+(?:resume\s+)?to\s+|max(?:imum)?\s+|limit\s+(?:to\s+)?)?(\d+)\s+page",
        instructions,
        re.IGNORECASE,
    )
    if match:
        limit = int(match.group(1))
        logger.info("Page limit from instructions: %d", limit)
        return limit
    logger.info("No page limit found in instructions — defaulting to 1")
    return 1


def _get_template_markers(template: str) -> list[str]:
    """Return list of section marker names found in the template."""
    return [m.group("section") for m in _INJECT_PATTERN.finditer(template)]


# ═══════════════════════════════════════════════════════════════════════
# Step 4 — Assemble verbatim LaTeX from LLM's selection JSON
# ═══════════════════════════════════════════════════════════════════════

def _assemble_section_content(
    marker: str,
    selection: dict,
    raw_sections: dict[str, str],
    block_registry: dict[str, dict[str, str]],
    template_blocks: dict[str, dict[str, str]] | None = None,
    template_style_examples: dict[str, str] | None = None,
    project_separator: str | None = None,
) -> str:
    """Assemble the LaTeX content for a single template marker.

    Uses 3-tier content resolution:
      TIER 1: Use template version verbatim (if item exists in template)
      TIER 2: Reformat master_cv content to match template style (LLM call)
      TIER 3: Use master_cv content as-is (if template has no examples)

    Args:
        marker: Template marker name (e.g. "PROJECTS", "EXPERIENCE").
        selection: The LLM's selection JSON.
        raw_sections: Parsed master_cv sections {name: raw_latex}.
        block_registry: Named sub-blocks from master_cv per section
                        {"PROJECTS": {"CouchHire": "<latex>", ...}, ...}.
        template_blocks: Named sub-blocks from the template per section
                         (parallel to block_registry). None if not available.
        template_style_examples: Full style examples per marker from template
                                 %%INJECT%% blocks. Used for reformatting.
        project_separator: Separator string between project blocks. Detected
                          from template. Falls back to "\n\n" if None.

    Returns:
        Raw LaTeX string to inject for this marker, or empty string if
        the section is excluded or has no content.
    """
    from agents.cv_content_helpers import SECTION_TO_MARKER, MARKER_TO_SECTION

    if template_blocks is None:
        template_blocks = {}
    if template_style_examples is None:
        template_style_examples = {}

    sections_to_include = selection.get("sections_to_include", {})
    value = sections_to_include.get(marker)

    # Section excluded by LLM — return "" to signal explicit removal
    if not value:
        logger.info("Section %s excluded by LLM selection", marker)
        return ""

    # HEADER is always from the template default — never from master_cv
    if marker == "HEADER":
        return None  # Sentinel: keep template default

    # Map marker back to master_cv section name
    master_section = MARKER_TO_SECTION.get(marker)
    if not master_section:
        logger.warning(
            "No master_cv section mapped to marker '%s' — keeping template default",
            marker,
        )
        return None  # Keep template default

    raw_content = raw_sections.get(master_section, "")
    if not raw_content:
        logger.warning(
            "Master CV has no content for section '%s' (marker: %s) — keeping template default",
            master_section, marker,
        )
        return None  # Keep template default

    # SKILLS with skill_category_order: reorder categories
    if marker == "SKILLS" and value is True:
        skill_order = selection.get("skill_category_order", [])
        if skill_order:
            from agents.cv_content_helpers import extract_skill_categories
            cat_blocks = extract_skill_categories(raw_content)
            if cat_blocks:
                ordered_parts: list[str] = []
                used_cats: set[str] = set()
                for cat_name in skill_order:
                    if cat_name in cat_blocks:
                        ordered_parts.append(cat_blocks[cat_name])
                        used_cats.add(cat_name)
                    else:
                        matched = _fuzzy_find(cat_name, cat_blocks)
                        if matched:
                            ordered_parts.append(cat_blocks[matched])
                            used_cats.add(matched)
                        else:
                            logger.warning(
                                "Skill category '%s' not found — skipping",
                                cat_name,
                            )
                # Append remaining categories not in the order list
                for cat_name, block in cat_blocks.items():
                    if cat_name not in used_cats:
                        ordered_parts.append(block)
                if ordered_parts:
                    return "\n".join(ordered_parts)
        return raw_content

    # For sections with value=True, apply Tier 1 principle:
    # prefer template content if available, fall back to master_cv.
    # This ensures template-curated content (e.g. Education without high
    # school) is preserved rather than being overwritten by verbose master_cv.
    if value is True:
        tmpl_content = template_style_examples.get(marker, "").strip()
        if tmpl_content:
            logger.info(
                "TIER 1 [%s]: using template version (value=True, template has content)",
                marker,
            )
            return tmpl_content
        return raw_content

    # For list-based selections (PROJECTS, EXPERIENCE, CERTIFICATIONS, LEADERSHIP),
    # use 3-tier content resolution: template → reformat → master_cv fallback
    if isinstance(value, list) and marker in block_registry:
        master_blocks = block_registry[marker]
        tmpl_blocks = template_blocks.get(marker, {})
        style_example = template_style_examples.get(marker, "")
        ordered_content: list[str] = []

        # Determine the order — use project_order/experience_order if available
        order_key = {
            "PROJECTS": "project_order",
            "EXPERIENCE": "experience_order",
            "CERTIFICATIONS": "certification_order",
            "LEADERSHIP": "leadership_order",
        }.get(marker)

        ordered_names = selection.get(order_key, value) if order_key else value

        for name in ordered_names:
            block = _resolve_block(
                name=name,
                marker=marker,
                master_blocks=master_blocks,
                tmpl_blocks=tmpl_blocks,
                style_example=style_example,
            )
            if block is not None:
                ordered_content.append(block)

        if not ordered_content:
            logger.warning(
                "No valid blocks found for %s — falling back to full section",
                marker,
            )
            return raw_content

        # Join blocks with appropriate spacing
        if marker == "PROJECTS":
            sep = project_separator if project_separator else "\n\n"
            return sep.join(ordered_content)
        else:
            return "\n\n".join(ordered_content)

    # Fallback: if value is a list but no block registry, inject full section
    return raw_content


def _resolve_block(
    name: str,
    marker: str,
    master_blocks: dict[str, str],
    tmpl_blocks: dict[str, str],
    style_example: str,
) -> str | None:
    """Resolve a single named block using 3-tier content resolution.

    TIER 1: Template match — use template version verbatim.
    TIER 2: Reformat — master_cv content reformatted to template style.
    TIER 3: Raw fallback — master_cv content as-is.

    Returns the LaTeX block string, or None if not found anywhere.
    """
    # TIER 1: Check template blocks for a matching name
    if tmpl_blocks:
        tmpl_key = _fuzzy_find_in_template(name, tmpl_blocks)
        if tmpl_key:
            logger.info(
                "TIER 1 [%s]: using template version for '%s' (matched '%s')",
                marker, name, tmpl_key,
            )
            return tmpl_blocks[tmpl_key]

    # Find the master_cv block
    master_block = None
    if name in master_blocks:
        master_block = master_blocks[name]
    else:
        matched_key = _fuzzy_find(name, master_blocks)
        if matched_key:
            master_block = master_blocks[matched_key]

    if master_block is None:
        logger.warning(
            "Block '%s' not found in %s (template or master_cv) — skipping",
            name, marker,
        )
        return None

    # TIER 2: Reformat if template has style examples and styles differ
    if style_example and needs_reformatting(master_block, style_example):
        logger.info(
            "TIER 2 [%s]: reformatting '%s' from master_cv to template style",
            marker, name,
        )
        try:
            reformatted = reformat_to_template_style(
                master_block, style_example, marker,
            )
            return reformatted
        except (ValueError, Exception) as exc:
            logger.warning(
                "TIER 2 [%s]: reformat failed for '%s': %s — falling back to TIER 3",
                marker, name, exc,
            )
            # Fall through to TIER 3

    # TIER 3: Use master_cv content as-is
    logger.info(
        "TIER 3 [%s]: using master_cv version for '%s' (no reformat needed)",
        marker, name,
    )
    return master_block


def _inject_into_template(
    template: str,
    assembled: dict[str, str],
) -> str:
    """Replace %%INJECT:SECTION%%...%%END:SECTION%% blocks in the template.

    For each marker:
    - If assembled[marker] is a non-empty string: replace with that content.
    - If assembled[marker] is None or marker not in assembled: keep template default.
    - If assembled[marker] is empty string "": section was explicitly excluded —
      remove the %%INJECT%%...%%END%% block AND the preceding \\section{} line.
    """
    def _replacer(match: re.Match) -> str:
        section = match.group("section")
        content = assembled.get(section)

        if content is None:
            # Not in assembled dict — keep template default
            return match.group(0)
        elif content == "":
            # Explicitly excluded — return empty (will be cleaned up)
            return ""
        else:
            # Has content — inject it
            return f"%%INJECT:{section}%%\n{content}\n%%END:{section}%%"

    result = _INJECT_PATTERN.sub(_replacer, template)

    # Clean up orphaned \section{} lines that precede now-empty blocks.
    # Pattern: \section{...}\n\n  (with nothing after until the next \section or \end)
    result = re.sub(
        r"\\section\*?\{[^}]*\}\s*\n\s*\n(?=\\section|\\end\{document\}|%---|$)",
        "",
        result,
    )
    # Also clean up any double-blank-line artifacts
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result


# ═══════════════════════════════════════════════════════════════════════
# Step 5 — Compile PDF with pdflatex
# ═══════════════════════════════════════════════════════════════════════

def _install_missing_packages(missing: list[str]) -> bool:
    """Attempt to install missing LaTeX packages via tlmgr.

    Returns True if installation succeeded, False otherwise.
    """
    logger.warning(
        "Missing LaTeX packages detected: %s — installing via tlmgr",
        ", ".join(missing),
    )

    tlmgr = shutil.which("tlmgr")
    if tlmgr is None:
        logger.error("tlmgr not found on PATH — cannot auto-install packages")
        return False

    cmd = ["sudo", tlmgr, "install"] + missing
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            logger.info("Packages installed successfully: %s", ", ".join(missing))
            return True
        else:
            logger.error(
                "tlmgr install failed (exit code %d).\nSTDOUT: %s\nSTDERR: %s",
                result.returncode,
                result.stdout[-500:] if result.stdout else "(empty)",
                result.stderr[-500:] if result.stderr else "(empty)",
            )
            return False
    except subprocess.TimeoutExpired:
        logger.error("tlmgr install timed out after 300 seconds")
        return False
    except OSError as exc:
        logger.error("Failed to run tlmgr: %s", exc)
        return False


def _run_pdflatex(cmd: list[str], tex_name: str) -> str:
    """Run pdflatex twice (for cross-refs) and return the last stdout.

    Logs warnings on non-zero exit codes but does not raise — the caller
    checks for the PDF file to determine success.
    """
    last_stdout = ""
    for run_num in (1, 2):
        logger.info("pdflatex run %d/2: %s", run_num, tex_name)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        last_stdout = result.stdout or ""

        if result.returncode != 0:
            logger.warning(
                "pdflatex run %d exited with code %d (may be non-fatal).\n"
                "--- STDOUT (last 1500 chars) ---\n%s\n--- STDERR ---\n%s",
                run_num,
                result.returncode,
                last_stdout[-1500:] if last_stdout else "(empty)",
                result.stderr[-500:] if result.stderr else "(empty)",
            )
    return last_stdout


def _compile_pdf(tex_path: Path) -> Path:
    """Compile a .tex file to PDF using pdflatex (run twice for cross-refs).

    If pdflatex fails due to missing .sty packages, attempts to auto-install
    them via tlmgr and retries compilation.

    Returns:
        Absolute path to the compiled .pdf file.

    Raises:
        RuntimeError: If pdflatex is not found, compilation fails, or
                      auto-install of missing packages fails.
    """
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        raise RuntimeError(
            "pdflatex not found on PATH. Install TeX Live:\n"
            "  macOS:  brew install --cask mactex\n"
            "  Ubuntu: sudo apt install texlive-latex-full\n"
            "  Docker: see Dockerfile (texlive-latex-full)"
        )

    output_dir = tex_path.parent
    cmd = [
        pdflatex,
        "-interaction=nonstopmode",
        f"-output-directory={output_dir}",
        str(tex_path),
    ]

    # First compilation attempt
    last_stdout = _run_pdflatex(cmd, tex_path.name)

    pdf_path = tex_path.with_suffix(".pdf")

    # If PDF was not produced, check for missing packages
    if not pdf_path.exists():
        missing = list(
            {m.group("package") for m in _MISSING_PKG_PATTERN.finditer(last_stdout)}
        )

        if missing:
            # Attempt auto-install and retry
            if _install_missing_packages(missing):
                logger.info("Retrying pdflatex after installing packages...")
                last_stdout = _run_pdflatex(cmd, tex_path.name)

                if not pdf_path.exists():
                    raise RuntimeError(
                        f"pdflatex still failed after installing packages: "
                        f"{', '.join(missing)}\n"
                        f"Debug the .tex file at: {tex_path}\n"
                        f"pdflatex output (last 2000 chars):\n"
                        f"{last_stdout[-2000:] if last_stdout else '(empty)'}"
                    )
            else:
                raise RuntimeError(
                    f"Missing LaTeX packages: {', '.join(missing)}\n"
                    f"Auto-install failed. Run manually:\n"
                    f"  sudo tlmgr install {' '.join(missing)}\n"
                    f"Then re-run the pipeline."
                )
        else:
            # No missing packages — generic compilation failure
            raise RuntimeError(
                f"pdflatex failed to produce a PDF. "
                f"Debug the .tex file at: {tex_path}\n"
                f"pdflatex output (last 2000 chars):\n"
                f"{last_stdout[-2000:] if last_stdout else '(empty)'}"
            )

    # Clean up auxiliary files
    for ext in (".aux", ".log", ".out"):
        aux_file = tex_path.with_suffix(ext)
        if aux_file.exists():
            aux_file.unlink()
            logger.debug("Cleaned up: %s", aux_file.name)

    logger.info("PDF compiled: %s", pdf_path)
    return pdf_path





# ═══════════════════════════════════════════════════════════════════════
# Step 6 — Page enforcement loop (instruction-aware)
# ═══════════════════════════════════════════════════════════════════════

# Sections that can be trimmed to fit page limit, in priority order
# (lowest priority = removed first). HEADER, EDUCATION, SKILLS are
# never removed — they're required.
_TRIMMABLE_SECTIONS_PRIORITY = [
    # First: trim projects from the bottom of project_order
    "PROJECTS",
    # Then: remove optional sections entirely
    "CERTIFICATIONS",
    "LEADERSHIP",
    # Last resort: trim experience entries
    "EXPERIENCE",
]


def _compile_and_count(
    template: str,
    assembled: dict[str, str],
    tex_path: Path,
) -> tuple[Path, int]:
    """Write assembled template to disk, compile PDF, return (pdf_path, page_count)."""
    tex_path.write_text(
        _inject_into_template(template, assembled),
        encoding="utf-8",
    )
    pdf_path = _compile_pdf(tex_path)
    pages = _get_page_count(pdf_path)
    return pdf_path, pages


def _enforce_page_limit(
    template: str,
    assembled: dict[str, str],
    selection: dict,
    block_registry: dict[str, dict[str, str]],
    raw_sections: dict[str, str],
    page_limit: int,
    tex_path: Path,
    template_blocks: dict[str, dict[str, str]] | None = None,
    template_style_examples: dict[str, str] | None = None,
    project_separator: str | None = None,
) -> Path:
    """Compile and enforce page limit by progressively trimming content.

    Trimming strategy (in order):
      1. Remove lowest-priority projects (last in project_order) one by one
      2. Remove CERTIFICATIONS section entirely
      3. Remove LEADERSHIP section entirely
      4. Remove EXPERIENCE entries (last first)

    Stops as soon as the PDF fits within page_limit.

    Returns:
        Path to the final PDF that meets the page constraint.
    """
    # Initial compile
    pdf_path, pages = _compile_and_count(template, assembled, tex_path)

    if pages <= page_limit:
        logger.info("PDF is %d page(s) — within %d-page limit ✓", pages, page_limit)
        return pdf_path

    logger.info(
        "PDF is %d page(s) — exceeds %d-page limit. Starting enforcement loop.",
        pages, page_limit,
    )

    sections_to_include = selection.get("sections_to_include", {})
    iteration = 0
    max_iterations = 20  # Safety valve

    # ── Phase 1: Trim projects from bottom of project_order ──
    project_order = list(selection.get("project_order", []))

    while pages > page_limit and len(project_order) > 1 and iteration < max_iterations:
        iteration += 1
        removed = project_order.pop()
        logger.info(
            "Enforcement [iter %d]: removing project '%s' "
            "(pages: %d, limit: %d, projects left: %d)",
            iteration, removed, pages, page_limit, len(project_order),
        )

        sections_to_include["PROJECTS"] = list(project_order)
        selection["project_order"] = list(project_order)
        assembled["PROJECTS"] = _assemble_section_content(
            "PROJECTS", selection, raw_sections, block_registry,
            template_blocks=template_blocks,
            template_style_examples=template_style_examples,
            project_separator=project_separator,
        )
        pdf_path, pages = _compile_and_count(template, assembled, tex_path)

    if pages <= page_limit:
        logger.info("Page enforcement succeeded after trimming projects ✓")
        return pdf_path

    # ── Phase 2: Remove optional sections entirely ──
    optional_sections = ["CERTIFICATIONS", "LEADERSHIP"]

    for section in optional_sections:
        if pages <= page_limit:
            break
        if not sections_to_include.get(section):
            continue  # Already excluded

        iteration += 1
        logger.info(
            "Enforcement [iter %d]: removing section '%s' "
            "(pages: %d, limit: %d)",
            iteration, section, pages, page_limit,
        )

        sections_to_include[section] = False
        assembled[section] = ""
        pdf_path, pages = _compile_and_count(template, assembled, tex_path)

    if pages <= page_limit:
        logger.info("Page enforcement succeeded after removing optional sections ✓")
        return pdf_path

    # ── Phase 3: Trim experience entries ──
    experience_raw = sections_to_include.get("EXPERIENCE", [])
    if isinstance(experience_raw, bool):
        # If True, convert to list of all experience names
        experience_entries = list(block_registry.get("EXPERIENCE", {}).keys())
    elif isinstance(experience_raw, list):
        experience_entries = list(experience_raw)
    else:
        experience_entries = []

    while (
        pages > page_limit
        and len(experience_entries) > 0
        and iteration < max_iterations
    ):
        iteration += 1
        removed = experience_entries.pop()
        logger.info(
            "Enforcement [iter %d]: removing experience '%s' "
            "(pages: %d, limit: %d)",
            iteration, removed, pages, page_limit,
        )

        if experience_entries:
            sections_to_include["EXPERIENCE"] = list(experience_entries)
            assembled["EXPERIENCE"] = _assemble_section_content(
                "EXPERIENCE", selection, raw_sections, block_registry,
                template_blocks=template_blocks,
                template_style_examples=template_style_examples,
            )
        else:
            # All experience removed — clear the section
            sections_to_include["EXPERIENCE"] = False
            assembled["EXPERIENCE"] = ""

        pdf_path, pages = _compile_and_count(template, assembled, tex_path)

    if pages <= page_limit:
        logger.info("Page enforcement succeeded after trimming experience ✓")
    else:
        logger.warning(
            "Could not reduce PDF to %d page(s) after all trimming — "
            "final is %d page(s). Remaining content may be irreducible "
            "(header + education + skills + 1 project).",
            page_limit, pages,
        )

    return pdf_path



# ═══════════════════════════════════════════════════════════════════════
# Step 7 — Generate resume_content summary (for cover_letter.py)
# ═══════════════════════════════════════════════════════════════════════

def _generate_resume_content_summary(
    selection: dict,
    requirements: dict,
    instructions: str,
) -> str:
    """Generate a structured plain-text summary of the tailored resume.

    This is a small LLM call that reads the selection JSON (what was
    included/excluded and in what order) and produces a summary for
    cover_letter.py to consume. It does NOT generate any LaTeX.
    """
    sections = selection.get("sections_to_include", {})
    project_order = selection.get("project_order", [])
    experience_note = selection.get("experience_note", "")

    prompt = f"""You are summarising a tailored resume for a cover letter writer.

ROLE: {requirements.get('role', 'Unknown')}
COMPANY: {requirements.get('company', 'Unknown')}

WHAT WAS INCLUDED IN THE RESUME:
- Sections included: {', '.join(k for k, v in sections.items() if v)}
- Projects (in order): {', '.join(project_order) if project_order else 'All'}
- Experience selection note: {experience_note or 'None'}

TAILORING INSTRUCTIONS USED:
{instructions}

Write a structured bullet-point summary (5-8 bullets) covering:
1. Which projects were selected and why they're relevant
2. What skills/experience were highlighted
3. What narrative angle or positioning was chosen
4. What was deliberately omitted and why
5. What the cover letter should complement (not repeat)

Be specific — reference actual project names and skills. Keep each bullet to 1-2 sentences.
Output ONLY the bullet points, no preamble."""

    system_prompt = (
        "You are a resume analysis assistant. Summarise what a tailored resume "
        "emphasises so a cover letter can complement it. Be concise and specific."
    )

    try:
        summary = complete(prompt, system_prompt=system_prompt)
        logger.info("Resume content summary generated (%d chars)", len(summary))
        return summary
    except Exception as exc:
        logger.error("Failed to generate resume content summary: %s", exc)
        # Fallback: build a basic summary from the selection data
        lines = [
            f"- Resume tailored for {requirements.get('role', 'the role')} at {requirements.get('company', 'the company')}",
            f"- Projects included (in order): {', '.join(project_order) if project_order else 'all from master CV'}",
            f"- Experience note: {experience_note or 'all experience included'}",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Public API — tailor()
# ═══════════════════════════════════════════════════════════════════════

def tailor(cv_sections: list[str], requirements: dict) -> tuple[str, str]:
    """Tailor the master CV for a job and compile to PDF.

    SELECT-ONLY approach: the LLM picks content from master_cv.tex.
    All LaTeX in the output exists verbatim in the source files.

    Args:
        cv_sections: Retrieved CV sections from ChromaDB (used for context
                     but the actual LaTeX comes from master_cv.tex directly).
        requirements: Parsed job requirements dict from jd_parser.

    Returns:
        (pdf_path, resume_content) — path to compiled PDF and a structured
        summary of what the resume emphasises.
    """
    from agents.cv_content_helpers import (
        parse_master_cv,
        extract_item_blocks,
        extract_experience_blocks,
        extract_certification_blocks,
        build_content_inventory,
        format_inventory_for_llm,
        SECTION_TO_MARKER,
    )
    from agents.llm_selector import select_content

    start_time = time.time()

    # ── Step 1: Load template and instructions from disk ──
    template = _load_template()
    instructions = _load_instructions()
    page_limit = _parse_page_limit(instructions)
    template_markers = _get_template_markers(template)

    logger.info("Template markers: %s", template_markers)
    logger.info("Page limit: %d", page_limit)

    # ── Step 2: Parse master_cv.tex into raw LaTeX sections ──
    raw_sections = parse_master_cv()
    logger.info("Parsed %d sections from master_cv.tex", len(raw_sections))

    # ── Step 3: Build content inventory and get LLM selection ──
    inventory = build_content_inventory(raw_sections)
    inventory_text = format_inventory_for_llm(inventory)

    selection = select_content(
        inventory=inventory,
        inventory_text=inventory_text,
        requirements=requirements,
        instructions=instructions,
        template_sections=template_markers,
    )

    logger.info("LLM selection: %s", selection)

    # ── Step 4: Build block registries for sub-item sections ──
    block_registry: dict[str, dict[str, str]] = {}

    # Projects
    if "Projects" in raw_sections:
        block_registry["PROJECTS"] = extract_item_blocks(
            raw_sections["Projects"]
        )
        logger.info(
            "Project blocks: %s",
            list(block_registry["PROJECTS"].keys()),
        )

    # Experience
    if "Experience" in raw_sections:
        block_registry["EXPERIENCE"] = extract_experience_blocks(
            raw_sections["Experience"]
        )
        logger.info(
            "Experience blocks: %s",
            list(block_registry["EXPERIENCE"].keys()),
        )

    # Certifications
    if "Certifications" in raw_sections:
        block_registry["CERTIFICATIONS"] = extract_certification_blocks(
            raw_sections["Certifications"]
        )
        logger.info(
            "Certification blocks: %s",
            list(block_registry["CERTIFICATIONS"].keys()),
        )

    # Leadership / Extra Curriculars
    ec_key = "Extra Curriculars"
    if ec_key in raw_sections:
        block_registry["LEADERSHIP"] = extract_item_blocks(
            raw_sections[ec_key]
        )
        logger.info(
            "Leadership blocks: %s",
            list(block_registry["LEADERSHIP"].keys()),
        )

    # —— Step 4b: Build template block registry (for 3-tier resolution) ——
    template_block_registry = build_template_block_registry(template)
    template_style_examples = extract_style_examples(template)

    # Detect project separator from template
    project_separator = None
    if "PROJECTS" in template_style_examples and template_style_examples["PROJECTS"]:
        project_separator = detect_project_separator(
            template_style_examples["PROJECTS"]
        )

    logger.info(
        "Template block registry: %s",
        {k: list(v.keys()) for k, v in template_block_registry.items()},
    )

    # ── Step 5: Assemble LaTeX for each template marker (3-tier resolution) ──
    assembled: dict[str, str] = {}
    for marker in template_markers:
        assembled[marker] = _assemble_section_content(
            marker, selection, raw_sections, block_registry,
            template_blocks=template_block_registry,
            template_style_examples=template_style_examples,
            project_separator=project_separator,
        )

    # ── Step 6: Write .tex, compile PDF, enforce page limit ──
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    company = requirements.get("company", "company").replace(" ", "_")
    role = requirements.get("role", "role").replace(" ", "_")
    timestamp = int(time.time())
    tex_name = f"resume_{company}_{role}_{timestamp}.tex"
    tex_path = _OUTPUT_DIR / tex_name

    pdf_path = _enforce_page_limit(
        template=template,
        assembled=assembled,
        selection=selection,
        block_registry=block_registry,
        raw_sections=raw_sections,
        page_limit=page_limit,
        tex_path=tex_path,
        template_blocks=template_block_registry,
        template_style_examples=template_style_examples,
        project_separator=project_separator,
    )

    # ── Step 7: Generate resume_content summary ──
    resume_content = _generate_resume_content_summary(
        selection, requirements, instructions,
    )

    elapsed = time.time() - start_time
    logger.info(
        "Resume tailored in %.1fs — PDF: %s (%d pages)",
        elapsed, pdf_path, _get_page_count(pdf_path),
    )

    return str(pdf_path), resume_content
