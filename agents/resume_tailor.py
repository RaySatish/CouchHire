"""Tailor the master CV to a specific job description and compile to PDF.

Retrieves the resume template and tailoring instructions from ChromaDB,
uses the LLM to generate tailored LaTeX for each %%INJECT:<SECTION>%%
marker, compiles the result with pdflatex, and returns the PDF path
alongside a structured summary of what the resume emphasises.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
from pathlib import Path

from config import CHROMA_STORE_DIR

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "master_cv"

# Regex to find %%INJECT:<SECTION>%% ... %%END:<SECTION>%% blocks
_INJECT_PATTERN = re.compile(
    r"%%INJECT:(?P<section>[A-Z_]+)%%\n(?P<default>.*?)%%END:(?P=section)%%",
    re.DOTALL,
)

# ── Output directory ──────────────────────────────────────────────────────
_CV_DIR = Path(__file__).resolve().parent.parent / "cv"
_OUTPUT_DIR = _CV_DIR / "output"

# ── Lazy ChromaDB singleton ──────────────────────────────────────────────
_chroma_collection = None


def _get_collection():
    """Lazily initialise ChromaDB client and return the master_cv collection."""
    global _chroma_collection
    if _chroma_collection is None:
        import chromadb

        chroma_path = Path(CHROMA_STORE_DIR)
        if not chroma_path.exists():
            raise RuntimeError(
                f"ChromaDB store not found at {chroma_path}. "
                "Run 'python cv/embed_cv.py' first to embed your CV."
            )

        client = chromadb.PersistentClient(path=str(chroma_path))
        _chroma_collection = client.get_collection(name=_COLLECTION_NAME)
        logger.info(
            "Connected to ChromaDB collection '%s' (%d documents)",
            _COLLECTION_NAME,
            _chroma_collection.count(),
        )
    return _chroma_collection


def _retrieve_by_type(chunk_type: str) -> str:
    """Retrieve a special chunk from ChromaDB by its type metadata.

    Args:
        chunk_type: 'template' or 'instructions'.

    Returns:
        The document text of the matching chunk.

    Raises:
        RuntimeError: If no chunk with the given type is found.
    """
    collection = _get_collection()

    results = collection.get(
        where={"type": chunk_type},
        include=["documents"],
    )

    documents = results.get("documents", [])
    if not documents:
        raise RuntimeError(
            f"No '{chunk_type}' chunk found in ChromaDB. "
            "Run 'python cv/embed_cv.py' to re-embed your CV with "
            "template and instructions."
        )

    logger.info("Retrieved '%s' chunk (%d chars)", chunk_type, len(documents[0]))
    return documents[0]


def _extract_sections(template: str) -> list[str]:
    """Extract section names from %%INJECT:<SECTION>%% markers in the template.

    Returns:
        List of section names (e.g. ['HEADER', 'EDUCATION', 'EXPERIENCE', ...]).
    """
    sections = [m.group("section") for m in _INJECT_PATTERN.finditer(template)]
    logger.info("Found %d injectable sections: %s", len(sections), sections)
    return sections


def _build_section_prompt(
    section_name: str,
    cv_sections: list[str],
    requirements: dict,
    instructions: str,
    default_content: str,
) -> str:
    """Build the LLM prompt for generating tailored LaTeX for one section."""
    role = requirements.get("role", "the role")
    company = requirements.get("company", "the company")
    skills = requirements.get("skills", [])

    return f"""You are a professional resume writer generating LaTeX content for a tailored resume.

TARGET ROLE: {role} at {company}
KEY SKILLS REQUIRED: {', '.join(skills) if skills else 'Not specified'}

TAILORING INSTRUCTIONS:
{instructions}

CANDIDATE'S CV DATA (relevant sections):
{chr(10).join(cv_sections)}

SECTION TO GENERATE: {section_name}

DEFAULT TEMPLATE CONTENT FOR THIS SECTION (use as a formatting reference only — do NOT copy placeholder names or data):
{default_content}

RULES:
- Output ONLY valid LaTeX content for the {section_name} section
- NO markdown, NO code fences, NO commentary, NO explanations
- Match the LaTeX formatting style of the default content (same commands, same structure)
- Be factual, achievement-focused, and keyword-rich — the cover letter handles narrative
- Quantify achievements wherever the CV data supports it
- Prioritise skills and experience most relevant to {role} at {company}
- Keep content concise — this resume must fit on 1 page total
- Use the candidate's actual data from CV DATA above — never fabricate
- If the CV data has no content for this section, output a minimal placeholder using the default structure

Generate the LaTeX content for the {section_name} section now:"""


def _build_resume_content_prompt(
    sections_generated: dict[str, str],
    cv_sections: list[str],
    requirements: dict,
    detailed: bool,
) -> str:
    """Build the LLM prompt for generating the structured resume_content summary."""
    role = requirements.get("role", "the role")
    company = requirements.get("company", "the company")

    detail_instruction = (
        "Be FULLY DETAILED — the cover letter agent will use this to complement the resume."
        if detailed
        else "Provide a brief summary — this will not be consumed downstream."
    )

    sections_text = "\n\n".join(
        f"--- {name} ---\n{content}" for name, content in sections_generated.items()
    )

    return f"""Analyse the tailored resume content below and produce a structured summary.

TARGET ROLE: {role} at {company}

ORIGINAL CV DATA:
{chr(10).join(cv_sections)}

TAILORED RESUME SECTIONS:
{sections_text}

{detail_instruction}

Output EXACTLY in this bullet-point format (no other text):
- Led with: <project or experience name> (<key technologies>, <metric if any>)
- Highlighted skills: <comma-separated list of skills emphasised>
- Included: <other projects/experiences that were included>
- Omitted: <what was left out from original CV and brief reason why>
- Foregrounded: <what angle/domain was emphasised, e.g. quantitative finance>
- Quantified achievements: <N> out of <total> bullets have metrics"""


def _fill_template(template: str, section_contents: dict[str, str]) -> str:
    """Replace %%INJECT:<SECTION>%% ... %%END:<SECTION>%% blocks with generated content."""

    def _replacer(match: re.Match) -> str:
        section_name = match.group("section")
        if section_name in section_contents:
            return section_contents[section_name]
        # If we didn't generate content for this section, keep the default
        return match.group("default")

    filled = _INJECT_PATTERN.sub(_replacer, template)
    return filled


def _clean_latex_content(content: str) -> str:
    """Post-process LLM-generated LaTeX to fix common issues.

    - Strips leading bare \\ (line breaks with nothing before them)
    - Removes completely empty lines that could cause paragraph breaks in bad spots
    """
    # Remove bare \\ at the very start (causes "There's no line here to end")
    content = re.sub(r"^\s*\\\\\s*\n", "", content)
    # Remove lines that are just \\ (bare line breaks with no content)
    content = re.sub(r"\n\s*\\\\\s*\n", "\n", content)
    return content.strip()


def _compile_pdf(tex_path: Path) -> Path:
    """Compile a .tex file to PDF using pdflatex (run twice for cross-refs).

    Args:
        tex_path: Absolute path to the .tex file.

    Returns:
        Absolute path to the compiled .pdf file.

    Raises:
        RuntimeError: If pdflatex is not found or compilation fails.
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

    # Run twice — standard practice to resolve cross-references.
    # pdflatex in nonstopmode often returns non-zero for warnings (e.g.
    # "There's no line here to end") yet still produces a valid PDF.
    # We log warnings but only raise if no PDF is produced.
    last_stdout = ""
    for run_num in (1, 2):
        logger.info("pdflatex run %d/2: %s", run_num, tex_path.name)
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

    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.exists():
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


def tailor(cv_sections: list[str], requirements: dict) -> tuple[str, str]:
    """Tailor the master CV to a specific job description and compile to PDF.

    Retrieves the resume template and tailoring instructions from ChromaDB,
    generates tailored LaTeX for each section using the LLM, compiles to PDF,
    and returns a structured summary of what the resume emphasises.

    Args:
        cv_sections: List of relevant CV section texts (from cv_rag).
        requirements: The requirements dict from jd_parser (must contain
                      'role', 'company', 'skills', 'cover_letter_required').

    Returns:
        A tuple of (resume_pdf_path, resume_content) where:
        - resume_pdf_path is the absolute path string to the compiled PDF
        - resume_content is a structured bullet-point summary for cover_letter.py
    """
    from llm.client import complete

    role = requirements.get("role", "unknown_role")
    company = requirements.get("company", "unknown_company")
    cover_letter_required = requirements.get("cover_letter_required", False)

    logger.info(
        "Tailoring resume for '%s' at '%s' (cover_letter_required=%s)",
        role,
        company,
        cover_letter_required,
    )

    # Step 1: Retrieve template and instructions from ChromaDB
    template = _retrieve_by_type("template")
    instructions = _retrieve_by_type("instructions")

    # Step 2: Extract injectable sections from template
    section_names = _extract_sections(template)
    if not section_names:
        raise RuntimeError(
            "No %%INJECT:<SECTION>%% markers found in the resume template. "
            "Check your resume_template.tex and re-run 'python cv/embed_cv.py'."
        )

    # Step 3: Generate tailored LaTeX for each section via LLM
    section_contents: dict[str, str] = {}
    section_defaults: dict[str, str] = {}

    # First, extract default content for each section (for reference in prompts)
    for match in _INJECT_PATTERN.finditer(template):
        section_defaults[match.group("section")] = match.group("default").strip()

    system_prompt = (
        "You are an expert resume writer. You generate precise, valid LaTeX content "
        "for resume sections. Never output markdown, code fences, or commentary. "
        "Output raw LaTeX only."
    )

    for section_name in section_names:
        default_content = section_defaults.get(section_name, "")
        prompt = _build_section_prompt(
            section_name=section_name,
            cv_sections=cv_sections,
            requirements=requirements,
            instructions=instructions,
            default_content=default_content,
        )

        logger.info("Generating LaTeX for section: %s", section_name)
        raw_response = complete(prompt, system_prompt=system_prompt)

        # Strip any accidental markdown fences the LLM might add
        content = raw_response.strip()
        content = re.sub(r"^```(?:latex|tex)?\s*\n?", "", content)
        content = re.sub(r"\n?```\s*$", "", content)
        content = content.strip()

        # Fix common LaTeX issues from LLM output
        content = _clean_latex_content(content)

        section_contents[section_name] = content
        logger.info(
            "Section '%s' generated (%d chars)", section_name, len(content)
        )

    # Step 4: Fill template with generated content
    filled_tex = _fill_template(template, section_contents)

    # Step 5: Write .tex file
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = str(int(time.time()))
    tex_filename = f"tailored_{timestamp}.tex"
    tex_path = _OUTPUT_DIR / tex_filename
    tex_path.write_text(filled_tex, encoding="utf-8")
    logger.info("Wrote tailored .tex: %s", tex_path)

    # Step 6: Compile to PDF
    pdf_path = _compile_pdf(tex_path)

    # Step 7: Generate resume_content summary
    resume_content_prompt = _build_resume_content_prompt(
        sections_generated=section_contents,
        cv_sections=cv_sections,
        requirements=requirements,
        detailed=cover_letter_required,
    )

    resume_content = complete(
        resume_content_prompt,
        system_prompt=(
            "You are analysing a tailored resume. Output ONLY the structured "
            "bullet-point summary in the exact format requested. No other text."
        ),
    ).strip()

    logger.info(
        "Resume tailoring complete — PDF: %s, resume_content: %d chars",
        pdf_path,
        len(resume_content),
    )

    return str(pdf_path), resume_content
