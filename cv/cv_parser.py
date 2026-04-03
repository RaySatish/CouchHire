"""Parses an uploaded master CV into named sections regardless of input format.

Supported formats: .tex (LaTeX), .pdf (via pdfplumber), .docx (via python-docx).
"""
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Common resume section names — used for PDF/DOCX heuristic splitting
_SECTION_NAMES = {
    "education", "experience", "work experience", "professional experience",
    "projects", "skills", "technical skills", "publications", "research",
    "certifications", "awards", "honors", "summary", "objective",
    "interests", "activities", "extracurricular", "volunteer",
    "references", "languages", "courses", "coursework",
}


def parse_cv(cv_path: Path) -> dict[str, str]:
    """Parse a CV file into named sections.

    Returns a dict mapping section names to their text content.
    Raises ValueError if the file format is unsupported.
    """
    cv_path = Path(cv_path)
    if not cv_path.exists():
        raise FileNotFoundError(f"CV file not found: {cv_path}")

    suffix = cv_path.suffix.lower()
    logger.info("Parsing CV: %s (format: %s)", cv_path.name, suffix)

    if suffix == ".tex":
        return _parse_tex(cv_path)
    elif suffix == ".pdf":
        return _parse_pdf(cv_path)
    elif suffix == ".docx":
        return _parse_docx(cv_path)
    else:
        raise ValueError(
            f"Unsupported CV format: {suffix}. "
            "Supported formats: .tex, .pdf, .docx"
        )


def _parse_tex(cv_path: Path) -> dict[str, str]:
    """Parse a LaTeX CV by section/subsection commands or explicit markers."""
    text = cv_path.read_text(encoding="utf-8")
    sections: dict[str, str] = {}

    # Pattern 1: explicit markers  %--- SECTION: <NAME> ---
    explicit_pattern = re.compile(r"^%---\s*SECTION:\s*(.+?)\s*---", re.MULTILINE)
    # Pattern 2: \section{} or \subsection{}
    latex_pattern = re.compile(r"\\(?:section|subsection)\{([^}]+)\}", re.MULTILINE)

    # Collect all split points with their positions
    markers: list[tuple[int, str]] = []

    for m in explicit_pattern.finditer(text):
        markers.append((m.start(), m.group(1).strip()))
    for m in latex_pattern.finditer(text):
        markers.append((m.start(), m.group(1).strip()))

    if not markers:
        logger.warning(
            "No section markers found in %s — returning full text as one section",
            cv_path.name,
        )
        return {"Full CV": text.strip()}

    # Sort by position
    markers.sort(key=lambda x: x[0])

    for i, (pos, name) in enumerate(markers):
        end = markers[i + 1][0] if i + 1 < len(markers) else len(text)
        chunk = text[pos:end]

        # Strip the marker line itself from the chunk
        first_newline = chunk.find("\n")
        if first_newline != -1:
            chunk = chunk[first_newline:].strip()
        else:
            chunk = chunk.strip()

        # Remove LaTeX noise but keep content readable
        chunk = _clean_latex(chunk)

        if chunk:
            sections[name] = chunk

    logger.info("Parsed %d sections from LaTeX: %s", len(sections), list(sections.keys()))
    return sections


def _clean_latex(text: str) -> str:
    """Light cleanup of LaTeX — remove commands but keep text content."""
    text = re.sub(r"\\(?:textbf|textit|emph|underline)\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:href)\{[^}]*\}\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:hfill|quad|qquad|\\\\|newline|vspace\{[^}]*\}|hspace\{[^}]*\})", " ", text)
    text = re.sub(r"\\item\s*", "- ", text)
    text = re.sub(r"\\begin\{[^}]*\}", "", text)
    text = re.sub(r"\\end\{[^}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"%.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
