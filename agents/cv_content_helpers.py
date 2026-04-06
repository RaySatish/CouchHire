r"""Helper functions for the SELECT-ONLY resume tailor pipeline.

Parses master_cv.tex into raw LaTeX sections, extracts individual
project/experience/certification blocks, and builds a plain-text
inventory for LLM selection. No LaTeX is ever sent to the LLM ---
only human-readable summaries.

Supports multiple LaTeX CV styles:
  - "standard"  --- \item \textbf{Name}, \textbf{Role} \hfill patterns
  - "jakes"     --- \resumeProjectHeading, \resumeSubheading, \resumeItem
  - "moderncv"  --- \cventry, \cvitem
  - "unknown"   --- fallback heuristics (whole-section blocks)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from cv.cv_parser import parse_cv

logger = logging.getLogger(__name__)

# -- Paths -----------------------------------------------------------------
_CV_DIR = Path(__file__).resolve().parent.parent / "cv"
_UPLOADS_DIR = _CV_DIR / "uploads"
_MASTER_CV_PATH = _UPLOADS_DIR / "master_cv.tex"


# -- Section name mapping: master_cv.tex -> template %%INJECT%% markers ----
# Keys are section names as returned by cv_parser._parse_tex()
# Values are template marker names (HEADER, EDUCATION, etc.)
SECTION_TO_MARKER: dict[str, str] = {
    "Education": "EDUCATION",
    "Experience": "EXPERIENCE",
    "Projects": "PROJECTS",
    "Skills": "SKILLS",
    "Certifications": "CERTIFICATIONS",
    "Extra Curriculars": "LEADERSHIP",
    # These master_cv sections have no direct template marker:
    # "Profile Summary" -> omitted (HEADER comes from template)
    # "Research Dissemination" -> can be folded into PROJECTS if selected
    # "Academic Achievements" -> can be folded into EDUCATION if selected
}

# Reverse mapping for convenience
MARKER_TO_SECTION: dict[str, str] = {v: k for k, v in SECTION_TO_MARKER.items()}


# ==========================================================================
# LaTeX style detection
# ==========================================================================

def detect_latex_style(latex_content: str) -> str:
    r"""Detect which LaTeX CV template style is being used.

    Returns one of: "standard" | "jakes" | "moderncv" | "unknown".
    """
    # Check for Jake's Resume template markers
    jakes_patterns = [
        r"\\resumeSubheading",
        r"\\resumeProjectHeading",
        r"\\resumeItem\b",
        r"\\resumeItemListStart",
        r"\\resumeSubHeadingListStart",
    ]
    jakes_hits = sum(
        1 for p in jakes_patterns if re.search(p, latex_content)
    )
    if jakes_hits >= 2:
        logger.info("Detected LaTeX style: jakes (%d markers found)", jakes_hits)
        return "jakes"

    # Check for moderncv markers
    moderncv_patterns = [
        r"\\cventry\b",
        r"\\cvitem\b",
        r"\\moderncvstyle\b",
        r"\\moderncvcolor\b",
    ]
    moderncv_hits = sum(
        1 for p in moderncv_patterns if re.search(p, latex_content)
    )
    if moderncv_hits >= 2:
        logger.info("Detected LaTeX style: moderncv (%d markers found)", moderncv_hits)
        return "moderncv"

    # Check for standard style: \item \textbf{} or \textbf{} \hfill
    standard_patterns = [
        r"\\item\s+\\textbf\{",
        r"\\textbf\{[^}]+\}\s*\\hfill",
    ]
    standard_hits = sum(
        1 for p in standard_patterns if re.search(p, latex_content)
    )
    if standard_hits >= 1:
        logger.info("Detected LaTeX style: standard (%d markers found)", standard_hits)
        return "standard"

    logger.info("Detected LaTeX style: unknown (no recognised patterns)")
    return "unknown"


# ==========================================================================
# Step 1 -- Parse master_cv.tex into raw LaTeX sections
# ==========================================================================

def parse_master_cv(cv_path: Path | None = None) -> dict[str, str]:
    r"""Parse master_cv.tex into named sections of raw LaTeX.

    Returns a dict like {"Education": "<raw latex>", "Projects": "<raw latex>", ...}.
    Uses cv_parser.parse_cv() under the hood.
    """
    path = cv_path or _MASTER_CV_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"master_cv.tex not found at {path}. "
            "Place your CV in cv/uploads/master_cv.tex"
        )
    sections = parse_cv(path)
    logger.info(
        "Parsed master_cv.tex: %d sections -- %s",
        len(sections),
        list(sections.keys()),
    )
    return sections


# ==========================================================================
# Step 2a -- Extract individual blocks from list-based sections
# ==========================================================================

def _extract_name_from_textbf(line: str) -> str:
    r"""Extract the name from a \textbf{Name} occurrence in a line."""
    match = re.search(r"\\textbf\{([^}]+)\}", line)
    if match:
        return match.group(1).strip()
    # Fallback: strip LaTeX commands, return first 80 chars
    cleaned = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", line)
    cleaned = re.sub(r"\\item\s*", "", cleaned).strip()
    return cleaned[:80] if cleaned else "(unnamed)"


def _extract_name_from_line(line: str) -> str:
    r"""Extract a meaningful name from any LaTeX line -- last resort."""
    # Try \textbf first
    match = re.search(r"\\textbf\{([^}]+)\}", line)
    if match:
        return match.group(1).strip()
    # Strip all LaTeX commands
    cleaned = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", line)
    cleaned = re.sub(r"\\[a-zA-Z]+\s*", "", cleaned)
    cleaned = re.sub(r"[{}$|]", "", cleaned).strip()
    return cleaned[:80] if cleaned else "(unnamed)"


def extract_item_blocks(section_latex: str) -> dict[str, str]:
    r"""Split a LaTeX section into individual blocks, indexed by name.

    Uses a multi-strategy approach to support different LaTeX CV styles:
      Strategy 1: \item \textbf{Name} -- standard style
      Strategy 2: \item \textbf{Name} $|$ -- common alternative
      Strategy 3: \resumeProjectHeading{\textbf{Name}} -- Jake's template
      Strategy 4: \cventry{...} -- moderncv style
      Strategy 5: \entry{...} -- other common styles
      Fallback: split on blank-line boundaries, use first 30 chars as key

    Returns {name: raw_latex_block}.
    """
    blocks: dict[str, str] = {}

    # -- Strategy 1 & 2: \item \textbf{Name} (with or without $|$) --
    item_textbf_positions: list[int] = []
    for match in re.finditer(r"\\item\s+\\textbf\{", section_latex):
        item_textbf_positions.append(match.start())

    if item_textbf_positions:
        logger.debug("extract_item_blocks: using strategy 1/2 (\\item \\textbf)")
        for i, start in enumerate(item_textbf_positions):
            end = (
                item_textbf_positions[i + 1]
                if i + 1 < len(item_textbf_positions)
                else len(section_latex)
            )
            block = section_latex[start:end].rstrip()
            first_line = block.split("\n")[0]
            name = _extract_name_from_textbf(first_line)
            blocks[name] = block
        return blocks

    # -- Strategy 3: \resumeProjectHeading{\textbf{Name} ...}{date} --
    # Also handles \resumeSubheading and similar Jake's macros
    resume_heading_positions: list[tuple[int, str]] = []
    for match in re.finditer(
        r"(\\resume(?:ProjectHeading|Subheading|SubItem)\{)", section_latex
    ):
        resume_heading_positions.append((match.start(), match.group(0)))

    if resume_heading_positions:
        logger.debug("extract_item_blocks: using strategy 3 (Jake's template)")
        for i, (start, _) in enumerate(resume_heading_positions):
            end = (
                resume_heading_positions[i + 1][0]
                if i + 1 < len(resume_heading_positions)
                else len(section_latex)
            )
            block = section_latex[start:end].rstrip()
            first_line = block.split("\n")[0]
            # Extract name from \textbf{Name} inside the heading macro
            name = _extract_name_from_line(first_line)
            blocks[name] = block
        return blocks

    # -- Strategy 4: \cventry{date}{role}{org}{location}{grade}{description} --
    cventry_positions: list[int] = []
    for match in re.finditer(r"\\cventry\b", section_latex):
        cventry_positions.append(match.start())

    if cventry_positions:
        logger.debug("extract_item_blocks: using strategy 4 (moderncv \\cventry)")
        for i, start in enumerate(cventry_positions):
            end = (
                cventry_positions[i + 1]
                if i + 1 < len(cventry_positions)
                else len(section_latex)
            )
            block = section_latex[start:end].rstrip()
            # cventry args: {date}{title}{org}{location}{grade}{desc}
            # Extract second arg as the name
            args = re.findall(r"\{([^}]*)\}", block)
            name = (
                args[1].strip()
                if len(args) >= 2
                else _extract_name_from_line(block.split("\n")[0])
            )
            blocks[name] = block
        return blocks

    # -- Strategy 5: \entry{...} -- other common styles --
    entry_positions: list[int] = []
    for match in re.finditer(r"\\entry\b", section_latex):
        entry_positions.append(match.start())

    if entry_positions:
        logger.debug("extract_item_blocks: using strategy 5 (\\entry)")
        for i, start in enumerate(entry_positions):
            end = (
                entry_positions[i + 1]
                if i + 1 < len(entry_positions)
                else len(section_latex)
            )
            block = section_latex[start:end].rstrip()
            args = re.findall(r"\{([^}]*)\}", block)
            name = (
                args[0].strip()
                if args
                else _extract_name_from_line(block.split("\n")[0])
            )
            blocks[name] = block
        return blocks

    # -- Fallback: plain \item splitting (no \textbf required) --
    item_positions: list[int] = []
    for match in re.finditer(r"\\item\s", section_latex):
        item_positions.append(match.start())

    if item_positions:
        logger.debug("extract_item_blocks: using fallback (plain \\item)")
        for i, start in enumerate(item_positions):
            end = (
                item_positions[i + 1]
                if i + 1 < len(item_positions)
                else len(section_latex)
            )
            block = section_latex[start:end].rstrip()
            first_line = block.split("\n")[0]
            name = _extract_name_from_line(first_line)
            blocks[name] = block
        return blocks

    # -- Ultimate fallback: split by blank lines, use first 30 chars as key --
    logger.debug("extract_item_blocks: using ultimate fallback (blank-line split)")
    paragraphs = re.split(r"\n\s*\n", section_latex.strip())
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Use first non-empty line, stripped of LaTeX, first 30 chars
        first_line = para.split("\n")[0]
        name = _extract_name_from_line(first_line)
        if not name or name == "(unnamed)":
            name = re.sub(r"\\[a-zA-Z]+[{}\[\]\s]*", "", first_line).strip()[:30]
            if not name:
                name = f"block_{len(blocks) + 1}"
        blocks[name] = para

    if not blocks:
        # Absolute last resort: return entire section as one block
        blocks["full_section"] = section_latex.strip()

    return blocks


def extract_experience_blocks(section_latex: str) -> dict[str, str]:
    r"""Split the Experience section into individual role blocks.

    Uses a multi-strategy approach:
      Strategy 1: \textbf{Role} \hfill dates -- standard style
      Strategy 2: \cventry{date}{role}{org}{...} -- moderncv
      Strategy 3: \resumeSubheading{role}{date}{org}{location} -- Jake's template
      Strategy 4: Lines with both \textbf{} and \hfill on the same line
      Fallback: return {"full_experience": entire_section_text}

    Returns {name: raw_latex_block}.
    """
    lines = section_latex.split("\n")
    blocks: dict[str, str] = {}

    # -- Strategy 1: \textbf{Role} \hfill -- standard style --
    entry_starts_s1: list[int] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("\\vspace") or stripped.startswith("\\begin") or stripped.startswith("\\end"):
            continue
        if re.match(r"\\textbf\{[^}]+\}\s*\\hfill", stripped):
            entry_starts_s1.append(i)

    if entry_starts_s1:
        logger.debug("extract_experience_blocks: using strategy 1 (\\textbf \\hfill)")
        for i, start_line in enumerate(entry_starts_s1):
            end_line = entry_starts_s1[i + 1] if i + 1 < len(entry_starts_s1) else len(lines)
            block_lines = lines[start_line:end_line]
            block = "\n".join(block_lines).rstrip()

            role_match = re.search(r"\\textbf\{([^}]+)\}", lines[start_line])
            role = role_match.group(1).strip() if role_match else "Unknown Role"

            org = ""
            if start_line + 1 < len(lines):
                org_match = re.search(r"\\textbf\{\\textit\{([^}]+)\}\}", lines[start_line + 1])
                if not org_match:
                    org_match = re.search(r"\\textit\{([^}]+)\}", lines[start_line + 1])
                if org_match:
                    org = org_match.group(1).strip()

            name = f"{role} at {org}" if org else role
            blocks[name] = block
        return blocks

    # -- Strategy 2: \cventry{date}{role}{org}{location}{grade}{desc} -- moderncv --
    cventry_positions: list[int] = []
    for i, line in enumerate(lines):
        if re.match(r"\s*\\cventry\b", line):
            cventry_positions.append(i)

    if cventry_positions:
        logger.debug("extract_experience_blocks: using strategy 2 (moderncv \\cventry)")
        for i, start_line in enumerate(cventry_positions):
            end_line = cventry_positions[i + 1] if i + 1 < len(cventry_positions) else len(lines)
            block_lines = lines[start_line:end_line]
            block = "\n".join(block_lines).rstrip()

            # cventry args: {date}{role}{org}{location}{grade}{desc}
            args = re.findall(r"\{([^}]*)\}", lines[start_line])
            role = args[1].strip() if len(args) >= 2 else "Unknown Role"
            org = args[2].strip() if len(args) >= 3 else ""
            name = f"{role} at {org}" if org else role
            blocks[name] = block
        return blocks

    # -- Strategy 3: \resumeSubheading{role}{date}{org}{location} -- Jake's --
    subheading_positions: list[int] = []
    for i, line in enumerate(lines):
        if re.match(r"\s*\\resumeSubheading\b", line):
            subheading_positions.append(i)

    if subheading_positions:
        logger.debug("extract_experience_blocks: using strategy 3 (Jake's \\resumeSubheading)")
        for i, start_line in enumerate(subheading_positions):
            end_line = subheading_positions[i + 1] if i + 1 < len(subheading_positions) else len(lines)
            block_lines = lines[start_line:end_line]
            block = "\n".join(block_lines).rstrip()

            # resumeSubheading args: {role}{date}{org}{location}
            # May span multiple lines -- join first few lines to parse
            header_text = " ".join(l.strip() for l in block_lines[:4])
            args = re.findall(r"\{([^}]*)\}", header_text)
            role = args[0].strip() if len(args) >= 1 else "Unknown Role"
            org = args[2].strip() if len(args) >= 3 else ""
            name = f"{role} at {org}" if org else role
            blocks[name] = block
        return blocks

    # -- Strategy 4: Lines with both \textbf{} and \hfill (looser match) --
    textbf_hfill_positions: list[int] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if "\\textbf{" in stripped and "\\hfill" in stripped:
            textbf_hfill_positions.append(i)

    if textbf_hfill_positions:
        logger.debug("extract_experience_blocks: using strategy 4 (\\textbf + \\hfill)")
        for i, start_line in enumerate(textbf_hfill_positions):
            end_line = textbf_hfill_positions[i + 1] if i + 1 < len(textbf_hfill_positions) else len(lines)
            block_lines = lines[start_line:end_line]
            block = "\n".join(block_lines).rstrip()

            role_match = re.search(r"\\textbf\{([^}]+)\}", lines[start_line])
            role = role_match.group(1).strip() if role_match else "Unknown Role"
            blocks[role] = block
        return blocks

    # -- Fallback: return entire section as one block --
    logger.debug("extract_experience_blocks: using fallback (whole section)")
    blocks["full_experience"] = section_latex.strip()
    return blocks


def extract_certification_blocks(section_latex: str) -> dict[str, str]:
    r"""Split the Certifications section into individual cert blocks.

    Uses a multi-strategy approach:
      Strategy 1: standalone \textbf{Name} header (not inside \item) -- standard
      Strategy 2: \item \textbf{Name} -- list style
      Strategy 3: \certentry{Name} -- some templates
      Fallback: return {"full_certifications": entire_section_text}

    Returns {name: raw_latex_block}.
    """
    lines = section_latex.split("\n")
    blocks: dict[str, str] = {}

    # -- Strategy 1: standalone \textbf{Name} header outside itemize --
    entry_starts: list[int] = []
    inside_itemize = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("\\begin{itemize}"):
            inside_itemize = True
            continue
        if stripped.startswith("\\end{itemize}"):
            inside_itemize = False
            continue
        if inside_itemize:
            continue
        if re.match(r"\\textbf\{[^}]+\}", stripped) and not stripped.startswith("\\vspace"):
            entry_starts.append(i)

    if entry_starts:
        logger.debug("extract_certification_blocks: using strategy 1 (standalone \\textbf header)")
        for i, start_line in enumerate(entry_starts):
            end_line = entry_starts[i + 1] if i + 1 < len(entry_starts) else len(lines)

            block_lines = lines[start_line:end_line]
            while block_lines and block_lines[-1].strip().startswith("\\vspace"):
                block_lines.pop()

            block = "\n".join(block_lines).rstrip()
            name_match = re.search(r"\\textbf\{([^}]+)\}", lines[start_line])
            name = name_match.group(1).strip() if name_match else f"Certification {i + 1}"
            blocks[name] = block
        return blocks

    # -- Strategy 2: \item \textbf{Name} -- list style --
    item_textbf_positions: list[int] = []
    for i, line in enumerate(lines):
        if re.match(r"\s*\\item\s+\\textbf\{", line):
            item_textbf_positions.append(i)

    if item_textbf_positions:
        logger.debug("extract_certification_blocks: using strategy 2 (\\item \\textbf)")
        for i, start_line in enumerate(item_textbf_positions):
            end_line = item_textbf_positions[i + 1] if i + 1 < len(item_textbf_positions) else len(lines)
            block_lines = lines[start_line:end_line]
            block = "\n".join(block_lines).rstrip()
            name_match = re.search(r"\\textbf\{([^}]+)\}", lines[start_line])
            name = name_match.group(1).strip() if name_match else f"Certification {i + 1}"
            blocks[name] = block
        return blocks

    # -- Strategy 3: \certentry{Name} -- some templates --
    certentry_positions: list[int] = []
    for i, line in enumerate(lines):
        if re.match(r"\s*\\certentry\b", line):
            certentry_positions.append(i)

    if certentry_positions:
        logger.debug("extract_certification_blocks: using strategy 3 (\\certentry)")
        for i, start_line in enumerate(certentry_positions):
            end_line = certentry_positions[i + 1] if i + 1 < len(certentry_positions) else len(lines)
            block_lines = lines[start_line:end_line]
            block = "\n".join(block_lines).rstrip()
            args = re.findall(r"\{([^}]*)\}", lines[start_line])
            name = args[0].strip() if args else f"Certification {i + 1}"
            blocks[name] = block
        return blocks

    # -- Fallback: return entire section as one block --
    logger.debug("extract_certification_blocks: using fallback (whole section)")
    blocks["full_certifications"] = section_latex.strip()
    return blocks


# ==========================================================================
# Step 2b -- LaTeX -> plain text conversion (for inventory only)
# ==========================================================================

def _latex_to_plain(latex: str) -> str:
    """Rough conversion of LaTeX to plain text for inventory display."""
    text = latex
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\textit\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\faGithub\s*", "", text)
    text = re.sub(r"\\faCode\s*", "", text)
    text = re.sub(r"\\hfill\s*", " | ", text)
    text = re.sub(r"\\\\\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\\item\s*", "• ", text)
    text = re.sub(r"\\begin\{[^}]*\}(\[[^\]]*\])?", "", text)
    text = re.sub(r"\\end\{[^}]*\}", "", text)
    text = re.sub(r"\\vspace\{[^}]*\}", "", text)
    # Jake's template macros
    text = re.sub(r"\\resumeItem\{([^}]*)\}", r"• \1", text)
    text = re.sub(
        r"\\resumeProjectHeading\{([^}]*)\}\{([^}]*)\}", r"\1 | \2", text
    )
    text = re.sub(
        r"\\resumeSubheading\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}",
        r"\1 | \2 | \3 | \4",
        text,
    )
    text = re.sub(r"\\resumeItemListStart", "", text)
    text = re.sub(r"\\resumeItemListEnd", "", text)
    text = re.sub(r"\\resumeSubHeadingListStart", "", text)
    text = re.sub(r"\\resumeSubHeadingListEnd", "", text)
    # moderncv macros
    text = re.sub(
        r"\\cventry\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}",
        r"\2 | \3 | \1 | \4",
        text,
    )
    text = re.sub(r"\\cvitem\{([^}]*)\}\{([^}]*)\}", r"\1: \2", text)
    # Generic cleanup
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\$([^$]*)\$", r"\1", text)
    text = re.sub(r"~", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ==========================================================================
# Step 2c -- Build plain-text summaries for each section type
# ==========================================================================

def _summarise_project(name: str, block_latex: str) -> str:
    """Create a one-line plain-text summary of a project block."""
    plain = _latex_to_plain(block_latex)

    # Extract date
    date_match = re.search(r"\|\s*([\w\s]+\d{4}\s*--\s*[\w\s]*)", plain)
    date = date_match.group(1).strip() if date_match else ""
    # Clean trailing "Tools" from date
    date = re.sub(r"\s*Tools\s*$", "", date)

    # Extract tools line
    tools_match = re.search(r"Tools?:\s*([^\n\u2022]+)", plain)
    tools = tools_match.group(1).strip().rstrip(".") if tools_match else ""

    # Get first description bullet
    bullets = re.findall(r"\u2022\s*([^\u2022]+)", plain)
    # Skip the header bullet, find first real description
    desc = ""
    for b in bullets:
        b = b.strip()
        if not b.startswith(name[:20]) and "Tools:" not in b and len(b) > 20:
            desc = b[:120]
            break

    parts = [name]
    if date:
        parts[0] += f" ({date})"
    if tools:
        parts.append(f"Tools: {tools[:100]}")
    if desc:
        parts.append(f"Summary: {desc}")
    return " | ".join(parts)


def _summarise_experience(name: str, block_latex: str) -> str:
    """Create a one-line plain-text summary of an experience block."""
    plain = _latex_to_plain(block_latex)

    # Extract date
    date_match = re.search(r"\|\s*([\w\s]+\d{4}\s*--\s*[\w\s]*\d{4})", plain)
    date = date_match.group(1).strip() if date_match else ""

    # Get first real bullet
    bullets = re.findall(r"\u2022\s*([^\u2022]+)", plain)
    first_bullet = ""
    for b in bullets:
        b = b.strip()
        if len(b) > 20:
            first_bullet = b[:120]
            break

    parts = [name]
    if date:
        parts[0] += f" ({date})"
    if first_bullet:
        parts.append(f"Summary: {first_bullet}")
    return " | ".join(parts)


def _summarise_certification(name: str, block_latex: str) -> str:
    """Create a one-line plain-text summary of a certification block."""
    # Extract relevant skill set if present
    skill_match = re.search(r"Relevant Skill Set:\}?\s*([^\n]+)", block_latex)
    skills = ""
    if skill_match:
        skills = skill_match.group(1).strip().rstrip(".")
        # Clean LaTeX from skills
        skills = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", skills)
        skills = re.sub(r"\\\\", "", skills)

    parts = [name]
    if skills:
        parts.append(f"Skills: {skills}")
    return " | ".join(parts)


def _summarise_skills(section_latex: str) -> list[str]:
    """Extract skill categories and their contents as plain text."""
    summaries = []
    # Use extract_skill_categories for accurate parsing (handles spaces,
    # multi-category lines like Programming \hfill Soft Skills)
    categories = extract_skill_categories(section_latex)
    for category, block in categories.items():
        if category == "full_skills":
            # Fallback — no categories detected, skip
            continue
        # Extract the content after the category header
        # Pattern: \textbf{Category: } content...
        match = re.search(
            r"\\textbf\{" + re.escape(category) + r"\s*:\s*\}\s*(.*)",
            block,
            re.DOTALL,
        )
        if match:
            skills = match.group(1).strip()
        else:
            skills = block
        # Clean LaTeX commands
        skills = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", skills)
        skills = re.sub(r"\\\\", "", skills).strip()
        # Remove \hfill and anything after it (for shared-line categories)
        skills = re.sub(r"\s*\\hfill.*$", "", skills).strip()
        # Clean trailing commas
        skills = skills.rstrip(",").strip()
        if skills:
            summaries.append(f"{category}: {skills}")
    return summaries


def _summarise_education(section_latex: str) -> list[str]:
    """Extract education entries as plain text lines."""
    plain = _latex_to_plain(section_latex)
    # Split by double-newline or by institution pattern
    lines = [line.strip() for line in plain.split("\n") if line.strip()]
    return lines[:5]


def _summarise_leadership(section_latex: str) -> list[str]:
    """Extract leadership/extra-curricular entries as plain text."""
    blocks = extract_item_blocks(section_latex)
    summaries = []
    for name, latex in blocks.items():
        date_match = re.search(r"\\hfill\s*\\textit\{([^}]+)\}", latex)
        date = date_match.group(1).strip() if date_match else ""
        summary = name
        if date:
            summary += f" ({date})"
        summaries.append(summary)
    return summaries


# ==========================================================================
# Step 2d -- Build the full content inventory
# ==========================================================================

def build_content_inventory(
    sections: dict[str, str],
) -> dict[str, dict]:
    """Build a plain-text inventory of all content in master_cv.tex.

    Returns a dict structured for LLM consumption:
    {
        "PROJECTS": {
            "items": ["CouchHire (Mar 2026 -- Present) | Tools: ... | Summary: ...", ...],
            "names": ["CouchHire -- Agentic Job Application Automation", ...],
            "count": 7
        },
        ...
    }
    """
    inventory: dict[str, dict] = {}

    # PROJECTS
    if "Projects" in sections:
        project_blocks = extract_item_blocks(sections["Projects"])
        inventory["PROJECTS"] = {
            "items": [
                _summarise_project(name, latex)
                for name, latex in project_blocks.items()
            ],
            "names": list(project_blocks.keys()),
            "count": len(project_blocks),
        }

    # EXPERIENCE
    if "Experience" in sections:
        exp_blocks = extract_experience_blocks(sections["Experience"])
        inventory["EXPERIENCE"] = {
            "items": [
                _summarise_experience(name, latex)
                for name, latex in exp_blocks.items()
            ],
            "names": list(exp_blocks.keys()),
            "count": len(exp_blocks),
        }

    # CERTIFICATIONS
    if "Certifications" in sections:
        cert_blocks = extract_certification_blocks(sections["Certifications"])
        inventory["CERTIFICATIONS"] = {
            "items": [
                _summarise_certification(name, latex)
                for name, latex in cert_blocks.items()
            ],
            "names": list(cert_blocks.keys()),
            "count": len(cert_blocks),
        }

    # SKILLS
    if "Skills" in sections:
        skill_summaries = _summarise_skills(sections["Skills"])
        inventory["SKILLS"] = {
            "items": skill_summaries,
            "names": [s.split(":")[0] for s in skill_summaries],
            "count": len(skill_summaries),
        }

    # EDUCATION
    if "Education" in sections:
        edu_summaries = _summarise_education(sections["Education"])
        inventory["EDUCATION"] = {
            "items": edu_summaries,
            "names": edu_summaries[:2],
            "count": len(edu_summaries),
        }

    # LEADERSHIP (Extra Curriculars)
    if "Extra Curriculars" in sections:
        leadership_summaries = _summarise_leadership(sections["Extra Curriculars"])
        inventory["LEADERSHIP"] = {
            "items": leadership_summaries,
            "names": [s.split(" (")[0] for s in leadership_summaries],
            "count": len(leadership_summaries),
        }

    # RESEARCH DISSEMINATION (bonus -- no template marker but available for selection)
    if "Research Dissemination" in sections:
        plain = _latex_to_plain(sections["Research Dissemination"])
        lines = [line.strip() for line in plain.split("\n") if line.strip()]
        inventory["RESEARCH_DISSEMINATION"] = {
            "items": lines[:5],
            "names": ["Research Dissemination"],
            "count": 1,
        }

    # ACADEMIC ACHIEVEMENTS (bonus -- no template marker but available for selection)
    if "Academic Achievements" in sections:
        ach_blocks = extract_item_blocks(sections["Academic Achievements"])
        inventory["ACADEMIC_ACHIEVEMENTS"] = {
            "items": [_latex_to_plain(latex)[:150] for latex in ach_blocks.values()],
            "names": list(ach_blocks.keys()),
            "count": len(ach_blocks),
        }

    logger.info(
        "Built content inventory: %s",
        {k: v["count"] for k, v in inventory.items()},
    )
    return inventory


def format_inventory_for_llm(inventory: dict[str, dict]) -> str:
    """Format the content inventory as a readable string for the LLM prompt.

    The LLM sees ONLY this -- never raw LaTeX.
    """
    lines: list[str] = []
    lines.append("=== CONTENT INVENTORY (from master CV) ===")
    lines.append("")

    for section_name, data in inventory.items():
        lines.append(f"-- {section_name} ({data['count']} entries) --")
        for i, item in enumerate(data["items"], 1):
            lines.append(f"  {i}. {item}")
        lines.append("")

    return "\n".join(lines)


def extract_skill_categories(section_latex: str) -> dict[str, str]:
    r"""Split the Skills section into individual category blocks.

    Expects patterns like:
        \textbf{Category:} skill1, skill2, ...
    or:
        \textbf{Category:} skill1, skill2, ... \\

    Returns {category_name: raw_latex_block} preserving the original LaTeX.
    """
    blocks: dict[str, str] = {}
    lines = section_latex.split("\n")

    # Find lines that start a skill category: \textbf{Something:}
    # Also handles \textbf{Something: } (space before closing brace)
    # and multiple categories on the same line (e.g. Programming \hfill Soft Skills)
    category_starts: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        # Find ALL \textbf{Name:} or \textbf{Name: } patterns on this line
        for match in re.finditer(r"\\textbf\{([^}]+?)\s*:\s*\}", line):
            category_starts.append((i, match.group(1).strip()))

    if not category_starts:
        logger.debug("extract_skill_categories: no \\textbf{Cat:} patterns found")
        return {"full_skills": section_latex.strip()}

    logger.debug(
        "extract_skill_categories: found %d categories: %s",
        len(category_starts),
        [name for _, name in category_starts],
    )

    for idx, (start_line, cat_name) in enumerate(category_starts):
        end_line = (
            category_starts[idx + 1][0]
            if idx + 1 < len(category_starts)
            else len(lines)
        )

        # Handle multiple categories on the same line (e.g. Programming \hfill Soft Skills)
        next_same_line = (
            idx + 1 < len(category_starts)
            and category_starts[idx + 1][0] == start_line
        )

        if next_same_line:
            # Split the shared line at the \hfill or \textbf boundary for the NEXT category
            line = lines[start_line]
            next_cat_name = category_starts[idx + 1][1]
            # Find where the next category's \textbf starts
            split_pattern = r"\\hfill\s*\\textbf\{" + re.escape(next_cat_name)
            split_match = re.search(split_pattern, line)
            if split_match:
                block = line[:split_match.start()].rstrip()
            else:
                # Fallback: split at \hfill
                hfill_pos = line.find(r"\hfill")
                if hfill_pos > 0:
                    block = line[:hfill_pos].rstrip()
                else:
                    block = line
        elif (
            idx > 0
            and category_starts[idx - 1][0] == start_line
        ):
            # This is the SECOND category on a shared line — extract from \hfill onwards
            line = lines[start_line]
            # Find this category's \textbf start
            cat_pattern = r"\\textbf\{" + re.escape(cat_name) + r"\s*:\s*\}"
            cat_match = re.search(cat_pattern, line)
            if cat_match:
                # Build a proper \item line from the extracted portion
                extracted = line[cat_match.start():].rstrip()
                # Ensure it starts with \item if the original line had it
                if r"\item" in line[:cat_match.start()] and r"\item" not in extracted:
                    extracted = r"    \item " + extracted
                block = extracted
            else:
                block = "\n".join(lines[start_line:end_line]).rstrip()
        else:
            block = "\n".join(lines[start_line:end_line]).rstrip()

        # Clean any container-closing commands that may have been captured
        # in the last category (e.g. \end{itemize}, \end{tabularx})
        block = re.sub(
            r'\s*\\end\{(?:itemize|enumerate|tabularx|tabular)\}\s*',
            '',
            block,
        ).rstrip()
        blocks[cat_name] = block

    return blocks
