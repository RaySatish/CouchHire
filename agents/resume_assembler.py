"""Resume assembler — template parsing, style matching, and content utilities.

Provides functions to extract style examples from LaTeX templates with
%%INJECT:X%%...%%END:X%% markers, fuzzy-match LaTeX content names,
detect formatting mismatches, reformat via LLM, and count PDF pages.

Also provides template block extraction for the 3-tier content resolution:
  TIER 1: Use template version verbatim (if item exists in template)
  TIER 2: Reformat master_cv content to match template style (if item only in master_cv)
  TIER 3: Use master_cv content as-is (if template has no examples for this section)

Imported by resume_tailor.py for the SELECT-ONLY pipeline.
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Regex patterns for inject/end markers
_INJECT_PATTERN = re.compile(r'^%%INJECT:([A-Z_]+)%%\s*$', re.MULTILINE)
_END_PATTERN_TEMPLATE = r'^%%END:{section}%%\s*$'


# ═══════════════════════════════════════════════════════════════════════════════
# Template parsing — extract style examples from %%INJECT%% blocks
# ═══════════════════════════════════════════════════════════════════════════════


def extract_style_examples(template_text: str) -> dict[str, str]:
    r"""Parse %%INJECT:X%%...%%END:X%% blocks from template text.

    Returns {section_name: content_between_markers} as style reference.
    Content is stripped of leading/trailing whitespace.
    Empty blocks return empty string.

    Accepts template text directly (not a path) for flexibility.
    """
    sections: dict[str, str] = {}

    for match in _INJECT_PATTERN.finditer(template_text):
        section_name = match.group(1)
        inject_end = match.end()

        # Find the corresponding %%END:SECTIONNAME%%
        end_pattern = re.compile(
            _END_PATTERN_TEMPLATE.format(section=re.escape(section_name)),
            re.MULTILINE,
        )
        end_match = end_pattern.search(template_text, inject_end)

        if end_match is None:
            logger.warning(
                "No %%END:%s%% marker found for %%INJECT:%s%% at offset %d — skipping",
                section_name,
                section_name,
                match.start(),
            )
            continue

        block_content = template_text[inject_end:end_match.start()].strip()
        sections[section_name] = block_content
        logger.debug(
            "Extracted section %s: %d chars",
            section_name,
            len(block_content),
        )

    logger.info(
        "Extracted %d sections from template: %s",
        len(sections),
        list(sections.keys()),
    )
    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# Template block extraction — split %%INJECT%% content into named sub-blocks
# ═══════════════════════════════════════════════════════════════════════════════


def _normalize_name(name: str) -> str:
    r"""Normalize a name for comparison: lowercase, strip LaTeX, collapse whitespace."""
    # Strip LaTeX escapes
    name = re.sub(r"\\([&%$#_{}~^])", r"\1", name)
    # Remove common LaTeX commands
    name = re.sub(r"\\textbf\{([^}]*)\}", r"\1", name)
    name = re.sub(r"\\textit\{([^}]*)\}", r"\1", name)
    name = re.sub(r"\\[a-zA-Z]+", "", name)
    # Remove braces, dollar signs, pipes
    name = re.sub(r"[{}$|]", "", name)
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip().lower()
    return name


def extract_template_project_blocks(projects_latex: str) -> dict[str, str]:
    r"""Extract individual project blocks from template PROJECTS content.

    Handles any template style by detecting block boundaries at
    top-level \textbf{Name} entries (not nested \textbf inside bullets).
    Returns {project_name: full_latex_block_including_itemize}.

    Detection strategy:
    - Lines starting with \textbf{Name} (possibly with leading whitespace)
      that are NOT inside a \begin{itemize}...\end{itemize} block
    - Each such line starts a new project block
    - Block includes everything until the next top-level \textbf{} or end
    """
    if not projects_latex.strip():
        return {}

    lines = projects_latex.split("\n")
    blocks: dict[str, str] = {}

    # Find top-level \textbf{} positions (not inside itemize)
    block_starts: list[tuple[int, str]] = []  # (line_index, project_name)
    itemize_depth = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Track itemize nesting
        itemize_depth += stripped.count(r"\begin{itemize}")
        itemize_depth -= stripped.count(r"\end{itemize}")
        # Clamp to 0 in case of parsing issues
        itemize_depth = max(0, itemize_depth)

        # Only consider top-level \textbf{} (depth == 0 BEFORE this line opened any)
        # We need to check before the itemize opens on this line
        if itemize_depth == 0 or (
            itemize_depth > 0
            and r"\begin{itemize}" in stripped
            and r"\textbf{" not in stripped
        ):
            continue

        # Check for top-level \textbf at depth 0
        # Re-check: the line must not be inside an itemize
        pass

    # Reset and use a cleaner approach
    block_starts = []
    itemize_depth = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # A top-level \textbf{Name} line is one that:
        # 1. Contains \textbf{ near the start (after optional whitespace)
        # 2. Is NOT preceded by \item (which would make it a bullet)
        # 3. Is at itemize depth 0

        if itemize_depth == 0 and re.match(r"\s*\\textbf\{", stripped):
            # This is a top-level project header
            name_match = re.search(r"\\textbf\{([^}]+)\}", stripped)
            if name_match:
                name = name_match.group(1).strip()
                # Clean up common suffixes like " -- Description"
                name = re.split(r"\s*--\s*", name)[0].strip()
                block_starts.append((i, name))

        # Track itemize nesting (after checking the line)
        itemize_depth += stripped.count(r"\begin{itemize}")
        itemize_depth -= stripped.count(r"\end{itemize}")
        itemize_depth = max(0, itemize_depth)

    if not block_starts:
        # Fallback: try \item \textbf{Name} pattern (some templates use this)
        for i, line in enumerate(lines):
            stripped = line.strip()
            match = re.match(r"\\item\s+\\textbf\{([^}]+)\}", stripped)
            if match:
                name = match.group(1).strip()
                name = re.split(r"\s*--\s*", name)[0].strip()
                block_starts.append((i, name))

    if not block_starts:
        # No individual blocks found — return entire content as one block
        logger.debug("extract_template_project_blocks: no blocks detected, returning whole section")
        return {"full_section": projects_latex.strip()}

    # Extract blocks
    for idx, (start_line, name) in enumerate(block_starts):
        end_line = (
            block_starts[idx + 1][0]
            if idx + 1 < len(block_starts)
            else len(lines)
        )
        block = "\n".join(lines[start_line:end_line]).rstrip()

        # Strip trailing \vspace{...} and blank lines from block end
        block = re.sub(r"\s*\\vspace\{[^}]*\}\s*$", "", block).rstrip()

        blocks[name] = block

    logger.info(
        "Extracted %d template project blocks: %s",
        len(blocks),
        list(blocks.keys()),
    )
    return blocks


def extract_template_experience_blocks(experience_latex: str) -> dict[str, str]:
    r"""Extract individual experience blocks from template EXPERIENCE content.

    Detects \textbf{Role} \hfill patterns at top level.
    Returns {role_name: full_latex_block}.
    """
    if not experience_latex.strip():
        return {}

    lines = experience_latex.split("\n")
    blocks: dict[str, str] = {}
    entry_starts: list[tuple[int, str]] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Match \textbf{Role} \hfill — standard experience header
        if re.match(r"\\textbf\{[^}]+\}\s*\\hfill", stripped):
            name_match = re.search(r"\\textbf\{([^}]+)\}", stripped)
            if name_match:
                name = name_match.group(1).strip()
                entry_starts.append((i, name))

    if not entry_starts:
        # Try \resumeSubheading pattern (Jake's template)
        for i, line in enumerate(lines):
            if re.match(r"\s*\\resumeSubheading\b", line.strip()):
                name_match = re.search(r"\\textbf\{([^}]+)\}", line)
                if name_match:
                    entry_starts.append((i, name_match.group(1).strip()))

    if not entry_starts:
        return {"full_experience": experience_latex.strip()}

    for idx, (start_line, name) in enumerate(entry_starts):
        end_line = (
            entry_starts[idx + 1][0]
            if idx + 1 < len(entry_starts)
            else len(lines)
        )
        block = "\n".join(lines[start_line:end_line]).rstrip()
        block = re.sub(r"\s*\\vspace\{[^}]*\}\s*$", "", block).rstrip()
        blocks[name] = block

    logger.info(
        "Extracted %d template experience blocks: %s",
        len(blocks),
        list(blocks.keys()),
    )
    return blocks


def extract_template_certification_blocks(certs_latex: str) -> dict[str, str]:
    r"""Extract individual certification blocks from template CERTIFICATIONS content.

    Handles both \item \textbf{Name} and standalone \textbf{Name} patterns.
    Returns {cert_name: full_latex_block}.
    """
    if not certs_latex.strip():
        return {}

    lines = certs_latex.split("\n")
    blocks: dict[str, str] = {}
    entry_starts: list[tuple[int, str]] = []

    # Strategy 1: \item \textbf{Name} pattern
    for i, line in enumerate(lines):
        stripped = line.strip()
        match = re.match(r"\\item\s+\\textbf\{([^}]+)\}", stripped)
        if match:
            entry_starts.append((i, match.group(1).strip()))

    if not entry_starts:
        # Strategy 2: standalone \textbf{Name} (not inside itemize)
        itemize_depth = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(r"\begin{itemize}"):
                itemize_depth += 1
                continue
            if stripped.startswith(r"\end{itemize}"):
                itemize_depth -= 1
                continue
            if itemize_depth == 0:
                match = re.match(r"\\textbf\{([^}]+)\}", stripped)
                if match:
                    entry_starts.append((i, match.group(1).strip()))

    if not entry_starts:
        return {"full_certifications": certs_latex.strip()}

    for idx, (start_line, name) in enumerate(entry_starts):
        end_line = (
            entry_starts[idx + 1][0]
            if idx + 1 < len(entry_starts)
            else len(lines)
        )
        block = "\n".join(lines[start_line:end_line]).rstrip()
        # Strip trailing \vspace
        block = re.sub(r"\s*\\vspace\{[^}]*\}\s*$", "", block).rstrip()
        blocks[name] = block

    logger.info(
        "Extracted %d template certification blocks: %s",
        len(blocks),
        list(blocks.keys()),
    )
    return blocks


def extract_template_leadership_blocks(leadership_latex: str) -> dict[str, str]:
    r"""Extract individual leadership blocks from template LEADERSHIP content.

    Handles \item \textbf{Name} patterns with nested itemize for descriptions.
    Returns {name: full_latex_block}.
    """
    if not leadership_latex.strip():
        return {}

    lines = leadership_latex.split("\n")
    blocks: dict[str, str] = {}
    entry_starts: list[tuple[int, str]] = []

    # Find top-level \item \textbf{Name} entries
    # These are inside the outer itemize but NOT inside nested itemize
    outer_itemize_depth = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith(r"\begin{itemize}"):
            outer_itemize_depth += 1
            continue
        if stripped.startswith(r"\end{itemize}"):
            outer_itemize_depth -= 1
            continue

        # Top-level items are at depth 1 (inside the outer itemize)
        if outer_itemize_depth == 1:
            match = re.match(r"\\item\s+\\textbf\{([^}]+)\}", stripped)
            if match:
                entry_starts.append((i, match.group(1).strip()))

    if not entry_starts:
        return {"full_leadership": leadership_latex.strip()}

    for idx, (start_line, name) in enumerate(entry_starts):
        end_line = (
            entry_starts[idx + 1][0]
            if idx + 1 < len(entry_starts)
            else len(lines)
        )
        block = "\n".join(lines[start_line:end_line]).rstrip()
        block = re.sub(r"\s*\\vspace\{[^}]*\}\s*$", "", block).rstrip()
        blocks[name] = block

    logger.info(
        "Extracted %d template leadership blocks: %s",
        len(blocks),
        list(blocks.keys()),
    )
    return blocks


def build_template_block_registry(
    template_text: str,
) -> dict[str, dict[str, str]]:
    """Build a complete block registry from template %%INJECT%% content.

    Returns {MARKER: {item_name: latex_block}} for all sections that have
    individual items (PROJECTS, EXPERIENCE, CERTIFICATIONS, LEADERSHIP).

    This is the template-side parallel to the master_cv block_registry
    built in resume_tailor.py.
    """
    style_examples = extract_style_examples(template_text)
    registry: dict[str, dict[str, str]] = {}

    if "PROJECTS" in style_examples and style_examples["PROJECTS"]:
        registry["PROJECTS"] = extract_template_project_blocks(
            style_examples["PROJECTS"]
        )

    if "EXPERIENCE" in style_examples and style_examples["EXPERIENCE"]:
        registry["EXPERIENCE"] = extract_template_experience_blocks(
            style_examples["EXPERIENCE"]
        )

    if "CERTIFICATIONS" in style_examples and style_examples["CERTIFICATIONS"]:
        registry["CERTIFICATIONS"] = extract_template_certification_blocks(
            style_examples["CERTIFICATIONS"]
        )

    if "LEADERSHIP" in style_examples and style_examples["LEADERSHIP"]:
        registry["LEADERSHIP"] = extract_template_leadership_blocks(
            style_examples["LEADERSHIP"]
        )

    logger.info(
        "Template block registry: %s",
        {k: list(v.keys()) for k, v in registry.items()},
    )
    return registry


def detect_project_separator(projects_latex: str) -> str:
    r"""Detect the separator used between projects in template content.

    Looks for \vspace{Xmm} or \vspace{Xcm} patterns between project blocks.
    Returns the full separator string including surrounding whitespace.
    Falls back to "\n\n" if no vspace pattern found.
    """
    # Look for \vspace{...} between project blocks
    vspace_match = re.search(
        r"([ \t]*\\vspace\{[^}]+\}[ \t]*)",
        projects_latex,
    )
    if vspace_match:
        vspace = vspace_match.group(1).strip()
        # Reconstruct with consistent surrounding whitespace
        separator = f"\n\n    {vspace}\n    \n"
        logger.debug("Detected project separator: %s", repr(vspace))
        return separator

    logger.debug("No vspace separator found — using double newline")
    return "\n\n"


# ═══════════════════════════════════════════════════════════════════════════════
# Fuzzy matching — LaTeX-escape-aware name matching
# ═══════════════════════════════════════════════════════════════════════════════


def _strip_latex_escapes(text: str) -> str:
    r"""Strip LaTeX escape sequences for comparison (e.g. \& -> &)."""
    return re.sub(r"\\([&%$#_{}~^])", r"\1", text)


def _fuzzy_find(name: str, blocks: dict[str, str]) -> str | None:
    """Find a block key that fuzzy-matches the given name.

    Uses case-insensitive substring matching with LaTeX escape stripping.
    Returns the matching key or None if no match found.
    """
    name_clean = _strip_latex_escapes(name).lower().strip()
    for key in blocks:
        key_clean = _strip_latex_escapes(key).lower().strip()
        if name_clean == key_clean:
            return key
        if name_clean in key_clean or key_clean in name_clean:
            return key
        # First 30 chars match (handles truncation)
        if len(name_clean) > 15 and name_clean[:30] == key_clean[:30]:
            return key
    return None


def _fuzzy_find_in_template(
    name: str,
    template_blocks: dict[str, str],
) -> str | None:
    """Find a matching block in template_blocks using normalized name comparison.

    More aggressive normalization than _fuzzy_find — strips all LaTeX commands,
    removes punctuation, and compares core words.
    """
    name_norm = _normalize_name(name)

    for key in template_blocks:
        key_norm = _normalize_name(key)
        # Exact normalized match
        if name_norm == key_norm:
            return key
        # Substring match (either direction)
        if name_norm in key_norm or key_norm in name_norm:
            return key
        # First significant word match (for cases like "CouchHire" vs
        # "CouchHire -- Agentic Job Application Automation")
        name_words = name_norm.split()
        key_words = key_norm.split()
        if name_words and key_words and name_words[0] == key_words[0]:
            return key

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Style detection — heuristic + LLM reformatting
# ═══════════════════════════════════════════════════════════════════════════════


def needs_reformatting(master_latex: str, template_style: str) -> bool:
    r"""Heuristic: returns True if master_latex uses different LaTeX
    formatting commands than template_style.

    Compares the first 3 unique LaTeX commands found in the first 500
    chars of each input. If the sets overlap by less than 50%, returns
    True (meaning the content needs reformatting).
    """
    def _extract_commands(text: str, limit: int = 500) -> set[str]:
        """Extract unique LaTeX command names from the first `limit` chars."""
        snippet = text[:limit]
        # Match \commandname (backslash + one or more letters)
        commands = re.findall(r"\\([a-zA-Z]+)", snippet)
        # Return first 3 unique commands (preserving discovery order)
        seen: set[str] = set()
        result: set[str] = set()
        for cmd in commands:
            if cmd not in seen:
                seen.add(cmd)
                result.add(cmd)
                if len(result) >= 3:
                    break
        return result

    master_cmds = _extract_commands(master_latex)
    template_cmds = _extract_commands(template_style)

    if not master_cmds or not template_cmds:
        # If either has no commands, can't compare — assume reformatting needed
        return True

    overlap = master_cmds & template_cmds
    union = master_cmds | template_cmds

    if not union:
        return False

    overlap_ratio = len(overlap) / len(union)
    logger.debug(
        "needs_reformatting: master_cmds=%s, template_cmds=%s, "
        "overlap=%.2f",
        master_cmds,
        template_cmds,
        overlap_ratio,
    )
    return overlap_ratio < 0.5


def reformat_to_template_style(
    content_latex: str,
    style_example: str,
    section_name: str,
) -> str:
    r"""Reformat content_latex to match the visual style of style_example.

    Uses one LLM call. Content facts, metrics, links are never changed.
    Raises ValueError if any \href URL from content_latex is missing in
    the LLM output.
    """
    from llm.client import complete

    prompt = (
        f"Here is how my resume template formats a {section_name} section:\n"
        f"<style_example>\n{style_example}\n</style_example>\n\n"
        f"Here is my content for this section:\n"
        f"<content>\n{content_latex}\n</content>\n\n"
        f"Reformat the content to match the template style.\n"
        f"Rules you MUST follow:\n"
        f"1. Do NOT change any text, names, metrics, dates, or numbers\n"
        f"2. Do NOT add any content that is not in the source\n"
        f"3. Do NOT remove any bullets, entries, or facts\n"
        f"4. Do NOT change any URLs inside \\href{{}} commands\n"
        f"5. Only change LaTeX formatting commands to match the template "
        f"visual style\n"
        f"6. Return ONLY the reformatted LaTeX content, nothing else\n"
        f"7. No markdown fences, no explanation, no preamble"
    )

    system_prompt = (
        "You are a LaTeX formatting assistant. You reformat LaTeX content "
        "to match a target visual style. You NEVER change any factual "
        "content — only formatting commands. You return raw LaTeX only."
    )

    reformatted = complete(prompt, system_prompt=system_prompt)

    # Strip any markdown fences the LLM might add despite instructions
    reformatted = re.sub(
        r"^```(?:latex|tex)?\s*\n?", "", reformatted, flags=re.MULTILINE
    )
    reformatted = re.sub(
        r"\n?```\s*$", "", reformatted, flags=re.MULTILINE
    )
    reformatted = reformatted.strip()

    # — Validate: every \href{URL} from input must appear in output —
    input_urls = set(re.findall(r"\\href\{([^}]+)\}", content_latex))
    if input_urls:
        output_urls = set(re.findall(r"\\href\{([^}]+)\}", reformatted))
        missing_urls = input_urls - output_urls
        if missing_urls:
            raise ValueError(
                f"Hallucination detected in {section_name}: "
                f"missing URLs: {missing_urls}"
            )

    logger.info(
        "reformat_to_template_style: %s reformatted (%d -> %d chars)",
        section_name,
        len(content_latex),
        len(reformatted),
    )
    return reformatted


# ═══════════════════════════════════════════════════════════════════════════════
# PDF utilities
# ═══════════════════════════════════════════════════════════════════════════════


def get_page_count(pdf_path: Path) -> int:
    """Return number of pages in a compiled PDF using pdfplumber.

    Returns 0 if file not found or cannot be read.
    """
    if not pdf_path.exists():
        logger.warning("get_page_count: file not found: %s", pdf_path)
        return 0

    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            count = len(pdf.pages)
        logger.debug("get_page_count: %s has %d page(s)", pdf_path.name, count)
        return count
    except Exception as exc:
        logger.warning("get_page_count: failed to read %s: %s", pdf_path, exc)
        return 0
