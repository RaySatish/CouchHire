"""LLM-based content selection for the SELECT-ONLY resume tailor pipeline.

Sends a plain-text inventory of CV content + job requirements to the LLM.
The LLM returns a JSON object specifying WHAT to include and in WHAT ORDER.
The LLM never sees or writes LaTeX — only plain-text summaries.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# System prompt — enforces JSON-only output, no LaTeX, no hallucination
# ══════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
You are a resume content strategist. Your ONLY job is to SELECT which \
existing content to include in a tailored 1-page resume and in what ORDER.

RULES:
- You MUST respond with valid JSON only — no markdown, no commentary, \
no code fences, no explanations.
- You may ONLY reference items that appear in the CONTENT INVENTORY below.
- You must use the EXACT names from the inventory — do not rephrase, \
abbreviate, or modify them.
- You do NOT write any resume content. You only select and order.
- If a section should be included in full, set its value to true.
- If specific items from a section should be included, list them by \
their EXACT names from the inventory.
- The final resume must fit on 1 page. Be selective — 3-4 projects is \
typical, not all 7."""


def _build_selection_prompt(
    inventory_text: str,
    requirements: dict,
    instructions: str,
    template_sections: list[str],
) -> str:
    """Build the user prompt for the LLM selection call.

    Args:
        inventory_text: Formatted plain-text inventory from
            format_inventory_for_llm().
        requirements: The requirements dict from jd_parser.
        instructions: Tailoring instructions text (from ChromaDB or file).
        template_sections: List of template marker names
            (e.g. ['HEADER', 'EDUCATION', ...]).

    Returns:
        The full user prompt string.
    """
    role = requirements.get("role", "the role")
    company = requirements.get("company", "the company")
    skills = requirements.get("skills", [])
    email_instructions = requirements.get("email_instructions")

    skills_text = ", ".join(skills) if skills else "Not specified"

    email_context = ""
    if email_instructions:
        email_context = f"\nEMAIL/APPLICATION INSTRUCTIONS FROM JD: {email_instructions}"

    return f"""\
TARGET ROLE: {role}
TARGET COMPANY: {company}
KEY SKILLS REQUIRED: {skills_text}
{email_context}

TAILORING INSTRUCTIONS:
{instructions}

AVAILABLE TEMPLATE SECTIONS (these are the sections in the resume template):
{', '.join(template_sections)}

{inventory_text}

Based on the job requirements and the content inventory above, select what \
to include in the tailored resume.

Return a JSON object with EXACTLY this structure:
{{
    "sections_to_include": {{
        "HEADER": true,
        "EDUCATION": true,
        "EXPERIENCE": ["<exact name from inventory>" or true to include all],
        "PROJECTS": ["<exact project name 1>", "<exact project name 2>", ...],
        "SKILLS": true,
        "CERTIFICATIONS": ["<exact cert name 1>", ...] or true for all,
        "LEADERSHIP": ["<exact entry name 1>", ...] or true for all
    }},
    "project_order": ["<most relevant project>", "<second most>", ...],
    "experience_order": ["<most relevant experience>", ...],
    "rationale": "<1-2 sentences explaining your selection strategy>"
}}

IMPORTANT:
- HEADER is ALWAYS true (contains personal details).
- EDUCATION is ALWAYS true.
- SKILLS is ALWAYS true.
- For PROJECTS: select 3-5 most relevant projects. Order them with the \
most relevant first.
- For EXPERIENCE: select relevant entries. If only 1 exists, include it.
- For CERTIFICATIONS: select the most relevant ones.
- For LEADERSHIP: include if space allows and entries are relevant.
- project_order and experience_order must list items in the ORDER they \
should appear (most relevant first).
- Every name you list MUST appear exactly as written in the CONTENT \
INVENTORY above.
- If a template section (e.g. LEADERSHIP) has no matching content in the \
inventory, set it to false or an empty list — do NOT invent content."""


def _clean_json_response(raw: str) -> str:
    """Strip markdown fences and other wrapping from LLM JSON response."""
    text = raw.strip()
    # Remove ```json ... ``` wrapping
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()
    # Fix LaTeX backslash escapes that break JSON parsing.
    # LLMs often copy LaTeX names verbatim (e.g. "TinyML \\& Visual").
    # These are invalid JSON escapes — replace with plain equivalents.
    text = text.replace("\\&", "&")
    text = text.replace("\\%", "%")
    text = text.replace("\\#", "#")
    text = text.replace("\\_", "_")
    text = text.replace("\\~", "~")
    # Also handle single-backslash LaTeX escapes that are invalid in JSON.
    # Inside JSON strings, \& is not a valid escape — json.loads() rejects it.
    # Fix: replace any \X where X is not a valid JSON escape char with just X.
    # Valid JSON escapes: \" \/ \\ \b \f \n \r \t \uXXXX
    # re already imported at module level
    text = re.sub(
        r'(?<!\\)\\(?!["\\/bfnrtu])',
        "",
        text,
    )
    return text


def _validate_selection(
    selection: dict,
    inventory: dict[str, dict],
) -> dict:
    """Validate and sanitise the LLM's selection JSON against the inventory.

    - Ensures all referenced names exist in the inventory.
    - Removes any names the LLM hallucinated.
    - Ensures required sections (HEADER, EDUCATION, SKILLS) are always true.
    - Logs warnings for any corrections made.

    Args:
        selection: The parsed JSON from the LLM.
        inventory: The content inventory dict from build_content_inventory().

    Returns:
        A cleaned selection dict safe for the assembly step.
    """
    sections = selection.get("sections_to_include", {})

    # Force required sections
    for required in ("HEADER", "EDUCATION", "SKILLS"):
        if not sections.get(required):
            logger.warning(
                "LLM omitted required section '%s' — forcing to true", required
            )
            sections[required] = True

    # Validate list-based selections against inventory names
    for section_key in ("PROJECTS", "EXPERIENCE", "CERTIFICATIONS", "LEADERSHIP"):
        value = sections.get(section_key)
        if isinstance(value, list) and section_key in inventory:
            valid_names = set(inventory[section_key]["names"])
            cleaned = []
            for name in value:
                if name in valid_names:
                    cleaned.append(name)
                else:
                    # Try fuzzy match — LLM might have slightly mangled the name
                    matched = _fuzzy_match(name, valid_names)
                    if matched:
                        logger.warning(
                            "LLM used '%s' — corrected to '%s'", name, matched
                        )
                        cleaned.append(matched)
                    else:
                        logger.warning(
                            "LLM hallucinated item '%s' in %s — removed",
                            name,
                            section_key,
                        )
            sections[section_key] = cleaned
        elif value is True:
            # "true" means include all — that's fine
            pass
        elif value is False or value is None:
            # Explicitly excluded — that's fine
            sections[section_key] = False

    selection["sections_to_include"] = sections

    # Validate project_order against what was selected
    if "project_order" in selection and isinstance(selection["project_order"], list):
        selected_projects = sections.get("PROJECTS", [])
        if isinstance(selected_projects, list):
            valid_project_names = set(selected_projects)
            selection["project_order"] = [
                p for p in selection["project_order"] if p in valid_project_names
            ]
            # Add any selected projects missing from the order
            for p in selected_projects:
                if p not in selection["project_order"]:
                    selection["project_order"].append(p)
    else:
        # Default: use the order from sections_to_include
        selected_projects = sections.get("PROJECTS", [])
        if isinstance(selected_projects, list):
            selection["project_order"] = list(selected_projects)
        else:
            selection["project_order"] = []

    # Validate experience_order similarly
    if "experience_order" in selection and isinstance(
        selection["experience_order"], list
    ):
        selected_exp = sections.get("EXPERIENCE", [])
        if isinstance(selected_exp, list):
            valid_exp_names = set(selected_exp)
            selection["experience_order"] = [
                e for e in selection["experience_order"] if e in valid_exp_names
            ]
            for e in selected_exp:
                if e not in selection["experience_order"]:
                    selection["experience_order"].append(e)
    else:
        selected_exp = sections.get("EXPERIENCE", [])
        if isinstance(selected_exp, list):
            selection["experience_order"] = list(selected_exp)
        else:
            selection["experience_order"] = []

    return selection


def _strip_latex_escapes(text: str) -> str:
    """Remove LaTeX backslash escapes for comparison (e.g. \\& -> &)."""
    for char in ("&", "%", "#", "_", "~"):
        text = text.replace("\\" + char, char)
    return text


def _fuzzy_match(name: str, valid_names: set[str]) -> str | None:
    """Try to match a slightly wrong name to a valid one.

    Uses simple substring matching — if the LLM's name is a substring of
    a valid name (or vice versa), return the valid name.
    Also strips LaTeX escapes before comparing (LLM may return plain & vs \\&).
    """
    name_lower = _strip_latex_escapes(name.lower().strip())
    for valid in valid_names:
        valid_lower = _strip_latex_escapes(valid.lower().strip())
        # Exact match (case-insensitive, LaTeX-escape-insensitive)
        if name_lower == valid_lower:
            return valid
        # Substring match (either direction)
        if name_lower in valid_lower or valid_lower in name_lower:
            return valid
        # First 30 chars match (handles truncation)
        if len(name_lower) > 15 and name_lower[:30] == valid_lower[:30]:
            return valid
    return None


def select_content(
    inventory: dict[str, dict],
    inventory_text: str,
    requirements: dict,
    instructions: str,
    template_sections: list[str],
) -> dict:
    """Ask the LLM to select which CV content to include in the tailored resume.

    This is the ONLY LLM call in the selection pipeline. The LLM sees
    plain-text inventory only — never raw LaTeX.

    Args:
        inventory: The structured inventory dict from build_content_inventory().
        inventory_text: The formatted plain-text inventory string.
        requirements: The requirements dict from jd_parser.
        instructions: Tailoring instructions text.
        template_sections: List of template section marker names.

    Returns:
        A validated selection dict with keys:
        - sections_to_include: dict mapping section names to True or list of names
        - project_order: list of project names in display order
        - experience_order: list of experience names in display order
        - rationale: str explaining the selection strategy

    Raises:
        RuntimeError: If the LLM fails to return valid JSON after retries.
    """
    from llm.client import complete

    prompt = _build_selection_prompt(
        inventory_text=inventory_text,
        requirements=requirements,
        instructions=instructions,
        template_sections=template_sections,
    )

    # Try up to 2 times — LLMs occasionally return malformed JSON
    last_error: Exception | None = None
    for attempt in range(1, 3):
        logger.info("LLM selection call (attempt %d/2)", attempt)

        raw_response = complete(prompt, system_prompt=_SYSTEM_PROMPT)
        cleaned = _clean_json_response(raw_response)

        try:
            selection = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning(
                "LLM returned invalid JSON (attempt %d): %s\nRaw: %s",
                attempt,
                exc,
                cleaned[:500],
            )
            last_error = exc
            continue

        if not isinstance(selection, dict):
            logger.warning(
                "LLM returned non-dict JSON (attempt %d): %s",
                attempt,
                type(selection).__name__,
            )
            last_error = TypeError(f"Expected dict, got {type(selection).__name__}")
            continue

        if "sections_to_include" not in selection:
            logger.warning(
                "LLM JSON missing 'sections_to_include' key (attempt %d)",
                attempt,
            )
            last_error = KeyError("sections_to_include")
            continue

        # Validate and sanitise
        validated = _validate_selection(selection, inventory)

        logger.info(
            "LLM selection validated — sections: %s, projects: %s",
            {
                k: (v if isinstance(v, bool) else f"{len(v)} items")
                for k, v in validated["sections_to_include"].items()
            },
            validated.get("project_order", []),
        )

        return validated

    # Both attempts failed
    raise RuntimeError(
        f"LLM failed to return valid selection JSON after 2 attempts. "
        f"Last error: {last_error}"
    )
