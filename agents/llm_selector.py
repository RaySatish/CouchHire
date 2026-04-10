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

_SYSTEM_PROMPT = (
    "You are a resume content strategist. Your ONLY job is to SELECT which "
    "existing content to include in a tailored resume and in what ORDER.\n"
    "\n"
    "═══ CORE SELECTION PHILOSOPHY ═══\n"
    "The resume must answer ONE question above all else: why is this person "
    "the right fit for THIS specific role at THIS specific company?\n"
    "- Lead with relevance. The strongest fit angle comes first.\n"
    "- Projects are EVIDENCE, not the story. Select projects that prove a "
    "capability the JD explicitly asks for. Fewer, sharper projects beat a "
    "full list. Prefer 3 well-chosen projects over 5 generic ones.\n"
    "- Skills: only categories that map directly to the JD. Cut the rest.\n"
    "- Sell the candidate — but with specificity, not adjectives. Strong "
    "selection creates a confident, focused resume without overselling.\n"
    "\n"
    "═══ ABSOLUTE PRIORITY — MANDATORY USER RULES ═══\n"
    "The prompt below contains a section called \"MANDATORY USER RULES\". "
    "These are NON-NEGOTIABLE constraints set by the resume owner. "
    "They OVERRIDE your own judgement about relevance, ordering, or fit.\n"
    "\n"
    "RULE INTERPRETATION GUIDE:\n"
    "- \"Always include X\" → X MUST appear in your selection. Period.\n"
    "- \"Do not include Y\" → Y MUST NOT appear. Period.\n"
    "- \"Lead with X\" / \"X should be first\" → X MUST be position 1 in the "
    "relevant order list.\n"
    "- Conditional rules (\"for AI/ML roles use...\", \"except for Quant\") → "
    "Check if the condition applies to THIS specific role. If yes, enforce. "
    "If no, skip.\n"
    "- \"Use X in skills\" / \"include X category\" → The named skill category "
    "MUST be in skill_categories_to_include.\n"
    "- \"Do not use X if not relevant/mentioned in JD\" → Check if X appears "
    "in the job description. If NOT mentioned, exclude it. If mentioned, keep it.\n"
    "\n"
    "CRITICAL: Process EVERY rule one by one. Do not skip any. "
    "A single missed rule is a critical failure.\n"
    "Violating ANY mandatory rule is worse than selecting suboptimal content.\n"
    "\n"
    "OUTPUT RULES:\n"
    "- You MUST respond with valid JSON only — no markdown, no commentary, "
    "no code fences, no explanations.\n"
    "- You may ONLY reference items that appear in the CONTENT INVENTORY below.\n"
    "- You must use the EXACT names from the inventory — do not rephrase, "
    "abbreviate, or modify them.\n"
    "- You do NOT write any resume content. You only select and order.\n"
    "- If a section should be included in full, set its value to true.\n"
    "- If specific items from a section should be included, list them by "
    "their EXACT names from the inventory.\n"
    "- Be selective — 3 focused projects beats 5 generic ones.\n"
    "- For SKILLS: you select CATEGORIES only. Individual skill items within "
    "categories are filtered separately — do not try to list individual skills."
)


def _build_selection_prompt(
    inventory_text: str,
    requirements: dict,
    instructions: str,
    template_sections: list[str],
    jd_text: str = "",
) -> str:
    """Build the user prompt for the LLM selection call.

    Args:
        inventory_text: Formatted plain-text inventory from
            format_inventory_for_llm().
        requirements: The requirements dict from jd_parser.
        instructions: Tailoring instructions text (from ChromaDB or file).
        template_sections: List of template marker names
            (e.g. ['HEADER', 'EDUCATION', ...]).
        jd_text: The raw job description text for evaluating conditional rules.

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

    jd_context = ""
    if jd_text:
        # Truncate very long JDs to avoid token waste
        truncated = jd_text[:3000] if len(jd_text) > 3000 else jd_text
        jd_context = f"""

═══════════════════════════════════════════════════════════════════
FULL JOB DESCRIPTION (for evaluating conditional rules like "if not
mentioned in the JD" or "if not relevant to the particular JD"):
═══════════════════════════════════════════════════════════════════
{truncated}
═══════════════════════════════════════════════════════════════════
END OF JOB DESCRIPTION
═══════════════════════════════════════════════════════════════════"""

    return f"""\
TARGET ROLE: {role}
TARGET COMPANY: {company}
KEY SKILLS REQUIRED: {skills_text}
{email_context}

═══════════════════════════════════════════════════════
MANDATORY USER RULES (HIGHEST PRIORITY — override your own judgement):
═══════════════════════════════════════════════════════
{instructions}
═══════════════════════════════════════════════════════
END OF MANDATORY RULES — every rule above MUST be followed literally.
═══════════════════════════════════════════════════════

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
        "skill_categories_to_include": ["<category name 1>", "<category name 2>", ...],
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
- SKILLS is ALWAYS true (the section is always included).
- Your real job for Skills is the "skill_categories_to_include" list:
  1. SELECT which skill categories to include — choose categories most relevant to the role.
  2. ORDER them by relevance — most relevant category first.
  3. Always include "Programming" and "Soft Skills" (if they exist in the inventory),
     but place them at the BOTTOM of the list (they are foundational, not differentiating).
  4. Follow any skill category rules in the MANDATORY USER RULES above.
- Item-level filtering (e.g. "do not use Raspberry Pi if not relevant") is handled
  automatically AFTER your selection — you do NOT need to worry about individual
  skills within a category. Focus only on which CATEGORIES to include and their ORDER.
- HEADER TITLE: The resume header must show the role as "{role} - {company}" \
(e.g. "AI/ML Engineer - CouchHire"). Include this in your rationale so the \
assembler knows to use this format.
- FIT-FIRST ORDERING: Order projects so the one that most directly proves \
fit for THIS role comes first. Ask: which project best answers "why this \
candidate for this role?" — that goes to position 1.
- For PROJECTS: select 3-4 most relevant projects. Fewer sharp picks beat \
a full list. Each selected project must prove a capability the JD asks for.
- For EXPERIENCE: select relevant entries. If only 1 exists, include it.
- For CERTIFICATIONS: select the most relevant ones.
- For LEADERSHIP: include if space allows and entries are relevant.
- project_order and experience_order must list items in the ORDER they \
should appear (most relevant first).
- Every name you list MUST appear exactly as written in the CONTENT \
INVENTORY above.
- If a template section (e.g. LEADERSHIP) has no matching content in the \
inventory, set it to false or an empty list — do NOT invent content.

FINAL CHECK — before returning your JSON, verify EVERY mandatory user rule:
- Re-read each rule in MANDATORY USER RULES above.
- For each "always include" rule: confirm the item IS in your selection.
- For each "do not include" / "exclude" rule: confirm the item is NOT in your selection.
- For each ordering rule (e.g. "lead with X"): confirm X is FIRST in the relevant order list.
- For each conditional rule (e.g. "for AI/ML roles use..."): check if the condition applies to THIS role, and if so, enforce it.
If ANY rule is violated, fix your JSON before returning it."""


def _extract_outermost_json(text: str) -> str:
    """Extract the outermost JSON object from text using brace-depth counting.

    Handles arbitrarily nested JSON that regex-based approaches miss.
    Returns the extracted JSON string, or "" if no valid object found.
    """
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            if in_string:
                escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return ""


def _clean_json_response(raw: str) -> str:
    """Strip markdown fences, thinking tags, and other wrapping from LLM JSON response."""
    text = raw.strip()
    # Remove <think>...</think> blocks (reasoning models like Qwen3, DeepSeek-R1)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()
    # Handle UNCLOSED <think> tags — model may output <think>... with no </think>.
    # In that case, try to extract JSON from after the reasoning, or from the
    # entire response if no JSON is found after the tag.
    if not text or (text.startswith("<think>") and "</think>" not in raw):
        # Unclosed think tag — use brace-depth extraction on the raw text
        extracted = _extract_outermost_json(raw)
        if extracted:
            text = extracted
        else:
            # No JSON found at all — return empty to trigger retry
            return ""
    # Remove ```json ... ``` wrapping
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()
    # If still empty after stripping, try brace-depth extraction
    if not text:
        text = _extract_outermost_json(raw)
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

    # Force required sections to exactly True (not a list).
    # The LLM sometimes returns EDUCATION as a list of entries, which
    # bypasses the TIER 1 template-based assembly (template content is
    # curated — e.g. college only, no school).  Forcing True ensures
    # _assemble_section_content uses the template version verbatim.
    for required in ("HEADER", "EDUCATION", "SKILLS"):
        if sections.get(required) is not True:
            if sections.get(required):
                logger.info(
                    "LLM returned '%s' as %s — forcing to True for template-based assembly",
                    required, type(sections[required]).__name__,
                )
            else:
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

    # Validate skill_categories_to_include against inventory.
    # The LLM may place it at the top level OR inside sections_to_include.
    # Check both locations; prefer sections_to_include if present.
    skill_cats = sections.get("skill_categories_to_include")
    if not skill_cats or not isinstance(skill_cats, list):
        skill_cats = selection.get("skill_categories_to_include")
    skill_inventory_names = set(inventory.get("SKILLS", {}).get("names", []))

    if not skill_cats or not isinstance(skill_cats, list):
        # Missing or empty: default to all categories from inventory
        if skill_inventory_names:
            selection["skill_categories_to_include"] = list(
                inventory["SKILLS"]["names"]
            )
            logger.warning(
                "skill_categories_to_include missing or empty — "
                "defaulting to all %d categories",
                len(skill_inventory_names),
            )
        else:
            selection["skill_categories_to_include"] = []
    else:
        # Validate each category name against inventory
        cleaned_cats: list[str] = []
        for cat in skill_cats:
            if cat in skill_inventory_names:
                cleaned_cats.append(cat)
            else:
                # Try fuzzy match (same pattern as project names)
                matched = _fuzzy_match(cat, skill_inventory_names)
                if matched:
                    logger.warning(
                        "Skill category '%s' corrected to '%s'", cat, matched
                    )
                    cleaned_cats.append(matched)
                else:
                    logger.warning(
                        "LLM hallucinated skill category '%s' — removed", cat
                    )
        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for c in cleaned_cats:
            if c not in seen:
                deduped.append(c)
                seen.add(c)
        selection["skill_categories_to_include"] = deduped

    # Enforce required skill categories: Programming and Soft Skills
    # must always be included if they exist in the inventory.
    # This mirrors the tailoring instructions requirement.
    _required_skill_cats = {"Programming", "Soft Skills"}
    current_cats = selection.get("skill_categories_to_include", [])
    current_cats_lower = {c.lower() for c in current_cats}
    for req_cat in _required_skill_cats:
        # Check if already present (case-insensitive)
        if req_cat.lower() not in current_cats_lower:
            # Check if it exists in inventory
            matched = _fuzzy_match(req_cat, skill_inventory_names)
            if matched:
                current_cats.append(matched)
                logger.info(
                    "Enforcing required skill category '%s' (matched '%s')",
                    req_cat, matched,
                )
    selection["skill_categories_to_include"] = current_cats

    # Store validated skill_categories at top level for _assemble_skills_section
    # and also inside sections_to_include for consistency
    sections["skill_categories_to_include"] = selection["skill_categories_to_include"]

    return selection


def _strip_latex_escapes(text: str) -> str:
    """Remove LaTeX backslash escapes for comparison (e.g. \\& -> &)."""
    for char in ("&", "%", "#", "_", "~"):
        text = text.replace("\\" + char, char)
    return text


def _fuzzy_match(name: str, valid_names: set[str]) -> str | None:
    """Try to match a slightly wrong name to a valid one.

    Uses multiple strategies:
    1. Exact match (case-insensitive, LaTeX-escape-insensitive)
    2. Substring match (either direction)
    3. First 30 chars match (handles truncation)
    4. First-word match for multi-word names (e.g. "Cloud & Systems" matches
       "Cloud & Tools" because both start with "Cloud")
    5. Significant word overlap (>50% of words match)

    Also strips LaTeX escapes before comparing (LLM may return plain & vs \\&).
    """
    name_lower = _strip_latex_escapes(name.lower().strip())
    # Strip common filler words for word-level matching
    _FILLER = {"&", "and", "the", "a", "an", "of", "in", "for", "with", "to"}
    name_words = {w for w in name_lower.split() if w not in _FILLER and len(w) > 1}

    best_match: str | None = None
    best_score = 0

    for valid in valid_names:
        valid_lower = _strip_latex_escapes(valid.lower().strip())
        # 1. Exact match
        if name_lower == valid_lower:
            return valid
        # 2. Substring match (either direction)
        if name_lower in valid_lower or valid_lower in name_lower:
            return valid
        # 3. First 30 chars match (handles truncation)
        if len(name_lower) > 15 and name_lower[:30] == valid_lower[:30]:
            return valid
        # 4. First significant word match (for categories like "Cloud & X")
        valid_words = {w for w in valid_lower.split() if w not in _FILLER and len(w) > 1}
        if name_words and valid_words:
            # Check overlap
            overlap = name_words & valid_words
            if overlap:
                # Score: prefer matches with more absolute overlap AND higher ratio.
                # Use (overlap_count, ratio) as a composite score to break ties.
                min_size = min(len(name_words), len(valid_words))
                ratio = len(overlap) / min_size
                # Composite: absolute overlap count * 10 + ratio (0-1)
                # This ensures 2 matching words always beats 1 matching word
                score = len(overlap) * 10 + ratio
                if score > best_score:
                    best_score = score
                    best_match = valid

    # Return best match if at least 1 significant word overlaps
    # AND the overlap ratio is >= 50%
    if best_match:
        valid_lower = _strip_latex_escapes(best_match.lower().strip())
        valid_words = {w for w in valid_lower.split() if w not in _FILLER and len(w) > 1}
        overlap = name_words & valid_words
        min_size = min(len(name_words), len(valid_words))
        ratio = len(overlap) / min_size if min_size > 0 else 0
        if ratio >= 0.5:
            return best_match
    return None




def _enforce_instructions(selection: dict, instructions: str, inventory: dict, requirements: dict | None = None) -> dict:
    """Post-process the LLM selection to enforce instruction rules that the
    LLM may have missed. This is a generic safety net — not a replacement
    for the LLM following the rules.

    Handles:
    - "lead with X" → force X to position 0 in project_order
    - "always include X" → ensure X is in the selection (any section)
    - "do not include X in Y" → remove X from section Y
    - Cross-section item resolution (e.g. Academic Achievement items
      referenced in LEADERSHIP instructions)
    """
    if not instructions:
        return selection

    if requirements is None:
        requirements = {}

    sections = selection.get("sections_to_include", {})
    project_order = selection.get("project_order", [])
    experience_order = selection.get("experience_order", [])
    instructions_lower = instructions.lower()

    # Helper: find an item in any inventory section by fuzzy name match
    def _find_in_inventory(name_lower: str) -> list[tuple[str, str]]:
        """Return list of (section_key, exact_name) for items matching name_lower."""
        results = []
        for sec_key, sec_data in inventory.items():
            for item_name in sec_data.get("names", []):
                if name_lower in item_name.lower() or item_name.lower() in name_lower:
                    results.append((sec_key, item_name))
        return results

    # ── Enforce "lead with X" for projects ──
    # Match patterns like "Lead with CouchHire" or "lead with X project for Y roles"
    lead_matches = re.findall(
        r"lead\s+with\s+([A-Za-z0-9][A-Za-z0-9 _\-&]+?)(?:\s+project|\s+for|\s*$|\s*\n)",
        instructions_lower,
    )
    for lead_name in lead_matches:
        lead_name = lead_name.strip()
        # Find the matching project in project_order (fuzzy)
        for i, proj in enumerate(project_order):
            if lead_name in proj.lower() or proj.lower().startswith(lead_name):
                if i != 0:
                    # Move to front
                    project_order.insert(0, project_order.pop(i))
                    logger.info(
                        "Instruction enforcement: moved '%s' to front of project_order",
                        proj,
                    )
                break
        else:
            # Project not in project_order — try to add it from inventory
            matches = _find_in_inventory(lead_name)
            for sec_key, exact_name in matches:
                if sec_key == "PROJECTS":
                    project_order.insert(0, exact_name)
                    if isinstance(sections.get("PROJECTS"), list):
                        if exact_name not in sections["PROJECTS"]:
                            sections["PROJECTS"].insert(0, exact_name)
                    logger.info(
                        "Instruction enforcement: added and fronted '%s' in project_order",
                        exact_name,
                    )
                    break

    selection["project_order"] = project_order

    # Also update sections_to_include PROJECTS to match new order
    if isinstance(sections.get("PROJECTS"), list):
        sections["PROJECTS"] = list(project_order)

    # ── Generic "always include X" enforcement ──
    # Parse all "always include" / "always use" rules
    always_matches = re.finditer(
        r"always\s+(?:include|use)\s+(?:the\s+)?(.+?)(?:\s*\(|$|\n)",
        instructions_lower,
        re.MULTILINE,
    )
    for m in always_matches:
        item_phrase = m.group(1).strip().rstrip(".,;")
        # Remove trailing context like "except for Quant roles"
        item_phrase = re.sub(r"\s+except\s+.*$", "", item_phrase)
        item_phrase = re.sub(r"\s+section$", "", item_phrase)

        # Try to find this item in the inventory
        inv_matches = _find_in_inventory(item_phrase)
        for sec_key, exact_name in inv_matches:
            # Map inventory section to selection section
            sel_key = sec_key
            # Common mappings
            if sec_key == "ACADEMIC_ACHIEVEMENTS":
                sel_key = "LEADERSHIP"

            current = sections.get(sel_key)
            if isinstance(current, list):
                if exact_name not in current:
                    current.append(exact_name)
                    sections[sel_key] = current
                    logger.info(
                        "Instruction enforcement (always include): added '%s' to %s",
                        exact_name, sel_key,
                    )
            elif current is True:
                pass  # Already including all
            elif not current:
                sections[sel_key] = [exact_name]
                logger.info(
                    "Instruction enforcement (always include): created %s with '%s'",
                    sel_key, exact_name,
                )

    # ── Section-level "always include" enforcement ──
    # Handle "always include the X section" rules (e.g. "Always Include
    # the experience section"). The generic item-level enforcement above
    # handles items within sections, but this handles the section itself.
    _section_name_map = {
        "experience": "EXPERIENCE",
        "leadership": "LEADERSHIP",
        "certifications": "CERTIFICATIONS",
        "projects": "PROJECTS",
        "skills": "SKILLS",
        "education": "EDUCATION",
    }
    for sec_m in re.finditer(
        r"always\s+(?:include|use)\s+(?:the\s+)?(\w+)\s+section",
        instructions_lower,
        re.MULTILINE,
    ):
        sec_phrase = sec_m.group(1).strip().lower()
        sel_key = _section_name_map.get(sec_phrase)
        if sel_key and not sections.get(sel_key):
            sections[sel_key] = True
            logger.info(
                "Instruction enforcement (section-level): forced '%s' to True "
                "(rule: 'always include the %s section')",
                sel_key, sec_phrase,
            )

    # ── Role-conditional project/section excludes ──
    # Handle "do not use X for Y roles" / "do not use X project for Y roles"
    # These are role-conditional — only apply when the current role matches.
    role_lower = (requirements.get("role", "") or "").lower()

    _ROLE_FILLER = {
        "a", "an", "the", "it", "is", "not", "based", "related",
        "role", "roles", "type", "position", "job", "any",
    }

    for m in re.finditer(
        r"(?:do\s+not|never)\s+(?:include|use)\s+(?:the\s+)?"
        r"(.+?)\s+(?:project\s+)?for\s+(.+?)\s*(?:roles?|positions?)"
        r"(?:\s*[,.]|\s*$|\n)",
        instructions_lower,
        re.MULTILINE,
    ):
        item_phrase = m.group(1).strip().rstrip(".,;")
        role_condition = m.group(2).strip().rstrip(".,;")

        # Extract meaningful terms from the role condition
        condition_terms = [
            t for t in re.split(r"[\s/,\-]+", role_condition)
            if t and t not in _ROLE_FILLER and len(t) > 1
        ]

        # Check if the current role matches the condition
        role_matches = any(
            term in role_lower for term in condition_terms
        )

        if role_matches:
            # Remove the item from PROJECTS and project_order
            inv_matches = _find_in_inventory(item_phrase)
            removed_any = False
            for sec_key, exact_name in inv_matches:
                # Remove from sections_to_include
                sel_key = sec_key
                current = sections.get(sel_key)
                if isinstance(current, list) and exact_name in current:
                    current.remove(exact_name)
                    sections[sel_key] = current
                    removed_any = True

                # Remove from project_order if it's a project
                if sel_key == "PROJECTS" and exact_name in project_order:
                    project_order.remove(exact_name)

            if removed_any:
                logger.info(
                    "Instruction enforcement (role-conditional exclude): "
                    "removed '%s' for role condition '%s' (role: %s)",
                    item_phrase, role_condition, role_lower,
                )

    selection["project_order"] = project_order
    if isinstance(sections.get("PROJECTS"), list):
        # Keep PROJECTS in sync with project_order
        sections["PROJECTS"] = list(project_order)

    # Track source sections restricted by "only include" rules for this role.
    # When a source section is restricted, only the explicitly listed items
    # from that section are allowed in LEADERSHIP — all others must be removed.
    _only_include_allowed: dict[str, set[str]] = {}

    # ── Role-conditional "only include X from Y for Z roles" ──
    # Handle "Only Include X and Y from Z for Q roles" patterns.
    # This adds specific items from a source section to LEADERSHIP
    # (or the appropriate target) when the role matches.
    # Collect matches from BOTH orderings:
    #   Pattern A: "only include X from Y for Z roles"
    #   Pattern B: "include X from Y only for Z roles"  (alternate word order)
    _only_include_matches: list[tuple[str, str, str]] = []
    for m in re.finditer(
        r"only\s+include\s+(.+?)\s+(?:in\s+|from\s+)(?:the\s+)?"
        r"(.+?)\s+for\s+(.+?)\s*(?:roles?|positions?)"
        r"(?:\s*[,.]|\s*$|\n)",
        instructions_lower,
        re.MULTILINE,
    ):
        _only_include_matches.append((
            m.group(1).strip().rstrip(".,;"),
            m.group(2).strip().rstrip(".,;"),
            m.group(3).strip().rstrip(".,;"),
        ))
    for m in re.finditer(
        r"include\s+(.+?)\s+(?:in\s+)?(?:from\s+)(?:the\s+)?"
        r"(.+?)\s+only\s+for\s+(.+?)\s*(?:roles?|positions?)"
        r"(?:\s*[,.]|\s*$|\n)",
        instructions_lower,
        re.MULTILINE,
    ):
        _only_include_matches.append((
            m.group(1).strip().rstrip(".,;"),
            m.group(2).strip().rstrip(".,;"),
            m.group(3).strip().rstrip(".,;"),
        ))

    for items_str, source_section, role_condition in _only_include_matches:

        # Extract role condition terms
        condition_terms = [
            t for t in re.split(r"[\s/,\-]+", role_condition)
            if t and t not in _ROLE_FILLER and len(t) > 1
        ]

        role_matches = any(
            term in role_lower for term in condition_terms
        )

        if role_matches:
            # Parse individual item names from "X and Y" or "X, Y"
            item_names = re.split(r"\s+and\s+|\s*,\s*", items_str)

            for item_name in item_names:
                item_name = item_name.strip().rstrip(".,;")
                if not item_name or len(item_name) < 3:
                    continue

                # Find the item in inventory — try exact substring first,
                # then fall back to word-overlap matching for fuzzy names
                # (e.g. "solved 400+ problems" → "Solved 400+ algorithmic problems (80+ Hard)")
                inv_matches = _find_in_inventory(item_name)
                if not inv_matches:
                    # Word-overlap fallback: extract significant words from
                    # the item name and find inventory items that share most
                    _stop = {"the", "a", "an", "and", "or", "in", "of", "for", "to", "from"}
                    query_words = {
                        w for w in re.split(r"[\s()\[\],;]+", item_name.lower())
                        if w and w not in _stop and len(w) > 1
                    }
                    best_match = None
                    best_overlap = 0
                    for sec_key, sec_data in inventory.items():
                        for inv_name in sec_data.get("names", []):
                            inv_words = {
                                w for w in re.split(r"[\s()\[\],;]+", inv_name.lower())
                                if w and w not in _stop and len(w) > 1
                            }
                            overlap = len(query_words & inv_words)
                            # Require at least 2 word overlap and >50% of query words
                            if overlap >= 2 and overlap > best_overlap and overlap >= len(query_words) * 0.5:
                                best_overlap = overlap
                                best_match = (sec_key, inv_name)
                    if best_match:
                        inv_matches = [best_match]

                for sec_key, exact_name in inv_matches:
                    # Map to the correct selection section
                    sel_key = sec_key
                    if sec_key == "ACADEMIC_ACHIEVEMENTS":
                        sel_key = "LEADERSHIP"

                    current = sections.get(sel_key)
                    if isinstance(current, list):
                        if exact_name not in current:
                            current.append(exact_name)
                            sections[sel_key] = current
                            logger.info(
                                "Instruction enforcement (role-conditional include): "
                                "added '%s' to %s for role condition '%s'",
                                exact_name, sel_key, role_condition,
                            )
                    elif current is True:
                        pass
                    elif not current:
                        sections[sel_key] = [exact_name]
                        logger.info(
                            "Instruction enforcement (role-conditional include): "
                            "created %s with '%s' for role condition '%s'",
                            sel_key, exact_name, role_condition,
                        )

            # ── "Only include" exclusion: remove non-specified items ──
            # The word "only" means these are the EXCLUSIVE items allowed
            # from this source section. Remove all other items that
            # originated from the source section.
            source_key = None
            for inv_key in inventory:
                if inv_key.lower().replace("_", " ").rstrip("s") in source_section.replace("_", " "):
                    source_key = inv_key
                    break
                if source_section.replace("_", " ") in inv_key.lower().replace("_", " "):
                    source_key = inv_key
                    break

            if source_key:
                # Build set of allowed exact names from this source
                allowed_names = set()
                for item_name in item_names:
                    item_name = item_name.strip().rstrip(".,;")
                    if not item_name or len(item_name) < 3:
                        continue
                    # Re-find matches (same logic as above)
                    matches = _find_in_inventory(item_name)
                    if not matches:
                        _stop = {"the", "a", "an", "and", "or", "in", "of", "for", "to", "from"}
                        query_words = {
                            w for w in re.split(r"[\s()\[\],;]+", item_name.lower())
                            if w and w not in _stop and len(w) > 1
                        }
                        best_match = None
                        best_overlap = 0
                        for sk, sd in inventory.items():
                            for inv_name in sd.get("names", []):
                                inv_words = {
                                    w for w in re.split(r"[\s()\[\],;]+", inv_name.lower())
                                    if w and w not in _stop and len(w) > 1
                                }
                                overlap = len(query_words & inv_words)
                                if overlap >= 2 and overlap > best_overlap and overlap >= len(query_words) * 0.5:
                                    best_overlap = overlap
                                    best_match = (sk, inv_name)
                        if best_match:
                            matches = [best_match]
                    for _, exact_name in matches:
                        allowed_names.add(exact_name)

                # Track this restriction for the cross-section handler
                _only_include_allowed[source_key] = allowed_names

                # Get all item names from the source section in inventory
                all_source_items = set(inventory.get(source_key, {}).get("names", []))
                disallowed = all_source_items - allowed_names

                # Remove disallowed items from LEADERSHIP (where source items are mapped)
                target_key = "LEADERSHIP" if source_key == "ACADEMIC_ACHIEVEMENTS" else source_key
                current = sections.get(target_key)
                if isinstance(current, list):
                    before_count = len(current)
                    sections[target_key] = [
                        item for item in current if item not in disallowed
                    ]
                    removed = before_count - len(sections[target_key])
                    if removed > 0:
                        logger.info(
                            "Instruction enforcement (only-include exclusion): "
                            "removed %d non-specified %s items from %s for role '%s'. "
                            "Allowed: %s",
                            removed, source_key, target_key, role_condition,
                            allowed_names,
                        )

    # ── Role-conditional "exclude X" from LEADERSHIP for specific roles ──
    # Handle "for Quant roles exclude X" patterns
    for m in re.finditer(
        r"for\s+(.+?)\s*(?:roles?|positions?)\s+exclude\s+(.+?)"
        r"(?:\s+(?:achievement|entry|item|if\b)|\s*[,.]|\s*$|\n)",
        instructions_lower,
        re.MULTILINE,
    ):
        role_condition = m.group(1).strip().rstrip(".,;")
        item_phrase = m.group(2).strip().rstrip(".,;")

        condition_terms = [
            t for t in re.split(r"[\s/,\-]+", role_condition)
            if t and t not in _ROLE_FILLER and len(t) > 1
        ]

        role_matches = any(
            term in role_lower for term in condition_terms
        )

        if role_matches:
            inv_matches = _find_in_inventory(item_phrase)
            for sec_key, exact_name in inv_matches:
                sel_key = sec_key
                if sec_key in ("LEADERSHIP", "ACADEMIC_ACHIEVEMENTS"):
                    sel_key = "LEADERSHIP"

                current = sections.get(sel_key)
                if isinstance(current, list) and exact_name in current:
                    current.remove(exact_name)
                    sections[sel_key] = current
                    logger.info(
                        "Instruction enforcement (role-conditional exclude): "
                        "removed '%s' from %s for role '%s'",
                        exact_name, sel_key, role_condition,
                    )

    # ── Generic "do not include X" enforcement ──
    # Parse "do not include X in Y" / "do not include X"
    exclude_matches = re.finditer(
        r"(?:do\s+not|never)\s+(?:include|use)\s+(?:the\s+)?(.+?)(?:\s+in\s+(?:the\s+)?(.+?))?(?:\s*[,.]|\s*\)|$|\n)",
        instructions_lower,
        re.MULTILINE,
    )
    for m in exclude_matches:
        item_phrase = m.group(1).strip().rstrip(".,;")
        target_section = m.group(2).strip().rstrip(".,;") if m.group(2) else None

        # Skip conditional rules here — they need JD context handled by LLM/filter
        if re.search(r"if\s+(?:not\s+)?(?:relevant|mentioned|it\s+is)", item_phrase):
            continue
        if target_section and re.search(r"if\s+(?:not\s+)?(?:relevant|mentioned|it\s+is)", target_section):
            continue

        # Remove trailing context
        item_phrase = re.sub(r"\s+(?:instead|except|if\s+|for\s+).*$", "", item_phrase)

        # Determine which sections to check
        sections_to_check = []
        if target_section:
            # Map common section name variants
            sec_map = {
                "leadership": "LEADERSHIP",
                "leadership and achievements": "LEADERSHIP",
                "skills": "SKILLS",
                "skill": "SKILLS",
                "projects": "PROJECTS",
                "experience": "EXPERIENCE",
                "certifications": "CERTIFICATIONS",
                "education": "EDUCATION",
            }
            for key, val in sec_map.items():
                if key in target_section:
                    sections_to_check.append(val)
                    break
        if not sections_to_check:
            sections_to_check = list(sections.keys())

        for sec_key in sections_to_check:
            current = sections.get(sec_key)
            if isinstance(current, list):
                cleaned = [
                    item for item in current
                    if item_phrase not in item.lower()
                ]
                if len(cleaned) != len(current):
                    sections[sec_key] = cleaned
                    logger.info(
                        "Instruction enforcement (exclude): removed items matching "
                        "'%s' from %s",
                        item_phrase, sec_key,
                    )

    # ── Skill category enforcement ──
    # Handle "do not include X section in Skills if not Y role" rules
    # and "for X roles use Y" skill category rules
    skill_cats = selection.get("skill_categories_to_include", [])
    if skill_cats:
        skill_inv_names = set(inventory.get("SKILLS", {}).get("names", []))

        # Parse "do not include X ... in ... skills ... if not Y" rules
        for m in re.finditer(
            r"do\s+not\s+include\s+(.+?)\s+(?:section\s+)?in\s+(?:the\s+)?"
            r"skills?\s+(?:section\s+)?if\s+(?:the\s+)?(?:it\s+is\s+)?"
            r"(not\s+)?(?:a\s+)?(.+?)(?:\s*$|\n)",
            instructions_lower,
            re.MULTILINE,
        ):
            items_str = m.group(1).strip()
            condition = m.group(3).strip().rstrip(".,;")

            # Parse category names from the items string
            cat_names = re.split(r"\s+and\s+|\s*,\s*", items_str)

            # Evaluate the condition against the role
            role_lower = (requirements.get("role", "") or "").lower()
            company_lower = (requirements.get("company", "") or "").lower()

            # Generic condition evaluation: extract key terms from the
            # condition string and check if the role matches or doesn't match.
            # Handles patterns like:
            #   "if not a Quant-based role" → exclude unless role contains "quant"
            #   "if the role is AI/ML related" → include if role contains "ai" or "ml"
            condition_met = False
            is_negated = m.group(2) is not None  # regex captured "not" before condition

            # Extract meaningful terms from the condition (strip filler words)
            _CONDITION_FILLER = {
                "a", "an", "the", "it", "is", "not", "based", "related",
                "role", "roles", "type", "position", "job",
            }
            condition_terms = [
                t for t in re.split(r"[\s/,\-]+", condition)
                if t and t not in _CONDITION_FILLER and len(t) > 1
            ]

            if condition_terms:
                # Check if any condition term appears in the role title
                role_has_term = any(
                    term in role_lower or term in company_lower
                    for term in condition_terms
                )
                # "if not X role" → condition met when role does NOT have X
                # "if X role" → condition met when role DOES have X
                condition_met = (not role_has_term) if is_negated else role_has_term

            if condition_met:
                for cat_name in cat_names:
                    cat_name = cat_name.strip().rstrip(".,;")
                    if not cat_name or len(cat_name) < 3:
                        continue
                    # Find matching category in skill_cats (fuzzy)
                    for i, existing_cat in enumerate(skill_cats):
                        if cat_name in existing_cat.lower():
                            removed_cat = skill_cats.pop(i)
                            logger.info(
                                "Instruction enforcement (skill category): "
                                "removed '%s' (condition: %s)",
                                removed_cat, condition,
                            )
                            break

        # Parse "for X roles use Y" skill category rules
        for m in re.finditer(
            r"for\s+(?:any\s+)?(.+?)\s+(?:related\s+)?roles?\s+use\s+(.+?)(?:\s*$|\n)",
            instructions_lower,
            re.MULTILINE,
        ):
            role_types = m.group(1).strip()
            cat_name = m.group(2).strip().rstrip(".,;")

            # Check if this role matches the condition
            role_lower = (requirements.get("role", "") or "").lower()
            role_type_terms = re.split(r"[/,]|\s+", role_types)
            role_matches = any(
                term.strip() in role_lower
                for term in role_type_terms
                if term.strip() and len(term.strip()) > 1
            )

            if role_matches:
                # Find the category in inventory and add if not present
                for inv_cat in skill_inv_names:
                    if cat_name in inv_cat.lower() or inv_cat.lower() in cat_name:
                        if inv_cat not in skill_cats:
                            skill_cats.append(inv_cat)
                            logger.info(
                                "Instruction enforcement (skill category): "
                                "added '%s' for role type '%s'",
                                inv_cat, role_types,
                            )
                        break

        selection["skill_categories_to_include"] = skill_cats
        sections["skill_categories_to_include"] = skill_cats

        # ── "instead include X" enforcement ──
    # Parse rules like "do not include X, instead include Y" or
    # "do not include X in Z, instead include Y"
    instead_matches = re.finditer(
        r"instead\s+(?:include|use)\s+(?:the\s+)?(.+?)(?:\s*\(|\s*[,.]|\s*$|\n)",
        instructions_lower,
        re.MULTILINE,
    )
    for m in instead_matches:
        item_phrase = m.group(1).strip().rstrip(".,;")
        # Remove trailing context
        item_phrase = re.sub(r"\s+(?:except|if\s+|for\s+).*$", "", item_phrase)
        # Remove "/ Merit Scholarship" style alternatives — handle each part
        alt_items = re.split(r"\s*/\s*", item_phrase)

        for alt in alt_items:
            alt = alt.strip()
            if not alt or len(alt) < 3:
                continue

            inv_matches = _find_in_inventory(alt)
            for sec_key, exact_name in inv_matches:
                # Skip if this source section is restricted by an
                # "only include" rule and this item isn't in the allowed set
                if sec_key in _only_include_allowed:
                    if exact_name not in _only_include_allowed[sec_key]:
                        logger.info(
                            "Instruction enforcement (instead include): "
                            "SKIPPED '%s' — %s is restricted by 'only include' rule",
                            exact_name, sec_key,
                        )
                        continue

                sel_key = sec_key
                if sec_key == "ACADEMIC_ACHIEVEMENTS":
                    sel_key = "LEADERSHIP"

                current = sections.get(sel_key)
                if isinstance(current, list):
                    if exact_name not in current:
                        current.append(exact_name)
                        sections[sel_key] = current
                        logger.info(
                            "Instruction enforcement (instead include): "
                            "added '%s' to %s",
                            exact_name, sel_key,
                        )
                elif current is True:
                    pass
                elif not current:
                    sections[sel_key] = [exact_name]
                    logger.info(
                        "Instruction enforcement (instead include): "
                        "created %s with '%s'",
                        sel_key, exact_name,
                    )

    # ── Cross-section resolution: Academic Achievement items in LEADERSHIP ──
    # If instructions mention "Certificate of Merit" or "Merit Scholarship"
    # in the Leadership context, and there's an ACADEMIC_ACHIEVEMENTS section
    # in the inventory, pull matching items into the LEADERSHIP selection.
    # SKIP if ACADEMIC_ACHIEVEMENTS is restricted by an "only include" rule —
    # in that case, only the explicitly allowed items should appear.
    if (
        "ACADEMIC_ACHIEVEMENTS" not in _only_include_allowed
        and any(term in instructions_lower for term in [
            "certificate of merit", "merit scholarship", "merit scholarships"
        ])
    ):
        leadership = sections.get("LEADERSHIP", [])
        if isinstance(leadership, list):
            # Check if already present
            has_merit = any(
                "merit" in item.lower() for item in leadership
            )
            if not has_merit:
                # Look in ACADEMIC_ACHIEVEMENTS inventory
                acad_names = inventory.get("ACADEMIC_ACHIEVEMENTS", {}).get("names", [])
                for acad_name in acad_names:
                    if "merit" in acad_name.lower():
                        leadership.append(acad_name)
                        logger.info(
                            "Instruction enforcement: added '%s' from "
                            "ACADEMIC_ACHIEVEMENTS to LEADERSHIP",
                            acad_name,
                        )
                        break
                # Also check all inventory sections for merit items
                if not any("merit" in item.lower() for item in leadership):
                    for sec_key, sec_data in inventory.items():
                        for item_name in sec_data.get("names", []):
                            if "merit" in item_name.lower():
                                leadership.append(item_name)
                                logger.info(
                                    "Instruction enforcement: added '%s' from "
                                    "%s to LEADERSHIP",
                                    item_name, sec_key,
                                )
                                break
                        if any("merit" in item.lower() for item in leadership):
                            break
                sections["LEADERSHIP"] = leadership

    selection["sections_to_include"] = sections
    return selection



def _verify_and_fix_selection(
    selection: dict,
    instructions: str,
    requirements: dict,
    inventory: dict[str, dict],
    inventory_text: str,
    jd_text: str = "",
) -> dict:
    """Ask the LLM to verify the selection against every instruction rule.

    This is a second LLM call that acts as a self-check. It receives the
    current selection JSON and the full instructions, and returns a corrected
    JSON if any rules are violated.

    This catches issues that _enforce_instructions (regex-based) cannot handle,
    like conditional rules, role-type-dependent rules, and complex multi-part rules.
    """
    from llm.client import complete

    if not instructions:
        return selection

    role = requirements.get("role", "Unknown")
    company = requirements.get("company", "Unknown")

    current_json = json.dumps(selection, indent=2)

    jd_snippet = ""
    if jd_text:
        truncated = jd_text[:2000] if len(jd_text) > 2000 else jd_text
        jd_snippet = f"""
JOB DESCRIPTION (for evaluating conditional rules):
{truncated}
"""

    verify_prompt = f"""You are a strict compliance auditor for resume content selection.

TARGET ROLE: {role}
TARGET COMPANY: {company}
{jd_snippet}
MANDATORY USER RULES (each one MUST be followed):
═══════════════════════════════════════════════════
{instructions}
═══════════════════════════════════════════════════

AVAILABLE CONTENT INVENTORY:
{inventory_text}

CURRENT SELECTION (to verify):
```json
{current_json}
```

YOUR TASK:
1. Read EVERY mandatory rule above, one by one.
2. For each rule, check if the current selection complies.
3. Pay special attention to:
   - Conditional rules: Does the condition apply to "{role}" at "{company}"?
   - "Always include X" rules: Is X actually in the selection?
   - "Do not include Y" rules: Is Y absent from the selection?
   - Ordering rules: Is the order correct?
   - Skill category rules: Are the right categories included/excluded?
4. If ALL rules are satisfied, return the selection JSON unchanged.
5. If ANY rule is violated, fix the JSON and return the corrected version.

IMPORTANT:
- Only use EXACT names from the CONTENT INVENTORY.
- Keep the same JSON structure as the input.
- If you add items, they MUST exist in the inventory.
- Preserve all existing correct selections — only fix violations.

Return ONLY the (possibly corrected) JSON. No commentary, no markdown fences."""

    verify_system = (
        "You are a compliance checker. Return valid JSON only. "
        "Fix any rule violations in the selection. "
        "If no violations, return the input JSON unchanged."
    )

    try:
        raw = complete(verify_prompt, system_prompt=verify_system)
        cleaned = _clean_json_response(raw)
        verified = json.loads(cleaned)

        if not isinstance(verified, dict) or "sections_to_include" not in verified:
            logger.warning(
                "Verification LLM returned invalid structure — keeping original"
            )
            return selection

        # Re-validate the verified selection
        verified = _validate_selection(verified, inventory)
        verified = _enforce_instructions(verified, instructions, inventory, requirements)

        # Log any changes
        orig_projects = selection.get("project_order", [])
        new_projects = verified.get("project_order", [])
        if orig_projects != new_projects:
            logger.info(
                "Verification changed project_order: %s → %s",
                orig_projects, new_projects,
            )

        orig_leadership = selection.get("sections_to_include", {}).get("LEADERSHIP", [])
        new_leadership = verified.get("sections_to_include", {}).get("LEADERSHIP", [])
        if orig_leadership != new_leadership:
            logger.info(
                "Verification changed LEADERSHIP: %s → %s",
                orig_leadership, new_leadership,
            )

        orig_skills = selection.get("skill_categories_to_include", [])
        new_skills = verified.get("skill_categories_to_include", [])
        if orig_skills != new_skills:
            logger.info(
                "Verification changed skill_categories: %s → %s",
                orig_skills, new_skills,
            )

        return verified

    except (json.JSONDecodeError, Exception) as exc:
        logger.warning(
            "Verification LLM call failed (%s) — keeping original selection",
            exc,
        )
        return selection


def select_content(
    inventory: dict[str, dict],
    inventory_text: str,
    requirements: dict,
    instructions: str,
    template_sections: list[str],
    jd_text: str = "",
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
        jd_text=jd_text,
    )

    # Try up to 3 times — LLMs occasionally return malformed JSON or get rate-limited
    import time as _time
    last_error: Exception | None = None
    for attempt in range(1, 4):
        logger.info("LLM selection call (attempt %d/3)", attempt)
        if attempt > 1:
            # Wait between retries to avoid rate limits (exponential backoff)
            delay = 5 * (attempt - 1)
            logger.info("Waiting %ds before retry (rate limit backoff)", delay)
            _time.sleep(delay)

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

        # Verification pass: ask LLM to check compliance with every rule
        # (runs _validate + _enforce internally as a first pass)
        validated = _verify_and_fix_selection(
            selection=validated,
            instructions=instructions,
            requirements=requirements,
            inventory=inventory,
            inventory_text=inventory_text,
            jd_text=jd_text,
        )

        # FINAL enforcement: regex-based rules get the last word.
        # This catches anything the verification LLM missed or undid.
        validated = _enforce_instructions(validated, instructions, inventory, requirements)

        logger.info(
            "LLM selection validated — sections: %s, projects: %s",
            {
                k: (v if isinstance(v, bool) else f"{len(v)} items")
                for k, v in validated["sections_to_include"].items()
            },
            validated.get("project_order", []),
        )

        return validated

    # All attempts failed
    raise RuntimeError(
        f"LLM failed to return valid selection JSON after 3 attempts. "
        f"Last error: {last_error}"
    )
