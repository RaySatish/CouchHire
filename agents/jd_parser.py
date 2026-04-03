"""Parse a raw job description into a structured requirements dict.

Uses the LLM (via llm/client.py) for semantic extraction and
nlp/ner_model.py for complementary spaCy-based skill extraction.
Results are merged and deduplicated.
"""

from __future__ import annotations

import json
import logging
import re

from llm.client import complete
from nlp.ner_model import extract_skills

logger = logging.getLogger(__name__)

# ── Default requirements dict ────────────────────────────────────────────
# Returned as-is when the LLM response cannot be parsed.
_DEFAULT_REQUIREMENTS: dict = {
    "company": None,
    "role": None,
    "skills": [],
    "apply_method": "unknown",
    "apply_target": None,
    "cover_letter_required": False,
    "subject_line_format": None,
    "email_instructions": None,
    "github_requested": False,
    "form_fields": [],
}

_VALID_APPLY_METHODS = {"email", "url", "unknown"}

# ── LLM prompt ───────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a job-description parser. Given a raw job description, extract \
structured information and return ONLY a valid JSON object — no markdown \
fences, no preamble, no explanation.

Return exactly this JSON schema:
{
  "company": "<company name or null>",
  "role": "<job title or null>",
  "skills": ["<skill1>", "<skill2>", ...],
  "apply_method": "<email | url | unknown>",
  "apply_target": "<email address or application URL or null>",
  "cover_letter_required": <true | false>,
  "subject_line_format": "<required subject format or null>",
  "email_instructions": "<specific instructions about what to include in the email or null>",
  "github_requested": <true | false>,
  "form_fields": ["<field1>", "<field2>", ...]
}

Rules:
- "apply_method": "email" if the JD says to email an address. "url" if it \
provides an application link (Greenhouse, Lever, Workday, Ashby, etc.). \
"unknown" if neither is found.
- "apply_target": the email address (for email) or URL (for url). null if unknown.
- "cover_letter_required": true only if the JD explicitly asks for a cover letter.
- "subject_line_format": the exact subject line format if specified. null otherwise.
- "email_instructions": any specific instructions about what to include in the \
application email (e.g. "mention salary expectations", "include portfolio link"). \
null if none.
- "github_requested": true if the JD asks for a GitHub profile/link.
- "form_fields": list of form fields mentioned (e.g. "LinkedIn URL", "visa status"). \
Empty list if none.
- "skills": technical skills, tools, frameworks, languages, and platforms mentioned. \
Be specific (e.g. "React" not "frontend framework").
- Return ONLY the JSON object. No other text."""


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers if present."""
    stripped = text.strip()
    # Match ```json\n...\n``` or ```\n...\n```
    match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", stripped, re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


def _merge_skills(llm_skills: list[str], ner_skills: list[str]) -> list[str]:
    """Merge LLM-extracted and NER-extracted skills, deduplicating case-insensitively."""
    seen: set[str] = set()
    merged: list[str] = []

    for skill in llm_skills:
        key = skill.strip().lower()
        if key and key not in seen:
            seen.add(key)
            merged.append(skill.strip())

    for skill in ner_skills:
        key = skill.strip().lower()
        if key and key not in seen:
            seen.add(key)
            merged.append(skill.strip())

    return merged


def _validate_and_normalise(raw: dict) -> dict:
    """Ensure the parsed dict has all required keys with correct types."""
    result = dict(_DEFAULT_REQUIREMENTS)  # start from defaults

    # company / role — str or None
    for key in ("company", "role", "subject_line_format", "email_instructions", "apply_target"):
        val = raw.get(key)
        result[key] = str(val) if val is not None else None

    # skills — list of strings
    raw_skills = raw.get("skills", [])
    if isinstance(raw_skills, list):
        result["skills"] = [str(s) for s in raw_skills if s]
    else:
        result["skills"] = []

    # apply_method — must be one of the valid values
    method = raw.get("apply_method", "unknown")
    result["apply_method"] = method if method in _VALID_APPLY_METHODS else "unknown"

    # booleans
    for key in ("cover_letter_required", "github_requested"):
        result[key] = bool(raw.get(key, False))

    # form_fields — list of strings
    raw_fields = raw.get("form_fields", [])
    if isinstance(raw_fields, list):
        result["form_fields"] = [str(f) for f in raw_fields if f]
    else:
        result["form_fields"] = []

    return result



def _filter_noise_skills(
    skills: list[str],
    company: str | None,
    role: str | None,
) -> list[str]:
    """Remove skills that are clearly not technical — role/company fragments, generic words."""
    # Build a set of words from company and role to filter against
    noise_words: set[str] = set()
    for text in (company, role):
        if text:
            for word in text.lower().split():
                noise_words.add(word)
            noise_words.add(text.lower())

    # Generic non-skill terms that slip through NER
    _extra_noise: set[str] = {
        "data", "scientist", "ml", "ai", "software", "quant",
        "stealth", "acme", "at", "needed", "cover", "letter",
        "cover letter", "resume", "cv", "application", "apply",
    }
    noise_words.update(_extra_noise)

    # First pass: collect candidates
    candidates: list[str] = []
    for skill in skills:
        low = skill.lower().strip()
        # Skip if it exactly matches a noise word
        if low in noise_words:
            continue
        # Skip if every word in the skill is a noise word (e.g. "Acme Corp Apply")
        words = low.split()
        if all(w in noise_words for w in words):
            continue
        # Skip single-char tokens
        if len(low) < 2:
            continue
        candidates.append(skill)

    # Second pass: remove single-word skills that are substrings of a
    # multi-word skill already in the list (e.g. "machine" when "machine learning" exists)
    multi_word = {c.lower() for c in candidates if " " in c}
    filtered: list[str] = []
    for skill in candidates:
        low = skill.lower()
        if " " not in low and any(low in mw for mw in multi_word):
            continue
        filtered.append(skill)

    return filtered

def parse_jd(jd_text: str) -> dict:
    """Parse a raw job description into a structured requirements dict.

    Returns a dict with keys: company, role, skills, apply_method,
    apply_target, cover_letter_required, subject_line_format,
    email_instructions, github_requested, form_fields.
    """
    if not jd_text or not jd_text.strip():
        logger.warning("Empty JD text received — returning defaults")
        return dict(_DEFAULT_REQUIREMENTS)

    # --- Step 1: LLM extraction ---
    logger.info("Parsing JD via LLM — %d chars", len(jd_text))
    try:
        raw_response = complete(jd_text, system_prompt=_SYSTEM_PROMPT)
    except Exception:
        logger.error("LLM call failed during JD parsing", exc_info=True)
        raise

    # --- Step 2: Parse LLM JSON response ---
    cleaned = _strip_markdown_fences(raw_response)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error(
            "Failed to parse LLM response as JSON. Raw response:\n%s",
            raw_response,
        )
        return dict(_DEFAULT_REQUIREMENTS)

    if not isinstance(parsed, dict):
        logger.error(
            "LLM response parsed but is not a dict (type=%s). Raw:\n%s",
            type(parsed).__name__,
            raw_response,
        )
        return dict(_DEFAULT_REQUIREMENTS)

    # --- Step 3: Validate and normalise ---
    requirements = _validate_and_normalise(parsed)

    # --- Step 4: NER skill extraction + merge ---
    ner_skills = extract_skills(jd_text)
    logger.info(
        "Skills — LLM: %d, NER: %d",
        len(requirements["skills"]),
        len(ner_skills),
    )
    requirements["skills"] = _merge_skills(requirements["skills"], ner_skills)

    # --- Step 5: Filter noise (role/company fragments, generic words) ---
    requirements["skills"] = _filter_noise_skills(
        requirements["skills"],
        requirements.get("company"),
        requirements.get("role"),
    )
    logger.info(
        "Final skills: %d — %s",
        len(requirements["skills"]),
        requirements["skills"],
    )

    return requirements
