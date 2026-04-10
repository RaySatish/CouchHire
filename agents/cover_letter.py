"""Generate a cover letter that complements the tailored resume.

Produces a 3-paragraph plain-text cover letter whose tone adapts to the
role type. Uses resume_content (the structured summary from resume_tailor)
to ensure the cover letter extends — never repeats — what the resume covers.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ── Tone keywords ────────────────────────────────────────────────────────
_QUANT_KEYWORDS = {"quant", "trading", "finance"}
_ML_KEYWORDS = {"ml", "machine learning", "ai", "llm"}


def _detect_tone(role: str) -> str:
    """Determine the cover letter tone from the role title.

    Returns one of: 'quant', 'ml', or 'professional'.
    """
    role_lower = role.lower()
    if any(kw in role_lower for kw in _QUANT_KEYWORDS):
        return "quant"
    if any(kw in role_lower for kw in _ML_KEYWORDS):
        return "ml"
    return "professional"


def _tone_instruction(tone: str) -> str:
    """Return the tone-specific writing instruction for the LLM prompt."""
    if tone == "quant":
        return (
            "Tone: analytical, precise, and numbers-forward. Lead with "
            "quantitative impact. Use concise, data-driven language — the "
            "reader values rigour and measurable results over narrative flourish."
        )
    if tone == "ml":
        return (
            "Tone: technically deep with a research angle. Demonstrate "
            "understanding of methodology and first-principles thinking. "
            "Reference specific techniques, architectures, or research "
            "contributions where relevant."
        )
    return (
        "Tone: confident and direct, but human. Not a list of achievements — "
        "a person explaining why they are the right fit. Lead with fit, "
        "not credentials. Sound like someone who has thought about this role "
        "specifically, not someone mass-applying."
    )


def _build_prompt(
    requirements: dict,
    cv_sections: list[str],
    resume_content: str,
    tone: str,
) -> str:
    """Build the LLM prompt for cover letter generation."""
    company = requirements.get("company", "the company")
    role = requirements.get("role", "the role")
    skills = requirements.get("skills", [])

    skills_str = ", ".join(skills) if skills else "not specified"
    cv_context = "\n".join(cv_sections) if cv_sections else "No CV sections provided."

    return f"""Write a cover letter for the role of {role} at {company}.

CANDIDATE'S CV DATA:
{cv_context}

RESUME SUMMARY (what the tailored resume already covers — DO NOT repeat this):
{resume_content}

REQUIRED SKILLS: {skills_str}

{_tone_instruction(tone)}

STRUCTURE — exactly 3 paragraphs, no headers, no salutation, no sign-off:

Paragraph 1 — Why {company} and this {role} specifically.
Be specific about what draws the candidate to this company and role.
Do NOT use generic phrases like "I am excited to apply" or "I am writing to express my interest".
Reference something concrete about the company or role requirements.
This paragraph is about THEM, not the candidate.

Paragraph 2 — Why the candidate fits.
This is NOT a project description. The resume already lists what was built.
This paragraph answers: "What does this person's background tell you about how they think and work?"
Take the strongest project or experience from the resume summary and explain:
  - What problem it was actually solving (not the tech used)
  - What it demonstrates about how the candidate approaches hard problems
  - Why that maps directly to what {company} needs for this {role}
One concrete example, extended into a point about fit. Not a list. Not a stack.
Do NOT re-list bullets, tech stacks, or percentages from the resume.
Write in clear, concise sentences. Maximum 2 clauses per sentence.
Do not chain sentences with 'and'. Each sentence should stand alone.

Paragraph 3 — Close.
2-3 short sentences. Express genuine interest without desperation.
End with a clean, direct call to action as its own sentence.

RULES:
- Output ONLY the 3 paragraphs as plain text
- No markdown, no headers, no "Dear Hiring Manager", no "Sincerely"
- CRITICAL: Only reference projects, skills, and experiences explicitly listed in the RESUME SUMMARY above. Do not add, invent, or imply anything else.
- The resume covers what was built. The cover letter covers what it says about the candidate.
- Sell the person, not the project list. Show fit, not just ability.
- Same voice as the resume — written by the same person, for the same application
- Keep total length between 150 and 300 words
- Separate paragraphs with a single blank line"""


def generate(
    requirements: dict, cv_sections: list[str], resume_content: str
) -> str:
    """Generate a cover letter that complements the tailored resume.

    Returns a plain-text cover letter (3 paragraphs) or an empty string
    if cover_letter_required is False.
    """
    if not requirements.get("cover_letter_required", False):
        logger.info("Cover letter not required — skipping generation.")
        return ""

    from llm.client import complete

    company = requirements.get("company", "the company")
    role = requirements.get("role", "the role")

    tone = _detect_tone(role)
    logger.info(
        "Generating cover letter for '%s' at '%s' (tone=%s)",
        role,
        company,
        tone,
    )

    prompt = _build_prompt(requirements, cv_sections, resume_content, tone)

    system_prompt = (
        "You are a professional cover letter writer. "
        "Your job is to sell the candidate as a person who fits this role — "
        "not to describe their projects. The resume already lists what they built. "
        "Your letter explains what kind of thinker and problem-solver they are, "
        "and why that maps to what this company needs. "
        "ABSOLUTE RULE: You may ONLY reference projects, skills, experiences, "
        "and achievements that are explicitly listed in the RESUME SUMMARY "
        "provided in the prompt. Do NOT invent, infer, or add any skill, "
        "technology, project detail, or achievement that is not word-for-word "
        "present in that summary. Treat the resume summary as the only source of truth. "
        "Output plain text only. No markdown, no headers, no salutations, no sign-offs."
    )

    try:
        raw_response = complete(prompt, system_prompt=system_prompt)
    except Exception:
        logger.error(
            "LLM call failed during cover letter generation for '%s' at '%s'",
            role,
            company,
            exc_info=True,
        )
        raise

    # Clean up: strip whitespace, remove any accidental markdown fencing
    cover_letter = raw_response.strip()
    # Remove code fences if the LLM wraps output
    if cover_letter.startswith("```"):
        lines = cover_letter.split("\n")
        # Drop first and last lines if they're fences
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        elif lines[0].startswith("```"):
            lines = lines[1:]
        cover_letter = "\n".join(lines).strip()

    word_count = len(cover_letter.split())
    logger.info(
        "Cover letter generated — %d words, %d chars",
        word_count,
        len(cover_letter),
    )

    return cover_letter
