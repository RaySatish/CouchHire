"""Draft the application email — subject line and body.

Subject construction is deterministic (no LLM call). Body is generated via
llm/client.py, constrained to 100–150 words, plain text, professional tone.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _build_subject(requirements: dict) -> str:
    """Construct the email subject line deterministically.

    Uses the JD-specified format if present (substituting [Your Name]),
    otherwise falls back to 'Application for <role> — <APPLICANT_NAME>'.
    """
    from config import APPLICANT_NAME

    fmt: str | None = requirements.get("subject_line_format")
    if fmt is not None:
        subject = fmt.replace("[Your Name]", APPLICANT_NAME)
        logger.info("Subject from JD format: %s", subject)
        return subject

    role = requirements.get("role", "the role")
    subject = f"Application for {role} — {APPLICANT_NAME}"
    logger.info("Subject from default format: %s", subject)
    return subject


def _build_body_prompt(
    requirements: dict,
    cover_letter_text: str,
    resume_pdf_path: str,
) -> str:
    """Build the LLM prompt for email body generation."""
    from config import GITHUB_URL, APPLICANT_NAME

    company = requirements.get("company", "the company")
    role = requirements.get("role", "the role")
    skills = requirements.get("skills", [])
    email_instructions: str | None = requirements.get("email_instructions")

    skills_str = ", ".join(skills) if skills else "not specified"
    has_cover_letter = bool(cover_letter_text and cover_letter_text.strip())

    # Base context
    lines: list[str] = [
        f"Write an application email body for the role of {role} at {company}.",
        f"Candidate name: {APPLICANT_NAME}.",
        f"Key skills: {skills_str}.",
        f"A tailored resume PDF is attached to this email.",
    ]

    # Cover letter branch
    if has_cover_letter:
        lines.append(
            "A cover letter is also attached. Structure the email as follows:\n"
            "- Sentence 1: Introduce the candidate by name and express interest in the specific role at this company.\n"
            "- Sentence 2: Briefly highlight the candidate's strongest relevant skill and how it fits the role.\n"
            "- Sentence 3: Mention what specifically draws the candidate to this company.\n"
            "- Sentence 4: Note that a cover letter and tailored resume are attached for further detail.\n"
            "Write each sentence with enough substance to be meaningful — avoid terse, clipped phrasing."
        )
    else:
        lines.append(
            "No cover letter is attached, so the email body must carry more weight. "
            "Structure the email as follows:\n"
            f"- Sentence 1: Introduce the candidate by name and state interest in the {role} role at {company}.\n"
            f"- Sentence 2: Describe the candidate's strongest skill from ({skills_str}) and how it directly applies to this role. Be specific.\n"
            f"- Sentence 3: Describe a second relevant skill from ({skills_str}) and its practical value for the role.\n"
            "- Sentence 4: Mention what draws the candidate to this company specifically.\n"
            "- Sentence 5: Note the attached resume and invite further discussion.\n"
            "Write each sentence with enough detail to be substantive — avoid short, generic phrasing."
        )

    # Honour JD-specified email instructions
    if email_instructions:
        lines.append(
            f"IMPORTANT — the job description requires: {email_instructions}. "
            "You MUST address this explicitly in the email body as an additional sentence."
        )

    # Closing and format rules
    lines.append(
        f'The email body MUST end with exactly this line on its own: "GitHub: {GITHUB_URL}"'
    )
    lines.append(
        "RULES:\n"
        "- Output ONLY the email body text, nothing else\n"
        "- Plain text only — no markdown, no bullet points, no headers, no asterisks\n"
        "- No salutation (no 'Dear Hiring Manager') — start directly with content\n"
        "- No sign-off (no 'Sincerely', 'Best regards') — the GitHub line is the last line\n"
        "- Write in first person ('I', not 'He/She')\n"
        "- Professional but not stiff — conversational confidence\n"
        "- STRICT word count: the body (including the GitHub line) must be between 100 and 150 words\n"
        "- Separate the GitHub line from the rest of the body with one blank line"
    )

    return "\n\n".join(lines)


def draft(
    requirements: dict,
    cover_letter_text: str,
    resume_pdf_path: str,
) -> tuple[str, str]:
    """Draft the application email subject and body.

    Returns a (subject, body) tuple. Subject is deterministic; body is
    LLM-generated, 100–150 words, plain text, ending with the GitHub URL.
    """
    from llm.client import complete
    from config import GITHUB_URL

    company = requirements.get("company", "the company")
    role = requirements.get("role", "the role")

    # ── Subject (deterministic) ──────────────────────────────────────────
    subject = _build_subject(requirements)

    # ── Body (LLM-generated) ─────────────────────────────────────────────
    logger.info(
        "Generating email body for '%s' at '%s'", role, company
    )

    prompt = _build_body_prompt(requirements, cover_letter_text, resume_pdf_path)

    system_prompt = (
        "You are a professional email writer for job applications. "
        "You write concise, confident emails in plain text. "
        "No markdown, no bullet points, no headers, no asterisks. "
        "Always write in first person. "
        "Output only the email body text. "
        "Your output must be between 100 and 150 words — this is a hard constraint. "
        "Count your words before responding and ensure compliance."
    )

    try:
        raw_body = complete(prompt, system_prompt=system_prompt)
    except Exception:
        logger.error(
            "LLM call failed during email body generation for '%s' at '%s'",
            role,
            company,
            exc_info=True,
        )
        raise

    # ── Clean up ─────────────────────────────────────────────────────────
    body = raw_body.strip()

    # Remove code fences if the LLM wraps output
    if body.startswith("```"):
        lines = body.split("\n")
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        elif lines[0].startswith("```"):
            lines = lines[1:]
        body = "\n".join(lines).strip()

    # Ensure the body ends with the GitHub URL line
    github_line = f"GitHub: {GITHUB_URL}"
    if github_line not in body:
        body = body.rstrip() + "\n\n" + github_line

    word_count = len(body.split())
    logger.info(
        "Email body generated — %d words, %d chars", word_count, len(body)
    )

    return subject, body
