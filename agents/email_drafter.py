"""Draft the application email — subject line and body.

Subject construction is deterministic (no LLM call). Body is generated via
llm/client.py, grounded in the tailored resume content so it references
specific projects and skills rather than being generic.
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
    resume_content: str,
) -> str:
    """Build the LLM prompt for email body generation."""
    from config import GITHUB_URL, APPLICANT_NAME, APPLICANT_EMAIL, APPLICANT_PHONE

    company = requirements.get("company") or "the company"
    role = requirements.get("role") or "the role"
    skills = requirements.get("skills", [])
    email_instructions: str | None = requirements.get("email_instructions")

    skills_str = ", ".join(skills) if skills else "not specified"
    has_cover_letter = bool(cover_letter_text and cover_letter_text.strip())

    # Build signature
    sig_parts = [APPLICANT_NAME]
    if APPLICANT_EMAIL:
        sig_parts.append(APPLICANT_EMAIL)
    if APPLICANT_PHONE:
        sig_parts.append(APPLICANT_PHONE)
    signature = " | ".join(sig_parts)

    # ── Example email for tone and structure ──
    example = (
        "Here is an EXAMPLE of the exact tone and structure I want "
        "(do NOT copy this content — use the candidate's actual resume context below):\n\n"
        "---\n"
        "Hi,\n\n"
        "I'm applying for the AI Research Intern role at Cloud First Technologies.\n\n"
        "I'm a final-year M.Sc. student in Computational Statistics and Data Analytics "
        "at VIT Vellore (CGPA: 9.23). My GitHub (github.com/RaySatish) has the best "
        "summary of what I've built — highlights include a Maritime Situational Awareness "
        "system using OCR + RAG pipelines, a NIFTY50 portfolio optimizer combining LSTM "
        "forecasting with FinBERT sentiment analysis, and a TinyML wake word detection "
        "model deployed on embedded hardware with ~70% size reduction through quantization.\n\n"
        "I have strong foundations in deep learning, LLMs, RAG, NLP, and Python, and I've "
        "done independent research at IIIT Allahabad. I'm comfortable experimenting with "
        "models and working in an open-ended research environment.\n\n"
        "Looking forward to your reply. Resume attached.\n\n"
        f"{signature}\n"
        "---"
    )

    # ── Main prompt ──
    lines: list[str] = [
        f"Write an application email body for the role of {role} at {company}.",
        "",
        example,
        "",
        "NOW write the email using THIS candidate's actual context:",
        "",
        f"Candidate name: {APPLICANT_NAME}",
        f"Target role: {role} at {company}",
        f"Key skills from JD: {skills_str}",
        "",
        "RESUME CONTEXT (what's in the tailored resume being attached — "
        "reference these specific projects and skills, don't make things up):",
        resume_content if resume_content else "(no resume context available)",
        "",
    ]

    if has_cover_letter:
        lines.append(
            "A cover letter is also attached, so keep the email brief. "
            "Don't repeat what the cover letter says — just introduce yourself, "
            "mention 1-2 highlights from the resume, and note that the cover letter "
            "and resume are attached."
        )
    else:
        lines.append(
            "No cover letter is attached, so the email carries the weight. "
            "Structure it like the example above:\n"
            "- Open with 'Hi,' then a line saying you're applying for the specific role.\n"
            "- A paragraph about your background — mention your education briefly, "
            "then reference your GitHub and describe 2-3 specific projects from the "
            "RESUME CONTEXT above. Be brief about each — one line per project, "
            "mentioning the tech used and what it does. Use dashes or commas to chain them, "
            "not bullet points.\n"
            "- A short paragraph about your core strengths relevant to this role "
            "and why you're a good fit.\n"
            "- Close with a simple line like 'Looking forward to your reply. Resume attached.'"
        )

    # Honour JD-specified email instructions
    if email_instructions:
        lines.append(
            f"\nIMPORTANT — the job description requires: {email_instructions}. "
            "You MUST address this explicitly in the email body."
        )

    # Signature and format rules
    lines.append(
        f"\nThe email MUST end with exactly this signature block "
        f"(separated from the body by one blank line):\n"
        f"{signature}"
    )

    if GITHUB_URL:
        lines.append(
            f"\nMention the GitHub URL ({GITHUB_URL}) naturally within the email body "
            f"(e.g. 'My GitHub (github.com/RaySatish) has...'). "
            f"Do NOT put it as a separate line at the end."
        )

    lines.append(
        "\nRULES:\n"
        "- Output ONLY the email body text, nothing else\n"
        "- Plain text only — no markdown, no bullet points, no headers, no asterisks, no bold\n"
        "- Start with 'Hi,' on its own line\n"
        "- No formal salutation (no 'Dear Hiring Manager')\n"
        "- No formal sign-off (no 'Sincerely', 'Best regards', 'Warm regards') — "
        "just the closing line then the signature\n"
        "- Write in first person\n"
        "- Sound like a real person wrote this — conversational, confident, not corporate\n"
        "- Reference SPECIFIC projects and tech from the RESUME CONTEXT — "
        "do not be vague or generic\n"
        "- Keep it between 100 and 180 words (including signature)\n"
        "- Do NOT use any asterisks (*) anywhere in the output"
    )

    return "\n".join(lines)


def draft(
    requirements: dict,
    cover_letter_text: str,
    resume_pdf_path: str,
    resume_content: str = "",
) -> tuple[str, str]:
    """Draft the application email subject and body.

    Returns a (subject, body) tuple. Subject is deterministic; body is
    LLM-generated, grounded in resume_content for specificity.
    """
    from llm.client import complete

    company = requirements.get("company") or "the company"
    role = requirements.get("role") or "the role"

    # ── Subject (deterministic) ──────────────────────────────────────
    subject = _build_subject(requirements)

    # ── Body (LLM-generated) ─────────────────────────────────────────
    logger.info(
        "Generating email body for '%s' at '%s'", role, company
    )

    prompt = _build_body_prompt(
        requirements, cover_letter_text, resume_content
    )

    system_prompt = (
        "You are writing a job application email on behalf of a real person. "
        "Write like a human — casual-professional, not corporate or stiff. "
        "Think of how a confident student would email about a job they're excited about. "
        "No buzzwords, no filler phrases like 'I am writing to express my interest'. "
        "Be specific — reference actual projects and tech, not vague claims. "
        "Plain text only. No markdown. No asterisks. No bold. No bullet points. "
        "Output only the email body."
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

    # ── Clean up ─────────────────────────────────────────────────────
    body = raw_body.strip()

    # Remove code fences if the LLM wraps output
    if body.startswith("```"):
        body_lines = body.split("\n")
        if body_lines[-1].strip() == "```":
            body_lines = body_lines[1:-1]
        elif body_lines[0].startswith("```"):
            body_lines = body_lines[1:]
        body = "\n".join(body_lines).strip()

    # Strip any remaining asterisks (bold/italic markup)
    body = body.replace("**", "").replace("*", "")

    # Ensure signature is present
    from config import APPLICANT_NAME, APPLICANT_EMAIL, APPLICANT_PHONE
    sig_parts = [APPLICANT_NAME]
    if APPLICANT_EMAIL:
        sig_parts.append(APPLICANT_EMAIL)
    if APPLICANT_PHONE:
        sig_parts.append(APPLICANT_PHONE)
    signature = " | ".join(sig_parts)

    if signature not in body:
        body = body.rstrip() + "\n\n" + signature

    word_count = len(body.split())
    logger.info(
        "Email body generated — %d words, %d chars", word_count, len(body)
    )

    return subject, body
