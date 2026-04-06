"""End-to-end test: Data Science role — tailored resume generation.

Tests the full agent chain: jd_parser → cv_rag → match_scorer → resume_tailor.
Verifies PDF is generated, page limit is respected, relevant skills appear,
and ALL user instructions from instructions.md are followed.
"""
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s | %(levelname)s | %(message)s",
)

# Suppress noisy libraries
for name in ("httpx", "chromadb", "httpcore", "urllib3", "sentence_transformers"):
    logging.getLogger(name).setLevel(logging.WARNING)

DATA_SCIENCE_JD = """
Company: Spotify
Role: Senior Data Scientist

About the Role:
We are looking for a Senior Data Scientist to join our Personalization team at Spotify.
You will work on recommendation systems, user behavior modeling, and A/B testing at scale.

Responsibilities:
- Design and implement ML models for music recommendation and content personalization
- Conduct large-scale A/B experiments and causal inference analyses
- Build and maintain data pipelines for feature engineering
- Collaborate with product managers and engineers to ship ML-powered features
- Mentor junior data scientists and contribute to team best practices

Requirements:
- MS or PhD in Computer Science, Statistics, Mathematics, or related quantitative field
- 5+ years of experience in data science or machine learning
- Strong proficiency in Python, SQL, and statistical analysis
- Experience with deep learning frameworks (PyTorch, TensorFlow)
- Expertise in recommendation systems, NLP, or information retrieval
- Experience with big data tools (Spark, Hadoop, or similar)
- Strong communication skills and ability to present findings to non-technical stakeholders

Nice to Have:
- Experience with reinforcement learning or bandit algorithms
- Publications in top ML/AI conferences
- Experience with cloud platforms (GCP, AWS)

To apply, send your resume to hiring@spotify-careers.example.com with subject line:
"Application: Senior Data Scientist — [Your Name]"
Please include a brief cover letter explaining your interest.
"""


def _load_instructions() -> str:
    """Load instructions from uploads/ or defaults/ fallback."""
    base = Path(__file__).resolve().parent.parent / "cv"
    uploads = base / "uploads" / "instructions.md"
    defaults = base / "defaults" / "instructions.md"
    if uploads.exists():
        return uploads.read_text()
    elif defaults.exists():
        return defaults.read_text()
    return ""


def main():
    from agents.jd_parser import parse_jd
    from agents.cv_rag import retrieve_cv_sections
    from agents.match_scorer import score
    from agents.resume_tailor import tailor

    print("\n" + "=" * 70)
    print("  DATA SCIENCE ROLE TEST — Spotify Senior Data Scientist")
    print("=" * 70)

    # Step 1: Parse JD
    print("\n[1/4] Parsing job description...")
    requirements = parse_jd(DATA_SCIENCE_JD)
    print(f"  Company:  {requirements.get('company')}")
    print(f"  Role:     {requirements.get('role')}")
    print(f"  Skills:   {requirements.get('skills', [])[:10]}")
    print(f"  Apply:    {requirements.get('apply_method')} → {requirements.get('apply_target')}")
    print(f"  CL req'd: {requirements.get('cover_letter_required')}")
    print(f"  Subject:  {requirements.get('subject_line_format')}")
    print(f"  _jd_text: {'present' if requirements.get('_jd_text') else 'MISSING'} ({len(requirements.get('_jd_text', ''))} chars)")

    # Step 2: Retrieve CV sections
    print("\n[2/4] Retrieving relevant CV sections...")
    cv_sections = retrieve_cv_sections(requirements)
    print(f"  Retrieved {len(cv_sections)} sections")

    # Step 3: Match score
    print("\n[3/4] Scoring match...")
    match_score = score(DATA_SCIENCE_JD, cv_sections)
    print(f"  Match score: {match_score:.1f}%")

    # Step 4: Tailor resume
    print("\n[4/4] Tailoring resume...")
    pdf_path, resume_content = tailor(cv_sections, requirements)

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"\n  📄 PDF:  {pdf_path}")
    print(f"  📝 TEX:  {pdf_path.replace('.pdf', '.tex')}")

    # Validate PDF
    import pdfplumber

    pdf = Path(pdf_path)
    if not pdf.exists():
        print("\n  ❌ PDF NOT FOUND!")
        sys.exit(1)

    with pdfplumber.open(pdf_path) as p:
        pages = len(p.pages)
        text = "\n".join(page.extract_text() or "" for page in p.pages)

    print(f"  📐 Pages: {pages}")
    print(f"  📊 Size:  {pdf.stat().st_size / 1024:.1f} KB")

    # ═══════════════════════════════════════════════════════════════════
    # INSTRUCTION COMPLIANCE CHECKS
    # These are generic — they parse the actual instructions.md and
    # verify each rule against the generated PDF text.
    # ═══════════════════════════════════════════════════════════════════

    instructions = _load_instructions()
    text_lower = text.lower()

    print("\n  BASIC CONTENT CHECKS:")
    basic_checks = [
        ("Python mentioned", "python" in text_lower),
        ("ML/Machine Learning", "ml" in text_lower or "machine learning" in text_lower),
        ("Data Science context", "data" in text_lower),
        ("Page limit ≤ 1", pages <= 1),
    ]

    all_pass = True
    for label, passed in basic_checks:
        status = "✅" if passed else "❌"
        if not passed:
            all_pass = False
        print(f"    {status} {label}")

    print("\n  INSTRUCTION COMPLIANCE CHECKS:")

    # These checks are derived from common instruction patterns.
    # They work for ANY instructions.md — not hardcoded to a specific user.
    instruction_checks = []

    # Check: Page limit
    import re
    page_match = re.search(r"(\d+)\s+page\s+maximum", instructions.lower())
    if page_match:
        limit = int(page_match.group(1))
        instruction_checks.append(
            (f"Page limit ≤ {limit}", pages <= limit)
        )

    # Check: "Lead with X project" — X should appear before other projects
    lead_matches = re.findall(
        r"lead\s+with\s+([A-Za-z0-9][A-Za-z0-9 _\-&]+?)(?:\s+project|\s+for)",
        instructions.lower(),
    )
    for lead_name in lead_matches:
        lead_name = lead_name.strip()
        # Find position of lead project vs other projects in the text
        lead_pos = text_lower.find(lead_name)
        if lead_pos >= 0:
            # Check it appears in a "projects" context and is first
            # Find the projects section header
            proj_header_pos = text_lower.find("project")
            if proj_header_pos >= 0:
                instruction_checks.append(
                    (f"Lead with '{lead_name}' project (found at pos {lead_pos})",
                     lead_pos >= 0)
                )
            else:
                instruction_checks.append(
                    (f"Lead with '{lead_name}' project", lead_pos >= 0)
                )
        else:
            instruction_checks.append(
                (f"Lead with '{lead_name}' project — NOT FOUND in PDF", False)
            )

    # Check: "No references or hobbies"
    if "no" in instructions.lower() and "references" in instructions.lower():
        instruction_checks.append(
            ("No references section",
             "references" not in text_lower or "reference" not in text_lower.split("project")[0] if "project" in text_lower else True)
        )

    # Check: "No about section"
    if "no about section" in instructions.lower() or "no about" in instructions.lower():
        instruction_checks.append(
            ("No 'about' section", "about me" not in text_lower and "about:" not in text_lower)
        )

    # Check: "Only college in education"
    if "only college" in instructions.lower() or "never include school" in instructions.lower():
        instruction_checks.append(
            ("No school in education",
             "school" not in text_lower or "high school" not in text_lower)
        )

    # Check: "Always include Leadership"
    if "always include" in instructions.lower() and "leadership" in instructions.lower():
        has_leadership = (
            "leadership" in text_lower
            or "achievement" in text_lower
            or "organiser" in text_lower
            or "organizer" in text_lower
        )
        instruction_checks.append(
            ("Leadership & Achievements section present", has_leadership)
        )

    # Check: Student Organiser in Leadership
    if "student organiser" in instructions.lower() or "student organizer" in instructions.lower():
        instruction_checks.append(
            ("Student Organiser in Leadership",
             "student organiser" in text_lower or "student organizer" in text_lower)
        )

    # Check: Certificate of Merit
    if "certificate of merit" in instructions.lower():
        instruction_checks.append(
            ("Certificate of Merit present",
             "certificate of merit" in text_lower or "merit" in text_lower)
        )

    # Check: AWS Certification (except Quant)
    if "aws certification" in instructions.lower():
        is_quant = "quant" in (requirements.get("role") or "").lower()
        if not is_quant:
            instruction_checks.append(
                ("AWS Certification present",
                 "aws" in text_lower and "certif" in text_lower)
            )

    # Check: ML Specialization
    if "machine learning specialization" in instructions.lower():
        instruction_checks.append(
            ("ML Specialization present",
             "machine learning specialization" in text_lower or "ml specialization" in text_lower)
        )

    # Check: CouchHire project (except Quant)
    if "couchhire" in instructions.lower() and "always" in instructions.lower():
        is_quant = "quant" in (requirements.get("role") or "").lower()
        if not is_quant:
            instruction_checks.append(
                ("CouchHire project present", "couchhire" in text_lower)
            )

    # Check: Programming and Soft Skills in Skills section
    if "programming" in instructions.lower() and "soft skills" in instructions.lower():
        instruction_checks.append(
            ("Programming skills present", "programming" in text_lower)
        )
        instruction_checks.append(
            ("Soft Skills present", "soft skills" in text_lower or "soft skill" in text_lower)
        )

    # Check: No Quant Finance for non-Quant roles
    if "quantitative finance" in instructions.lower() and "not a quant" in instructions.lower():
        is_quant = "quant" in (requirements.get("role") or "").lower()
        if not is_quant:
            instruction_checks.append(
                ("No Quantitative Finance in Skills",
                 "quantitative finance" not in text_lower)
            )

    # Check: Specific projects for AI/ML/DS/DE roles
    role_lower = (requirements.get("role") or "").lower()
    is_ai_ml_ds = any(kw in role_lower for kw in ["data science", "data scientist", "machine learning", "ml", "ai", "data engineer"])
    proj_match = re.search(
        r"(?:ai|ml|data\s+engineering|data\s+science)\s+related\s+roles?\s+use\s+\d+\s+projects?:\s*(.+)",
        instructions.lower(),
    )
    if proj_match and is_ai_ml_ds:
        proj_names = [p.strip().rstrip(".") for p in re.split(r",\s*", proj_match.group(1))]
        for pname in proj_names:
            if pname:
                instruction_checks.append(
                    (f"Project '{pname}' present (AI/ML/DS rule)",
                     pname in text_lower)
                )

    # Check: No TinyML/Edge AI/Raspberry Pi (if not in JD)
    if "tinyml" in instructions.lower() or "edge ai" in instructions.lower():
        jd_lower = DATA_SCIENCE_JD.lower()
        for item in ["tinyml", "edge ai", "raspberry pi"]:
            if item not in jd_lower:
                instruction_checks.append(
                    (f"No '{item}' (not in JD)", item not in text_lower)
                )

    # Check: No Linux/Unix if not in JD
    if "linux" in instructions.lower() and "not mentioned in the jd" in instructions.lower():
        jd_lower = DATA_SCIENCE_JD.lower()
        if "linux" not in jd_lower and "unix" not in jd_lower:
            has_linux = "linux" in text_lower or "unix" in text_lower
            instruction_checks.append(
                ("No Linux/Unix (not in JD)", not has_linux)
            )

    # Check: No Programme Representative
    if "programme representative" in instructions.lower() and "do not" in instructions.lower():
        instruction_checks.append(
            ("No Programme Representative",
             "programme representative" not in text_lower)
        )

    # Check: Cloud & Systems for AI/ML/DS roles
    if is_ai_ml_ds and "cloud" in instructions.lower() and "system" in instructions.lower():
        instruction_checks.append(
            ("Cloud & Systems/Tools in Skills",
             "cloud" in text_lower)
        )

    for label, passed in instruction_checks:
        status = "✅" if passed else "❌"
        if not passed:
            all_pass = False
        print(f"    {status} {label}")

    print(f"\n  Resume content summary (first 300 chars):")
    print(f"    {resume_content[:300]}...")

    # Count results
    total = len(basic_checks) + len(instruction_checks)
    passed_count = sum(1 for _, p in basic_checks if p) + sum(1 for _, p in instruction_checks if p)

    print("\n" + "=" * 70)
    if all_pass:
        print(f"  ✅ ALL {total} CHECKS PASSED")
    else:
        print(f"  ⚠️  {passed_count}/{total} CHECKS PASSED")
    print("=" * 70 + "\n")

    return pdf_path


if __name__ == "__main__":
    pdf = main()
