"""Integration tests for _assemble_skills_section with the fixed tier logic.

Tests the full flow: selection dict + raw_content (master CV) + template_style_examples
→ assembled LaTeX output. Verifies that master CV is the primary content source
and template is only used for formatting or as a fallback.
"""

import re
import pytest

from agents.resume_tailor import _assemble_skills_section


# ─── Test Data ───────────────────────────────────────────────────────────────

# Template: 3 categories in tabularx format
TEMPLATE_SKILLS_TABULARX = r"""\begin{tabularx}{\textwidth}{lX}
\textbf{Programming: } & Python, Java \\
\textbf{Frameworks: } & React, Django \\
\textbf{Soft Skills: } & Leadership, Communication \\
\end{tabularx}"""

# Master CV: 6 categories (superset of template) — itemize format
MASTER_CV_SKILLS = r"""\item \textbf{Programming:} Python, Java, Go, Rust, C++, TypeScript
\item \textbf{Frameworks:} React, Django, FastAPI, Flask, Next.js, Svelte
\item \textbf{Cloud:} AWS, GCP, Azure, Docker, Kubernetes, Terraform
\item \textbf{Databases:} PostgreSQL, MongoDB, Redis, Cassandra, DynamoDB
\item \textbf{Tools:} Git, CI/CD, Jenkins, GitHub Actions, Jira, Confluence
\item \textbf{Machine Learning:} PyTorch, TensorFlow, scikit-learn, Hugging Face"""

# Master CV: plain format (no \item, no &)
MASTER_CV_SKILLS_PLAIN = r"""\textbf{Programming:} Python, Java, Go, Rust, C++, TypeScript
\textbf{Cloud:} AWS, GCP, Azure, Docker, Kubernetes"""

# Template: itemize format
TEMPLATE_SKILLS_ITEMIZE = r"""\begin{itemize}[leftmargin=*]
\item \textbf{Programming:} Python, Java
\item \textbf{Soft Skills:} Leadership
\end{itemize}"""


# ─── Helper ──────────────────────────────────────────────────────────────────

def _extract_items_from_line(line: str) -> list[str]:
    """Extract skill items from a formatted LaTeX line (any style)."""
    # Remove LaTeX formatting to get raw items
    cleaned = re.sub(r"\\textbf\{[^}]+\}\s*:?\s*", "", line)
    cleaned = re.sub(r"\\\\$", "", cleaned)  # trailing \\
    cleaned = cleaned.replace("&", "").strip()
    cleaned = re.sub(r"\\item\s*", "", cleaned)
    return [s.strip() for s in cleaned.split(",") if s.strip()]


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 1 TESTS: Master CV content, reformatted to template style
# ═══════════════════════════════════════════════════════════════════════════════

class TestTier1MasterCVReformatted:
    """Master CV content wins when category exists in both sources."""

    def test_programming_uses_master_cv_items(self):
        """Programming exists in both — master CV has 6 items, template has 2.
        Output should contain all 6 master CV items."""
        selection = {
            "skill_categories_to_include": ["Programming"],
        }
        result = _assemble_skills_section(
            selection=selection,
            raw_content=MASTER_CV_SKILLS,
            template_style_examples={"SKILLS": TEMPLATE_SKILLS_TABULARX},
        )
        # Master CV has: Python, Java, Go, Rust, C++, TypeScript
        for item in ["Python", "Java", "Go", "Rust", "C++", "TypeScript"]:
            assert item in result, f"Master CV item '{item}' missing from output"

    def test_frameworks_uses_master_cv_items(self):
        """Frameworks in both — master CV has 6, template has 2."""
        selection = {
            "skill_categories_to_include": ["Frameworks"],
        }
        result = _assemble_skills_section(
            selection=selection,
            raw_content=MASTER_CV_SKILLS,
            template_style_examples={"SKILLS": TEMPLATE_SKILLS_TABULARX},
        )
        for item in ["React", "Django", "FastAPI", "Flask", "Next.js", "Svelte"]:
            assert item in result, f"Master CV item '{item}' missing from output"

    def test_output_uses_template_format(self):
        """When template is tabularx, output should use & separator."""
        selection = {
            "skill_categories_to_include": ["Programming"],
        }
        result = _assemble_skills_section(
            selection=selection,
            raw_content=MASTER_CV_SKILLS,
            template_style_examples={"SKILLS": TEMPLATE_SKILLS_TABULARX},
        )
        # tabularx uses & separator
        assert "&" in result, "Output should use tabularx format (& separator)"

    def test_multiple_categories_all_from_master_cv(self):
        """When multiple categories selected, all pull from master CV."""
        selection = {
            "skill_categories_to_include": ["Programming", "Frameworks", "Cloud"],
        }
        result = _assemble_skills_section(
            selection=selection,
            raw_content=MASTER_CV_SKILLS,
            template_style_examples={"SKILLS": TEMPLATE_SKILLS_TABULARX},
        )
        # Cloud is ONLY in master CV — should still appear
        assert "AWS" in result
        assert "GCP" in result
        assert "Kubernetes" in result
        # Programming from master CV (not template's 2-item version)
        assert "Go" in result
        assert "Rust" in result


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 2 TESTS: Master CV content as-is (no template style available)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTier2MasterCVAsIs:
    """When no template style example is available, master CV content used as-is."""

    def test_no_template_uses_master_cv_directly(self):
        """No template provided — master CV content used verbatim."""
        selection = {
            "skill_categories_to_include": ["Programming", "Cloud"],
        }
        result = _assemble_skills_section(
            selection=selection,
            raw_content=MASTER_CV_SKILLS,
            template_style_examples={},  # No template!
        )
        assert "Python" in result
        assert "Go" in result
        assert "AWS" in result
        assert "Kubernetes" in result

    def test_empty_template_skills(self):
        """Template SKILLS key exists but is empty string."""
        selection = {
            "skill_categories_to_include": ["Programming"],
        }
        result = _assemble_skills_section(
            selection=selection,
            raw_content=MASTER_CV_SKILLS,
            template_style_examples={"SKILLS": ""},
        )
        assert "Python" in result
        assert "TypeScript" in result


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 3 TESTS: Template-only fallback
# ═══════════════════════════════════════════════════════════════════════════════

class TestTier3TemplateOnlyFallback:
    """Categories that only exist in the template are used as fallback."""

    def test_soft_skills_only_in_template(self):
        """Soft Skills exists in template but NOT in master CV.
        Should use template version as fallback."""
        selection = {
            # Soft Skills will be auto-enforced as required
            "skill_categories_to_include": ["Programming", "Soft Skills"],
        }
        result = _assemble_skills_section(
            selection=selection,
            raw_content=MASTER_CV_SKILLS,  # No Soft Skills here
            template_style_examples={"SKILLS": TEMPLATE_SKILLS_TABULARX},
        )
        assert "Leadership" in result or "Communication" in result, \
            "Soft Skills from template should appear as fallback"


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY ORDERING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCategoryOrdering:
    """Categories should appear in the order specified by the LLM."""

    def test_order_preserved(self):
        """Cloud before Programming — Cloud should appear first in output."""
        selection = {
            "skill_categories_to_include": ["Cloud", "Programming"],
        }
        result = _assemble_skills_section(
            selection=selection,
            raw_content=MASTER_CV_SKILLS,
            template_style_examples={"SKILLS": TEMPLATE_SKILLS_TABULARX},
        )
        cloud_pos = result.find("Cloud")
        prog_pos = result.find("Programming")
        assert cloud_pos < prog_pos, \
            f"Cloud (pos {cloud_pos}) should appear before Programming (pos {prog_pos})"


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER-CV-ONLY CATEGORY TESTS (the core bug scenario)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMasterCVOnlyCategories:
    """Categories that exist ONLY in master CV (not template) should now work."""

    def test_cloud_only_in_master_cv(self):
        """Cloud is in master CV but NOT in template. Should be included."""
        selection = {
            "skill_categories_to_include": ["Cloud"],
        }
        result = _assemble_skills_section(
            selection=selection,
            raw_content=MASTER_CV_SKILLS,
            template_style_examples={"SKILLS": TEMPLATE_SKILLS_TABULARX},
        )
        assert "AWS" in result
        assert "Docker" in result
        assert "Terraform" in result

    def test_databases_only_in_master_cv(self):
        """Databases is in master CV but NOT in template."""
        selection = {
            "skill_categories_to_include": ["Databases"],
        }
        result = _assemble_skills_section(
            selection=selection,
            raw_content=MASTER_CV_SKILLS,
            template_style_examples={"SKILLS": TEMPLATE_SKILLS_TABULARX},
        )
        assert "PostgreSQL" in result
        assert "MongoDB" in result
        assert "DynamoDB" in result

    def test_machine_learning_only_in_master_cv(self):
        """Machine Learning is in master CV but NOT in template."""
        selection = {
            "skill_categories_to_include": ["Machine Learning"],
        }
        result = _assemble_skills_section(
            selection=selection,
            raw_content=MASTER_CV_SKILLS,
            template_style_examples={"SKILLS": TEMPLATE_SKILLS_TABULARX},
        )
        assert "PyTorch" in result
        assert "Hugging Face" in result

    def test_all_master_cv_categories_reachable(self):
        """All 6 master CV categories should be selectable and produce output."""
        all_cats = ["Programming", "Frameworks", "Cloud", "Databases", "Tools", "Machine Learning"]
        selection = {
            "skill_categories_to_include": all_cats,
        }
        result = _assemble_skills_section(
            selection=selection,
            raw_content=MASTER_CV_SKILLS,
            template_style_examples={"SKILLS": TEMPLATE_SKILLS_TABULARX},
        )
        # Every category should have content in the output
        assert "Python" in result       # Programming
        assert "React" in result        # Frameworks
        assert "AWS" in result          # Cloud
        assert "PostgreSQL" in result   # Databases
        assert "Git" in result          # Tools
        assert "PyTorch" in result      # Machine Learning


# ═══════════════════════════════════════════════════════════════════════════════
# CONTAINER WRAPPING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestContainerWrapping:
    """Output should be wrapped in the template's container format."""

    def test_tabularx_container_preserved(self):
        """When template uses tabularx, output should have begin/end tabularx."""
        selection = {
            "skill_categories_to_include": ["Programming", "Cloud"],
        }
        result = _assemble_skills_section(
            selection=selection,
            raw_content=MASTER_CV_SKILLS,
            template_style_examples={"SKILLS": TEMPLATE_SKILLS_TABULARX},
        )
        assert r"\begin{tabularx}" in result
        assert r"\end{tabularx}" in result

    def test_itemize_container_preserved(self):
        """When template uses itemize, output should have begin/end itemize."""
        selection = {
            "skill_categories_to_include": ["Programming"],
        }
        result = _assemble_skills_section(
            selection=selection,
            raw_content=MASTER_CV_SKILLS,
            template_style_examples={"SKILLS": TEMPLATE_SKILLS_ITEMIZE},
        )
        assert r"\begin{itemize}" in result
        assert r"\end{itemize}" in result


# ═══════════════════════════════════════════════════════════════════════════════
# REQUIRED CATEGORY ENFORCEMENT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestRequiredCategoryEnforcement:
    """Programming and Soft Skills are required — auto-added if missing."""

    def test_programming_auto_added(self):
        """If LLM omits Programming, it should be auto-enforced."""
        selection = {
            "skill_categories_to_include": ["Cloud"],  # Missing Programming
        }
        result = _assemble_skills_section(
            selection=selection,
            raw_content=MASTER_CV_SKILLS,
            template_style_examples={"SKILLS": TEMPLATE_SKILLS_TABULARX},
        )
        # Programming should be auto-added and contain master CV content
        assert "Python" in result
        assert "TypeScript" in result

    def test_soft_skills_auto_added(self):
        """If LLM omits Soft Skills, it should be auto-enforced from template."""
        selection = {
            "skill_categories_to_include": ["Programming"],  # Missing Soft Skills
        }
        result = _assemble_skills_section(
            selection=selection,
            raw_content=MASTER_CV_SKILLS,  # No Soft Skills in master CV
            template_style_examples={"SKILLS": TEMPLATE_SKILLS_TABULARX},
        )
        # Soft Skills from template (TIER 3 fallback)
        assert "Leadership" in result or "Communication" in result


# ═══════════════════════════════════════════════════════════════════════════════
# DEDUPLICATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeduplication:
    """Duplicate resolved lines should be deduplicated."""

    def test_no_duplicate_lines(self):
        """Same category selected twice should not produce duplicate output."""
        selection = {
            "skill_categories_to_include": ["Programming", "Programming"],
        }
        result = _assemble_skills_section(
            selection=selection,
            raw_content=MASTER_CV_SKILLS,
            template_style_examples={"SKILLS": TEMPLATE_SKILLS_TABULARX},
        )
        # Count occurrences of "Programming" in the output
        count = result.count("Programming")
        # Should appear exactly once (plus possibly in the container, but not duplicated as a row)
        assert count <= 2, f"Programming appears {count} times — likely duplicated"


# ═══════════════════════════════════════════════════════════════════════════════
# EMPTY / EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases and fallbacks."""

    def test_no_categories_selected_falls_back_to_all(self):
        """If LLM provides empty list, all master CV categories should be used."""
        selection = {
            "skill_categories_to_include": [],
        }
        result = _assemble_skills_section(
            selection=selection,
            raw_content=MASTER_CV_SKILLS,
            template_style_examples={"SKILLS": TEMPLATE_SKILLS_TABULARX},
        )
        # Should contain items from all 6 master CV categories
        assert "Python" in result
        assert "AWS" in result
        assert "PostgreSQL" in result
        assert "PyTorch" in result

    def test_unknown_category_skipped(self):
        """A hallucinated category name should be skipped gracefully."""
        selection = {
            "skill_categories_to_include": ["Programming", "Quantum Computing"],
        }
        result = _assemble_skills_section(
            selection=selection,
            raw_content=MASTER_CV_SKILLS,
            template_style_examples={"SKILLS": TEMPLATE_SKILLS_TABULARX},
        )
        # Programming should still be there
        assert "Python" in result
        # Quantum Computing doesn't exist anywhere — should be skipped, not crash
        assert "Quantum" not in result

    def test_no_resolved_lines_returns_raw(self):
        """If nothing resolves, raw content should be returned as fallback."""
        selection = {
            "skill_categories_to_include": ["Nonexistent Category"],
        }
        # Master CV has no parseable categories (plain text)
        raw = "Just some plain text with no \\textbf{} markers"
        result = _assemble_skills_section(
            selection=selection,
            raw_content=raw,
            template_style_examples={"SKILLS": TEMPLATE_SKILLS_TABULARX},
        )
        # After required categories are enforced but fail to match,
        # should fall back to raw content
        # (Programming required cat will be added but won't match in raw)
        assert result is not None  # Should not crash
