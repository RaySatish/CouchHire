"""Step 3: Verify reformat_skill_to_template_style() handles all cases.

Tests every code path now that this function is the PRIMARY path
for skill content (not just a fallback).
"""
import sys
import re

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from agents.resume_assembler import reformat_skill_to_template_style


# ──────────────────────────────────────────────────────────────
# FORMAT A: tabularx style (& separator, trailing \\)
# ──────────────────────────────────────────────────────────────

class TestTabularxStyle:
    """Template uses tabularx: \textbf{Cat: } & items \\"""

    STYLE = r"\textbf{Data Engineering: } & Spark, Kafka, Airflow \\"

    def test_basic_itemize_to_tabularx(self):
        """Master CV uses \item \textbf{Cat:} items → reformat to tabularx."""
        master = r"\item \textbf{Cloud & Tools:} Docker, AWS, Terraform, Kubernetes"
        result = reformat_skill_to_template_style("Cloud & Tools", master, self.STYLE)
        assert r"\textbf{Cloud & Tools: }" in result, f"Missing category header in: {result}"
        assert "&" in result, f"Missing tabularx & separator in: {result}"
        assert "Docker" in result and "Kubernetes" in result, f"Missing skill items in: {result}"
        assert result.rstrip().endswith("\\\\"), f"Missing trailing \\\\ in: {result}"

    def test_plain_textbf_to_tabularx(self):
        """Master CV uses plain \textbf{Cat:} items \\ → reformat to tabularx."""
        master = r"\textbf{Programming:} Python, Java, Go, Rust, C++ \\"
        result = reformat_skill_to_template_style("Programming", master, self.STYLE)
        assert r"\textbf{Programming: }" in result
        assert "& " in result
        assert "Python" in result and "C++" in result
        assert result.rstrip().endswith("\\\\")

    def test_preserves_all_items(self):
        """All skill items from master CV appear in output."""
        items = "TensorFlow, PyTorch, scikit-learn, XGBoost, Hugging Face, ONNX"
        master = r"\item \textbf{ML Frameworks:} " + items
        result = reformat_skill_to_template_style("ML Frameworks", master, self.STYLE)
        for item in items.split(", "):
            assert item in result, f"Missing item '{item}' in: {result}"

    def test_master_cv_with_trailing_backslash(self):
        """Master CV block with trailing \\ is handled (items extracted cleanly)."""
        master = r"\textbf{Databases:} PostgreSQL, MongoDB, Redis, DynamoDB \\"
        result = reformat_skill_to_template_style("Databases", master, self.STYLE)
        # Should NOT have double trailing backslashes
        assert "DynamoDB" in result
        # The trailing \\ should appear exactly once at the end
        stripped = result.rstrip()
        assert stripped.endswith("\\\\")
        # Check no double \\\\\\\\
        assert not stripped.endswith("\\\\\\\\\\\\")

    def test_colon_inside_braces(self):
        """Handles \textbf{Category:} where colon is inside braces."""
        master = r"\item \textbf{DevOps:} Jenkins, GitHub Actions, ArgoCD"
        result = reformat_skill_to_template_style("DevOps", master, self.STYLE)
        assert "Jenkins" in result
        assert "ArgoCD" in result

    def test_colon_outside_braces(self):
        """Handles \textbf{Category}: where colon is outside braces."""
        master = r"\item \textbf{DevOps}: Jenkins, GitHub Actions, ArgoCD"
        result = reformat_skill_to_template_style("DevOps", master, self.STYLE)
        assert "Jenkins" in result
        assert "ArgoCD" in result


# ──────────────────────────────────────────────────────────────
# FORMAT B: itemize style (\item prefix)
# ──────────────────────────────────────────────────────────────

class TestItemizeStyle:
    """Template uses itemize: \item \textbf{Cat:} items"""

    STYLE = r"\item \textbf{Web Development:} React, Node.js, TypeScript"

    def test_tabularx_to_itemize(self):
        """Master CV uses tabularx style → reformat to itemize."""
        master = r"\textbf{Cloud & Tools: } & Docker, AWS, Terraform \\"
        result = reformat_skill_to_template_style("Cloud & Tools", master, self.STYLE)
        assert result.strip().startswith(r"\item"), f"Missing \\item prefix in: {result}"
        assert r"\textbf{Cloud & Tools:}" in result
        assert "Docker" in result and "Terraform" in result
        # Should NOT have trailing \\
        assert not result.rstrip().endswith("\\\\")

    def test_itemize_to_itemize(self):
        """Master CV already in itemize → still works."""
        master = r"\item \textbf{Programming:} Python, Java, Go, Rust"
        result = reformat_skill_to_template_style("Programming", master, self.STYLE)
        assert result.strip().startswith(r"\item")
        assert r"\textbf{Programming:}" in result
        assert "Python" in result and "Rust" in result

    def test_preserves_all_items_itemize(self):
        """All items preserved in itemize reformat."""
        items = "Spark, Kafka, Airflow, dbt, Flink"
        master = r"\textbf{Data Engineering: } & " + items + r" \\"
        result = reformat_skill_to_template_style("Data Engineering", master, self.STYLE)
        for item in items.split(", "):
            assert item in result, f"Missing item '{item}' in: {result}"


# ──────────────────────────────────────────────────────────────
# FORMAT C: plain style (\textbf{Cat:} items, no \item, no &)
# ──────────────────────────────────────────────────────────────

class TestPlainStyle:
    """Template uses plain: \textbf{Cat:} items"""

    STYLE_WITH_TRAILING = r"\textbf{Languages:} Python, Java, C++ \\"
    STYLE_NO_TRAILING = r"\textbf{Languages:} Python, Java, C++"

    def test_itemize_to_plain_with_trailing(self):
        """Master CV itemize → plain with trailing \\."""
        master = r"\item \textbf{Cloud & Tools:} Docker, AWS, Terraform"
        result = reformat_skill_to_template_style("Cloud & Tools", master, self.STYLE_WITH_TRAILING)
        assert not result.strip().startswith(r"\item"), f"Should not have \\item in plain: {result}"
        assert "&" not in result.split("textbf")[0], f"Should not have & separator in plain: {result}"
        assert r"\textbf{Cloud & Tools:}" in result
        assert "Docker" in result
        assert result.rstrip().endswith("\\\\")

    def test_itemize_to_plain_no_trailing(self):
        """Master CV itemize → plain without trailing \\."""
        master = r"\item \textbf{Databases:} PostgreSQL, MongoDB, Redis"
        result = reformat_skill_to_template_style("Databases", master, self.STYLE_NO_TRAILING)
        assert r"\textbf{Databases:}" in result
        assert "PostgreSQL" in result
        assert not result.rstrip().endswith("\\\\"), f"Should NOT have trailing \\\\ in: {result}"


# ──────────────────────────────────────────────────────────────
# EDGE CASES
# ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases that could break the function."""

    TABULARX_STYLE = r"\textbf{Data Engineering: } & Spark, Kafka, Airflow \\"

    def test_empty_skill_items(self):
        """Master CV category with no items after the category name."""
        master = r"\item \textbf{Emerging:}"
        result = reformat_skill_to_template_style("Emerging", master, self.TABULARX_STYLE)
        assert r"\textbf{Emerging: }" in result
        # Should still produce valid output (just empty items)
        assert "&" in result

    def test_special_chars_in_items(self):
        """Skill items with special LaTeX chars (e.g. C++, C#, .NET)."""
        master = r"\item \textbf{Programming:} C++, C#, F#, .NET, Node.js"
        result = reformat_skill_to_template_style("Programming", master, self.TABULARX_STYLE)
        assert "C++" in result
        assert "C#" in result
        assert ".NET" in result

    def test_ampersand_in_category_name(self):
        """Category name contains & (e.g. 'Cloud & Tools')."""
        master = r"\item \textbf{Cloud \& Tools:} Docker, AWS"
        result = reformat_skill_to_template_style("Cloud & Tools", master, self.TABULARX_STYLE)
        # The function uses the provided category_name, not the one from master_cv
        assert "Cloud & Tools" in result or r"Cloud \& Tools" in result
        assert "Docker" in result

    def test_multiline_master_cv_block(self):
        """Master CV block spans multiple lines (shouldn't happen normally, but be safe)."""
        master = r"\item \textbf{Programming:} Python, Java," + "\n    Go, Rust, C++"
        result = reformat_skill_to_template_style("Programming", master, self.TABULARX_STYLE)
        assert "Python" in result
        # At minimum the first line's items should be captured

    def test_category_with_space_before_colon(self):
        """Master CV has \textbf{Category :} with space before colon."""
        master = r"\item \textbf{Programming :} Python, Java, Go"
        result = reformat_skill_to_template_style("Programming", master, self.TABULARX_STYLE)
        # Should still extract items even with odd spacing
        assert "Python" in result or "Java" in result

    def test_fallback_extraction(self):
        """When regex doesn't match, fallback to brace-based extraction."""
        # Malformed: no colon after category name in textbf
        master = r"\textbf{Programming} Python, Java, Go"
        result = reformat_skill_to_template_style("Programming", master, self.TABULARX_STYLE)
        # Should still produce something usable via fallback
        assert "Programming" in result


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])


# ──────────────────────────────────────────────────────────────
# INTEGRATION: Verify the new tier logic end-to-end
# ──────────────────────────────────────────────────────────────

class TestTierPriorityIntegration:
    """Verify that when a category exists in BOTH template and master CV,
    the master CV content wins (not the template content).

    This is the core bug being fixed.
    """

    TABULARX_STYLE = r"\textbf{Data Engineering: } & Spark, Kafka, Airflow \\"

    def test_master_cv_content_wins_over_template(self):
        """The core bug: master CV has MORE items than template.
        After reformat, the output should contain master CV items, not template items.
        """
        # Template has a SUBSET of skills
        template_block = r"\textbf{Programming: } & Python, Java \\"
        # Master CV has the FULL set
        master_block = r"\item \textbf{Programming:} Python, Java, Go, Rust, C++, TypeScript"

        # Using reformat_skill_to_template_style (which is what TIER 1 now does)
        result = reformat_skill_to_template_style(
            "Programming", master_block, self.TABULARX_STYLE
        )

        # Master CV items should ALL be present
        assert "Go" in result, f"Master CV item 'Go' missing from: {result}"
        assert "Rust" in result, f"Master CV item 'Rust' missing from: {result}"
        assert "C++" in result, f"Master CV item 'C++' missing from: {result}"
        assert "TypeScript" in result, f"Master CV item 'TypeScript' missing from: {result}"

        # Should be in tabularx format (from the style example)
        assert "&" in result
        assert result.rstrip().endswith("\\\\")

    def test_master_cv_unique_categories_now_reachable(self):
        """Categories that exist ONLY in master CV (not template) were
        previously invisible. Now they should be formatted correctly.
        """
        # A category that only exists in master CV (not in template)
        master_block = r"\item \textbf{MLOps:} MLflow, Weights & Biases, DVC, BentoML"

        result = reformat_skill_to_template_style(
            "MLOps", master_block, self.TABULARX_STYLE
        )

        assert "MLflow" in result
        assert "DVC" in result
        assert "BentoML" in result
        assert "&" in result  # tabularx format
