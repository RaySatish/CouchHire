"""Offline tests for cv/cv_parser.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from cv.cv_parser import parse_cv, _is_section_heading


def test_parse_tex_extracts_sections(tmp_path: Path) -> None:
    tex = tmp_path / "resume.tex"
    tex.write_text(
        r"""
\section{Experience}
Built ML pipelines.
\section{Skills}
Python, PyTorch
\end{document}
""".strip(),
        encoding="utf-8",
    )
    sections = parse_cv(tex)
    assert "Experience" in sections
    assert "Skills" in sections


def test_parse_cv_unsupported_suffix(tmp_path: Path) -> None:
    bad = tmp_path / "resume.md"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_cv(bad)


@pytest.mark.parametrize("line,expected", [("EXPERIENCE", True), ("skills:", True), ("- bullet", False)])
def test_is_section_heading(line: str, expected: bool) -> None:
    assert _is_section_heading(line) is expected
