"""Offline tests for cv/embed_cv.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from cv import embed_cv


def test_load_text_file_prefers_user_file(tmp_path: Path) -> None:
    user = tmp_path / "user.txt"
    default = tmp_path / "default.txt"
    user.write_text("user", encoding="utf-8")
    default.write_text("default", encoding="utf-8")
    assert embed_cv._load_text_file(user, default, "x") == "user"


def test_load_text_file_falls_back_to_default(tmp_path: Path) -> None:
    user = tmp_path / "user-missing.txt"
    default = tmp_path / "default.txt"
    default.write_text("default", encoding="utf-8")
    assert embed_cv._load_text_file(user, default, "x") == "default"


def test_find_master_cv_returns_tex_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    tex = uploads / "master_cv.tex"
    tex.write_text("x", encoding="utf-8")
    monkeypatch.setattr(embed_cv, "_UPLOADS_DIR", uploads)
    assert embed_cv._find_master_cv() == tex
