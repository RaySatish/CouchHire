"""Offline tests for nlp/ner_model.py."""

from __future__ import annotations

import pytest

from nlp import ner_model


def test_extract_skills_returns_empty_for_blank() -> None:
    assert ner_model.extract_skills("   ") == []


def test_extract_skills_returns_list(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEnt:
        def __init__(self, text: str, label: str):
            self.text = text
            self.label_ = label

    class FakeToken:
        def __init__(self, text: str):
            self.text = text
            self.is_stop = False
            self.is_punct = False
            self.is_space = False
            self.pos_ = "NOUN"

    class FakeChunk:
        text = "Python"
        start_char = 0

        def __iter__(self):
            token = FakeToken("Python")
            token.idx = 0
            token.pos_ = "NOUN"
            token.is_stop = False
            return iter([token])

    class FakeDoc:
        ents = [FakeEnt("PyTorch", "PRODUCT")]
        noun_chunks = [FakeChunk()]

        def __iter__(self):
            return iter([FakeToken("TensorFlow")])

    monkeypatch.setattr(ner_model, "_load_model", lambda: (lambda _text: FakeDoc()))
    out = ner_model.extract_skills("Need Python and TensorFlow")
    assert isinstance(out, list)
    assert out
