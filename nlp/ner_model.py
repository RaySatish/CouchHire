"""Skill extraction from job descriptions using spaCy NER + noun chunks."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import spacy.language

logger = logging.getLogger(__name__)

# Lazy-loaded spaCy model — initialised on first call to extract_skills()
_nlp: spacy.language.Language | None = None


def _load_model() -> spacy.language.Language:
    """Load en_core_web_sm once and cache in module-level variable."""
    global _nlp
    if _nlp is None:
        import spacy

        logger.info("Loading spaCy model en_core_web_sm")
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# NER labels that commonly tag tech terms / tools / platforms
_SKILL_ENT_LABELS: set[str] = {"ORG", "PRODUCT", "GPE", "WORK_OF_ART", "LAW"}

# Tokens that slip through NER but are never skills
_STOPWORDS: set[str] = {
    "we", "our", "you", "your", "they", "the", "a", "an", "and", "or",
    "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "shall", "can", "must", "need", "experience",
    "knowledge", "understanding", "ability", "team", "work", "working",
    "years", "year", "plus", "strong", "good", "great", "excellent",
    "required", "preferred", "minimum", "including", "etc", "e.g",
    "role", "position", "job", "company", "candidate", "ideal",
    "responsibilities", "requirements", "qualifications", "skills",
    "developer", "engineer", "manager", "analyst", "designer",
    "senior", "junior", "mid", "level", "lead", "staff", "principal",
    "apply", "application", "submit", "send", "attach", "include",
    "resume", "cv", "cover", "letter", "subject", "email", "mail",
    "salary", "compensation", "benefits", "location", "remote",
    "trading", "startup", "corp", "inc", "llc", "ltd",
}

# Regex: valid skill-like token (letters, digits, dots, hyphens, plus, hash, slash)
_SKILL_PATTERN: re.Pattern[str] = re.compile(
    r"^[A-Za-z][A-Za-z0-9.+#/\-]*(?:\s[A-Za-z][A-Za-z0-9.+#/\-]*){0,3}$"
)


def _is_skill_like(text: str) -> bool:
    """Return True if *text* looks like a plausible technical skill token."""
    if len(text) < 2 or len(text) > 30:
        return False
    if text.lower() in _STOPWORDS:
        return False
    if not _SKILL_PATTERN.match(text):
        return False
    return True


def _normalise(text: str) -> str:
    """Strip and collapse whitespace."""
    return " ".join(text.split())


def _extract_chunk_candidate(chunk) -> str | None:
    """Extract a clean skill candidate from a spaCy noun chunk.

    Strips leading stopwords/determiners/adjectives, then reconstructs the
    remaining span using character offsets so hyphenated compounds like
    'scikit-learn' stay intact.
    """
    # Find the first content token (skip determiners, adjectives, stopwords)
    content_start = None
    for token in chunk:
        if not token.is_stop and token.pos_ not in ("DET", "ADJ", "ADP"):
            content_start = token.idx
            break

    if content_start is None:
        return None

    # Slice the original text from the first content token to the chunk end
    # This preserves hyphens and original spacing
    raw = chunk.text
    offset = content_start - chunk.start_char
    candidate = raw[offset:].strip()

    return candidate if candidate else None


def extract_skills(text: str) -> list[str]:
    """Extract technical skill tokens from *text* using spaCy NER + noun chunks.

    Returns a deduplicated list of skill-like strings found in the input.
    """
    if not text or not text.strip():
        return []

    nlp = _load_model()
    doc = nlp(text)

    seen: set[str] = set()
    skills: list[str] = []

    def _add(candidate: str) -> None:
        norm = _normalise(candidate)
        key = norm.lower()
        if key not in seen and _is_skill_like(norm):
            seen.add(key)
            skills.append(norm)

    # --- Pass 1: Named entities ---
    for ent in doc.ents:
        if ent.label_ in _SKILL_ENT_LABELS:
            _add(ent.text)

    # --- Pass 2: Noun chunks (preserves hyphenated compounds) ---
    for chunk in doc.noun_chunks:
        candidate = _extract_chunk_candidate(chunk)
        if candidate:
            _add(candidate)

    # --- Pass 3: Individual tokens (catches single-word skills missed above) ---
    for token in doc:
        if (
            not token.is_stop
            and not token.is_punct
            and not token.is_space
            and token.pos_ in ("PROPN", "NOUN")
        ):
            _add(token.text)

    logger.debug("Extracted %d skills from %d-char text", len(skills), len(text))
    return skills
