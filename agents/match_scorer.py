"""Compute a match score between a job description and retrieved CV sections.

Uses the 0xnbk/nbk-ats-semantic-v1-en model from Hugging Face — a pre-trained
ATS semantic scorer fine-tuned specifically for resume/JD matching. This is
intentionally a different model from the all-MiniLM-L6-v2 retrieval model used
in cv_rag.py; retrieval and scoring are separate concerns.

The model's HF repo is missing its 1_Pooling/config.json, so we manually
assemble the Transformer + Pooling pipeline on first load.

No LLM calls — pure sentence-transformers inference.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SCORING_MODEL = "0xnbk/nbk-ats-semantic-v1-en"

# Lazy singleton — initialised on first call, not at import time.
_model = None


def _get_model():
    """Lazily load the ATS semantic scoring model.

    The HF repo for this model is missing its Pooling config directory,
    so we load the Transformer component directly and attach a mean-pooling
    layer manually.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.models import Transformer, Pooling

        logger.info("Loading ATS scoring model: %s", _SCORING_MODEL)

        # Load transformer weights from the HF repo
        transformer = Transformer(_SCORING_MODEL)
        embed_dim = transformer.get_word_embedding_dimension()

        # Attach mean-pooling (the repo's modules.json declares Pooling
        # but the 1_Pooling/config.json is absent — we supply it here)
        pooling = Pooling(
            word_embedding_dimension=embed_dim,
            pooling_mode_mean_tokens=True,
        )

        _model = SentenceTransformer(modules=[transformer, pooling])
        logger.info(
            "ATS scoring model loaded (dim=%d, pooling=mean)",
            embed_dim,
        )
    return _model


def score(jd_text: str, cv_sections: list[str]) -> float:
    """Compute a match score between a job description and CV sections.

    Embeds the JD and each CV section independently, computes pairwise
    cosine similarity, and returns the mean similarity scaled to 0–100.

    Args:
        jd_text: The raw job description text.
        cv_sections: List of CV section texts (from cv_rag.retrieve_cv_sections).

    Returns:
        A float between 0.0 and 100.0 representing the match quality.
        Returns 0.0 if cv_sections is empty.
    """
    if not cv_sections:
        logger.warning("No CV sections provided — returning score 0.0")
        return 0.0

    if not jd_text or not jd_text.strip():
        logger.warning("Empty JD text provided — returning score 0.0")
        return 0.0

    from sentence_transformers import util

    model = _get_model()

    # Step 1: Embed JD as a single vector
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)

    # Step 2: Embed each CV section individually
    section_embeddings = model.encode(cv_sections, convert_to_tensor=True)

    # Step 3: Compute cosine similarity between JD and each section
    similarities = util.cos_sim(jd_embedding, section_embeddings)[0]

    logger.info("Per-section similarities for JD (%.80s...):", jd_text[:80])
    for i, (section, sim) in enumerate(zip(cv_sections, similarities)):
        logger.info(
            "  Section %d (%.60s...): %.4f",
            i + 1,
            section[:60],
            sim.item(),
        )

    # Step 4: Mean similarity * 100
    mean_sim = similarities.mean().item()
    match_score = round(max(0.0, min(100.0, mean_sim * 100)), 2)

    logger.info(
        "Match score: %.2f (mean cosine similarity: %.4f)",
        match_score,
        mean_sim,
    )
    return match_score
