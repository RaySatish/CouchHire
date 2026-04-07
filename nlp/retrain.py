"""Fine-tune the ATS match scorer on real application outcomes.

Uses CouchHire's own outcome labels (from Telegram /outcome command) as the
ONLY training signal. Public HuggingFace resume/JD datasets were evaluated
and rejected — their scores are synthetic cosine similarity (circular and
useless).

Base model: 0xnbk/nbk-ats-semantic-v1-en
Training: sentence-transformers legacy model.fit() API with CosineSimilarityLoss.
Each retrain starts from the base model using ALL labeled data (not incremental).
"""

from __future__ import annotations

import logging
import time
from collections import Counter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Outcome → similarity label mapping
# ---------------------------------------------------------------------------
# These float labels represent how well the resume matched the JD,
# as evidenced by the real-world outcome.
OUTCOME_LABELS: dict[str, float] = {
    "offer": 1.0,        # Best possible signal — strong match
    "interview": 0.8,    # Got past screening — good match
    "no_response": 0.25, # Weak negative — might be bad match OR bad luck
    "rejected": 0.1,     # Strong negative — explicit rejection means poor fit
    # "withdrawn" is excluded from training — not a signal about match quality
}

# Maximum negative:positive ratio before oversampling kicks in
_MAX_IMBALANCE_RATIO = 3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrain(force: bool = False) -> dict:
    """Fine-tune the ATS match scorer on real application outcomes.

    Fetches all labeled outcomes from Supabase, builds training pairs,
    fine-tunes the base model, and saves the result. The fine-tuned model
    is automatically picked up by match_scorer.py on its next call.

    Args:
        force: If True, skip the minimum label count check. Useful for
               testing or manual retraining from the dashboard.

    Returns:
        dict with keys:
            - "status": "success" | "skipped" | "failed"
            - "reason": str — why it was skipped or failed
            - "num_labels": int — total labels found
            - "num_positive": int — interview + offer count
            - "num_negative": int — rejected + no_response count
            - "num_excluded": int — withdrawn count (excluded from training)
            - "epochs": int — training epochs used
            - "model_path": str — path to saved model (if success)
    """
    from config import MIN_RETRAIN_LABELS

    # 1. Fetch labeled data
    labeled_data, num_excluded = _fetch_training_data()
    num_labels = len(labeled_data)

    # Count by outcome
    outcome_counts = Counter(row["outcome"] for row in labeled_data)
    num_positive = outcome_counts.get("interview", 0) + outcome_counts.get("offer", 0)
    num_negative = outcome_counts.get("rejected", 0) + outcome_counts.get("no_response", 0)

    base_result = {
        "num_labels": num_labels,
        "num_positive": num_positive,
        "num_negative": num_negative,
        "num_excluded": num_excluded,
    }

    # 2. Check minimum threshold
    if not force and num_labels < MIN_RETRAIN_LABELS:
        return {
            **base_result,
            "status": "skipped",
            "reason": f"Only {num_labels} labels, need {MIN_RETRAIN_LABELS} minimum",
            "epochs": 0,
            "model_path": "",
        }

    # 3. Need at least 1 positive AND 1 negative for contrastive learning
    if num_positive == 0 or num_negative == 0:
        return {
            **base_result,
            "status": "skipped",
            "reason": (
                f"Need both positive and negative examples. "
                f"Have {num_positive} positive, {num_negative} negative"
            ),
            "epochs": 0,
            "model_path": "",
        }

    # 4. Build training pairs with class balancing
    try:
        examples = _build_training_pairs(labeled_data)
        logger.info("Built %d training examples (after balancing)", len(examples))
    except ValueError as exc:
        return {
            **base_result,
            "status": "failed",
            "reason": f"Failed to build training pairs: {exc}",
            "epochs": 0,
            "model_path": "",
        }

    # 5. Load base model (always from scratch, not from previous fine-tune)
    try:
        model = _get_base_model()
    except Exception as exc:
        logger.error("Failed to load base model: %s", exc, exc_info=True)
        return {
            **base_result,
            "status": "failed",
            "reason": f"Failed to load base model: {exc}",
            "epochs": 0,
            "model_path": "",
        }

    # 6. Train
    epochs = _calculate_epochs(len(examples))
    try:
        start_time = time.monotonic()
        model_path = _train_model(model, examples, epochs)
        elapsed = time.monotonic() - start_time
        logger.info(
            "Training complete in %.1fs — %d examples, %d epochs, saved to %s",
            elapsed, len(examples), epochs, model_path,
        )
    except Exception as exc:
        logger.error("Training failed: %s", exc, exc_info=True)
        return {
            **base_result,
            "status": "failed",
            "reason": f"Training failed: {exc}",
            "epochs": epochs,
            "model_path": "",
        }

    # 7. Force match_scorer to reload on next call
    try:
        from agents.match_scorer import reload_model
        reload_model()
    except ImportError:
        logger.warning(
            "Could not import match_scorer.reload_model — "
            "model will reload on next restart"
        )

    return {
        **base_result,
        "status": "success",
        "reason": "Retrained successfully",
        "epochs": epochs,
        "model_path": model_path,
    }


def should_retrain() -> bool:
    """Check if auto-retrain should trigger based on label count.

    Returns True if:
    1. RETRAIN_EVERY > 0 (auto-retrain is enabled)
    2. Total labeled outcomes >= MIN_RETRAIN_LABELS
    3. Total labeled outcomes is a multiple of RETRAIN_EVERY

    This is meant to be called after each /outcome command in the
    Telegram bot. If it returns True, the bot kicks off retrain()
    in a background thread.
    """
    from config import MIN_RETRAIN_LABELS, RETRAIN_EVERY

    if RETRAIN_EVERY <= 0:
        logger.debug("Auto-retrain disabled (RETRAIN_EVERY=%d)", RETRAIN_EVERY)
        return False

    try:
        from db.supabase_client import get_labeled_outcomes
        outcomes = get_labeled_outcomes()
    except Exception as exc:
        logger.warning("Failed to check labeled outcomes for auto-retrain: %s", exc)
        return False

    total = len(outcomes)

    if total < MIN_RETRAIN_LABELS:
        logger.debug(
            "Not enough labels for retrain: %d < %d", total, MIN_RETRAIN_LABELS
        )
        return False

    if total % RETRAIN_EVERY != 0:
        logger.debug(
            "Label count %d is not a multiple of RETRAIN_EVERY=%d",
            total, RETRAIN_EVERY,
        )
        return False

    logger.info(
        "Auto-retrain triggered: %d labels (multiple of %d, min %d)",
        total, RETRAIN_EVERY, MIN_RETRAIN_LABELS,
    )
    return True


# ---------------------------------------------------------------------------
# Internal functions
# ---------------------------------------------------------------------------

def _fetch_training_data() -> tuple[list[dict], int]:
    """Fetch and filter labeled outcomes from Supabase.

    Returns:
        A tuple of (filtered_data, num_excluded) where:
        - filtered_data: rows with jd_text, resume_content, and non-withdrawn outcome
        - num_excluded: count of withdrawn rows excluded
    """
    from db.supabase_client import get_labeled_outcomes

    all_outcomes = get_labeled_outcomes()
    logger.info("Fetched %d total labeled outcomes from Supabase", len(all_outcomes))

    num_excluded = 0
    filtered = []

    for row in all_outcomes:
        # Filter 1: must have jd_text
        jd_text = row.get("jd_text")
        if not jd_text or not jd_text.strip():
            logger.debug(
                "Skipping row %s: missing jd_text", row.get("id", "?")
            )
            continue

        # Filter 2: must have resume_content
        resume_content = row.get("resume_content")
        if not resume_content or not resume_content.strip():
            logger.debug(
                "Skipping row %s: missing resume_content", row.get("id", "?")
            )
            continue

        # Filter 3: exclude withdrawn (not a signal about match quality)
        outcome = row.get("outcome")
        if outcome == "withdrawn":
            num_excluded += 1
            continue

        # Filter 4: outcome must be in our label mapping
        if outcome not in OUTCOME_LABELS:
            logger.warning(
                "Skipping row %s: unknown outcome '%s'",
                row.get("id", "?"), outcome,
            )
            continue

        filtered.append(row)

    # Log breakdown
    outcome_counts = Counter(row["outcome"] for row in filtered)
    logger.info(
        "After filtering: %d usable rows (excluded %d withdrawn). "
        "Breakdown: %s",
        len(filtered), num_excluded,
        dict(outcome_counts),
    )

    return filtered, num_excluded


def _build_training_pairs(labeled_data: list[dict]) -> list:
    """Build training pairs from labeled outcome data.

    Each pair is (jd_text, resume_content) with a float similarity label
    derived from the outcome. Applies class balancing via oversampling
    of positive examples when negatives outnumber them by more than 3:1.

    Returns:
        A list of InputExample objects for sentence-transformers training.
    """
    from sentence_transformers import InputExample

    examples = []
    positive_examples = []
    negative_examples = []

    for row in labeled_data:
        jd = row["jd_text"]
        resume = row["resume_content"]
        label = OUTCOME_LABELS[row["outcome"]]

        example = InputExample(texts=[jd, resume], label=label)
        examples.append(example)

        # Track positives vs negatives for balancing
        if label >= 0.5:
            positive_examples.append(example)
        else:
            negative_examples.append(example)

    num_pos = len(positive_examples)
    num_neg = len(negative_examples)

    logger.info(
        "Training pairs before balancing: %d positive, %d negative (ratio %.1f:1)",
        num_pos, num_neg,
        num_neg / max(num_pos, 1),
    )

    # Class imbalance handling: oversample positives if ratio > 3:1
    if num_pos > 0 and num_neg > _MAX_IMBALANCE_RATIO * num_pos:
        # Need enough positive copies so that neg/pos <= 3
        target_pos = -(-num_neg // _MAX_IMBALANCE_RATIO)  # ceiling division
        copies_needed = target_pos - num_pos

        logger.info(
            "Oversampling %d positive examples to balance ratio "
            "(target: %d positive, %d negative, ratio %.1f:1)",
            copies_needed, target_pos, num_neg,
            num_neg / target_pos,
        )

        # Duplicate positive examples cyclically
        for i in range(copies_needed):
            src = positive_examples[i % num_pos]
            examples.append(InputExample(texts=list(src.texts), label=src.label))

    # Log final counts
    final_pos = sum(1 for ex in examples if ex.label >= 0.5)
    final_neg = sum(1 for ex in examples if ex.label < 0.5)
    logger.info(
        "Training pairs after balancing: %d total (%d positive, %d negative, ratio %.1f:1)",
        len(examples), final_pos, final_neg,
        final_neg / max(final_pos, 1),
    )

    return examples


def _get_base_model():
    """Load the base ATS scoring model with manual pooling assembly.

    Always loads from the HuggingFace repo, never from a previous fine-tune.
    The HF repo for 0xnbk/nbk-ats-semantic-v1-en is missing its
    1_Pooling/config.json, so we manually build Transformer + Pooling.

    Returns:
        A SentenceTransformer model ready for fine-tuning.
    """
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Transformer, Pooling

    model_name = "0xnbk/nbk-ats-semantic-v1-en"
    logger.info("Loading base model for retraining: %s", model_name)

    transformer = Transformer(model_name)
    embed_dim = transformer.get_word_embedding_dimension()
    pooling = Pooling(
        word_embedding_dimension=embed_dim,
        pooling_mode_mean_tokens=True,
    )
    model = SentenceTransformer(modules=[transformer, pooling])

    logger.info(
        "Base model loaded (dim=%d, pooling=mean)", embed_dim
    )
    return model


def _train_model(model, examples: list, epochs: int) -> str:
    """Fine-tune the model on training examples and save it.

    Args:
        model: A SentenceTransformer model to fine-tune.
        examples: List of InputExample objects.
        epochs: Number of training epochs.

    Returns:
        The path where the fine-tuned model was saved.
    """
    from sentence_transformers import losses
    from torch.utils.data import DataLoader

    from config import FINETUNED_MODEL_DIR

    # Training config — conservative for small datasets
    batch_size = min(16, max(4, len(examples) // 4))

    train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model)

    output_path = str(FINETUNED_MODEL_DIR)

    logger.info(
        "Starting training: %d examples, batch_size=%d, epochs=%d, "
        "warmup_steps=%d, output=%s",
        len(examples), batch_size, epochs,
        max(1, len(train_dataloader) // 10),
        output_path,
    )

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=max(1, len(train_dataloader) // 10),
        output_path=output_path,
        show_progress_bar=True,
    )

    return output_path


def _calculate_epochs(num_examples: int) -> int:
    """Calculate adaptive epoch count based on dataset size.

    Smaller datasets need more passes to learn; larger datasets
    converge faster and risk overfitting with too many epochs.

    Returns:
        Number of training epochs (2–8).
    """
    if num_examples <= 20:
        return 8
    elif num_examples <= 50:
        return 5
    elif num_examples <= 100:
        return 3
    else:
        return 2
