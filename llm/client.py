"""LiteLLM wrapper — single entry point for all LLM calls in CouchHire.

Every agent calls `complete()` from this module. Never import or call
LiteLLM directly from anywhere else in the codebase.

Implements automatic 9-model fallback chain across 5 providers on
RateLimitError or APIError. The fallback chain is built dynamically in
config.py based on which API keys are present. Ollama (local) is always
the final fallback.
"""

import logging

import litellm
from litellm.exceptions import APIError, RateLimitError

logger = logging.getLogger(__name__)

# Provider → model mapping lives in config.py (single source of truth).
# Imported lazily inside complete() to avoid import-time side effects.

# OpenRouter requires these headers for their free tier.
_OPENROUTER_HEADERS: dict[str, str] = {
    "HTTP-Referer": "https://github.com/RaySatish/CouchHire",
    "X-Title": "CouchHire",
}


def _is_ollama_running() -> bool:
    """Check if Ollama is reachable on localhost. Returns False on any failure."""
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


def _friendly_provider_name(model_string: str) -> str:
    """Extract a human-readable provider name from a litellm model string."""
    return model_string.split("/")[0] if "/" in model_string else model_string


def complete(prompt: str, system_prompt: str | None = None) -> str:
    """Send a prompt to the configured LLM provider and return the response text.

    Tries the primary provider first, then walks the FALLBACK_CHAIN from
    config.py on RateLimitError or APIError. Ollama is checked for
    availability before attempting it so it doesn't hang the pipeline.

    Args:
        prompt: The user/instruction prompt.
        system_prompt: Optional system-level instruction prepended to the
            conversation.

    Returns:
        The model's response as a plain string.

    Raises:
        ValueError: If LLM_PROVIDER is not one of the supported values.
        RuntimeError: If all providers (primary + fallbacks) fail.
    """
    # Import config lazily so the module is importable without side effects
    # when config validation hasn't run yet (e.g. during tool_search or tests
    # that mock config).
    from config import FALLBACK_CHAIN, LLM_PROVIDER, PROVIDER_MODEL_MAP

    if LLM_PROVIDER not in PROVIDER_MODEL_MAP:
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. "
            f"Valid options: {list(PROVIDER_MODEL_MAP.keys())}"
        )

    primary_model = PROVIDER_MODEL_MAP[LLM_PROVIDER]

    # Build candidate list: primary first, then fallbacks
    candidates: list[str] = [primary_model] + list(FALLBACK_CHAIN)

    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Track failures for the final RuntimeError if all candidates fail
    failures: list[tuple[str, str]] = []

    for i, model in enumerate(candidates):
        provider_name = _friendly_provider_name(model)
        is_primary = (i == 0)
        is_ollama = model.startswith("ollama/")
        is_openrouter = model.startswith("openrouter/")

        # Ollama: check if running before attempting — don't let it hang
        if is_ollama:
            if not _is_ollama_running():
                reason = "Ollama not running — skipping local fallback"
                logger.warning(reason)
                failures.append((model, reason))
                continue
            logger.info("All cloud providers exhausted. Falling back to Ollama (local).")

        logger.info(
            "LLM call — provider=%s  model=%s  prompt_len=%d%s",
            provider_name,
            model,
            len(prompt),
            "" if is_primary else "  (fallback)",
        )

        try:
            # Build kwargs for litellm.completion
            kwargs: dict = {"model": model, "messages": messages}

            # OpenRouter requires HTTP-Referer header for their free tier
            if is_openrouter:
                kwargs["extra_headers"] = _OPENROUTER_HEADERS

            response = litellm.completion(**kwargs)
            text: str = response.choices[0].message.content
            logger.info("LLM response — provider=%s  chars=%d", provider_name, len(text))

            if not is_primary:
                logger.info(
                    "Successfully used fallback provider %s after primary (%s) failed.",
                    provider_name,
                    LLM_PROVIDER,
                )

            return text

        except RateLimitError as exc:
            reason = f"RateLimitError: {exc}"
            failures.append((model, reason))
            if i + 1 < len(candidates):
                next_model = candidates[i + 1]
                next_name = _friendly_provider_name(next_model)
                logger.warning(
                    "Rate limited on %s — trying next fallback: %s.",
                    model,
                    next_name,
                )
            else:
                logger.warning(
                    "Rate limited on %s — no more fallbacks remaining.",
                    model,
                )

        except APIError as exc:
            reason = f"APIError: {exc}"
            failures.append((model, reason))
            if i + 1 < len(candidates):
                next_model = candidates[i + 1]
                next_name = _friendly_provider_name(next_model)
                logger.warning(
                    "API error on %s: %s — trying next fallback: %s.",
                    model,
                    exc,
                    next_name,
                )
            else:
                logger.warning(
                    "API error on %s: %s — no more fallbacks remaining.",
                    model,
                    exc,
                )

        except Exception as exc:
            # Catch-all for unexpected errors (auth failures, network, etc.)
            reason = f"{type(exc).__name__}: {exc}"
            failures.append((model, reason))
            if i + 1 < len(candidates):
                next_model = candidates[i + 1]
                next_name = _friendly_provider_name(next_model)
                logger.warning(
                    "Unexpected error on %s (%s) — trying next fallback: %s.",
                    model,
                    type(exc).__name__,
                    next_name,
                )
            else:
                logger.warning(
                    "Unexpected error on %s (%s) — no more fallbacks remaining.",
                    model,
                    type(exc).__name__,
                )

    # All candidates exhausted
    failure_summary = "\n".join(
        f"  - {model}: {reason}" for model, reason in failures
    )
    logger.error("All providers failed including Ollama.\n%s", failure_summary)
    raise RuntimeError(
        f"All LLM providers failed. Tried {len(failures)} provider(s):\n"
        f"{failure_summary}"
    )
