"""LiteLLM wrapper — single entry point for all LLM calls in CouchHire.

Every agent calls `complete()` from this module. Never import or call
LiteLLM directly from anywhere else in the codebase.
"""

import logging

import litellm
from litellm.exceptions import APIError

logger = logging.getLogger(__name__)

# ── Provider → model mapping ────────────────────────────────────────────────
# LiteLLM model strings use a provider prefix so the router knows which API
# to hit.  Keep these in sync with the providers listed in config.py.
_PROVIDER_MODEL_MAP: dict[str, str] = {
    "groq": "groq/llama-3.3-70b-versatile",
    "gemini": "gemini/gemini-2.0-flash",
    "anthropic": "anthropic/claude-sonnet-4-20250514",
    "openai": "openai/gpt-4o",
}

_VALID_PROVIDERS = list(_PROVIDER_MODEL_MAP.keys())


def complete(prompt: str, system_prompt: str | None = None) -> str:
    """Send a prompt to the configured LLM provider and return the response text.

    Args:
        prompt: The user/instruction prompt.
        system_prompt: Optional system-level instruction prepended to the
            conversation.

    Returns:
        The model's response as a plain string.

    Raises:
        ValueError: If LLM_PROVIDER is not one of the four supported values.
        litellm.exceptions.APIError: Re-raised after logging context.
    """
    # Import config lazily so the module is importable without side effects
    # when config validation hasn't run yet (e.g. during tool_search or tests
    # that mock config).
    from config import LLM_PROVIDER

    if LLM_PROVIDER not in _PROVIDER_MODEL_MAP:
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. "
            f"Valid options: {_VALID_PROVIDERS}"
        )

    model = _PROVIDER_MODEL_MAP[LLM_PROVIDER]

    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    logger.info(
        "LLM call — provider=%s  model=%s  prompt_len=%d",
        LLM_PROVIDER,
        model,
        len(prompt),
    )

    try:
        response = litellm.completion(model=model, messages=messages)
    except APIError:
        logger.error(
            "LiteLLM APIError — provider=%s  prompt_len=%d",
            LLM_PROVIDER,
            len(prompt),
            exc_info=True,
        )
        raise

    text: str = response.choices[0].message.content
    logger.info("LLM response — chars=%d", len(text))
    return text
