"""CouchHire configuration — loads .env, validates required keys, exposes constants.

Import this module before anything else. If any required environment variable
is missing, a RuntimeError is raised listing every missing key.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load .env from project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
_ENV_PATH = _PROJECT_ROOT / ".env"

if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)
else:
    logger.warning(".env file not found at %s — relying on environment variables", _ENV_PATH)

# ---------------------------------------------------------------------------
# Helper — read env var (returns str or None)
# ---------------------------------------------------------------------------

def _env(key: str) -> str | None:
    """Read an environment variable, returning None if unset or empty."""
    value = os.environ.get(key, "").strip()
    return value if value else None


# ---------------------------------------------------------------------------
# LLM provider + API key
# ---------------------------------------------------------------------------
LLM_PROVIDER: str | None = _env("LLM_PROVIDER")

# Maps provider name → environment variable holding the API key.
_PROVIDER_KEY_MAP: dict[str, str] = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

# Resolved after validation
LLM_API_KEY: str | None = None

# ---------------------------------------------------------------------------
# Single source of truth: provider → litellm model string.
# Imported by llm/client.py — never duplicate this mapping elsewhere.
# ---------------------------------------------------------------------------
PROVIDER_MODEL_MAP: dict[str, str] = {
    "groq": "groq/llama-3.3-70b-versatile",
    "gemini": "gemini/gemini-3.1-flash-lite-preview",
    "mistral": "mistral/mistral-small-latest",
    "openrouter": "openrouter/auto",
    "anthropic": "anthropic/claude-haiku-4-5",
    "openai": "openai/gpt-4o-mini",
}

# ---------------------------------------------------------------------------
# Groq fallback models — separate quota buckets, same API key.
# Each model has its own TPM/RPM allocation on Groq's free tier,
# so exhausting one doesn't block the others.
# ---------------------------------------------------------------------------
GROQ_FALLBACK_MODELS: list[str] = [
    "groq/qwen/qwen3-32b",
    "groq/openai/gpt-oss-120b",
    "groq/meta-llama/llama-4-scout-17b-16e-instruct",
    "groq/llama-3.1-8b-instant",
]

# ---------------------------------------------------------------------------
# Required variables (always needed regardless of provider)
# ---------------------------------------------------------------------------
SUPABASE_URL: str | None = _env("SUPABASE_URL")
SUPABASE_KEY: str | None = _env("SUPABASE_KEY")

TELEGRAM_BOT_TOKEN: str | None = _env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID: str | None = _env("TELEGRAM_CHAT_ID")

GMAIL_MCP_URL: str | None = _env("GMAIL_MCP_URL")

INDEED_MCP_TOKEN: str | None = _env("INDEED_MCP_TOKEN")

GITHUB_URL: str | None = _env("GITHUB_URL")
APPLICANT_NAME: str | None = _env("APPLICANT_NAME")

# Match threshold — integer 0-100
_raw_threshold = _env("MATCH_THRESHOLD")
MATCH_THRESHOLD: int = int(_raw_threshold) if _raw_threshold is not None else 60

# ---------------------------------------------------------------------------
# Optional variables (not validated — sensible defaults)
# ---------------------------------------------------------------------------

# Direct Postgres connection string — used ONLY by db/create_tables.py.
# Not needed at runtime. See .env.example for format details.
SUPABASE_DB_URL: str | None = _env("SUPABASE_DB_URL")

# Supabase Management API token — used ONLY by db/create_tables.py as
# a fallback when direct Postgres connection fails.
# Generate at: https://supabase.com/dashboard/account/tokens
SUPABASE_ACCESS_TOKEN: str | None = _env("SUPABASE_ACCESS_TOKEN")

# ---------------------------------------------------------------------------
# Optional variables — browser agent / ATS form filling
# ---------------------------------------------------------------------------

# Chrome DevTools Protocol port for browser agent sessions.
# Default 9222 — only change if that port is already in use.
_raw_cdp_port = _env("CDP_PORT")
CDP_PORT: int = int(_raw_cdp_port) if _raw_cdp_port is not None else 9222

# Applicant contact info — used to pre-fill ATS form fields.
APPLICANT_PHONE: str | None = _env("APPLICANT_PHONE")
APPLICANT_LINKEDIN: str | None = _env("APPLICANT_LINKEDIN")

# Whether to run the browser agent headless (no visible window).
# Default: false (visible browser for development). Set to true in production.
BROWSER_HEADLESS: bool = os.getenv("BROWSER_HEADLESS", "false").lower() in ("true", "1", "yes")

# Path to persistent answer memory for ATS form fields.
# Defaults to cv/output/form_answers.json (gitignored).
_raw_form_answers = _env("FORM_ANSWERS_PATH")
FORM_ANSWERS_PATH: Path = (
    Path(_raw_form_answers) if _raw_form_answers is not None
    else _PROJECT_ROOT / "cv" / "output" / "form_answers.json"
)

# ---------------------------------------------------------------------------
# Optional variables — NLP retraining
# ---------------------------------------------------------------------------

# Minimum number of labeled outcomes required before retraining is allowed.
# Below this threshold, there's not enough signal to fine-tune meaningfully.
_raw_min_retrain = _env("MIN_RETRAIN_LABELS")
MIN_RETRAIN_LABELS: int = int(_raw_min_retrain) if _raw_min_retrain is not None else 10

# How often to auto-retrain: every N new outcome labels.
# Set to 0 to disable auto-retrain (manual only via dashboard/CLI).
_raw_retrain_every = _env("RETRAIN_EVERY")
RETRAIN_EVERY: int = int(_raw_retrain_every) if _raw_retrain_every is not None else 10

# ---------------------------------------------------------------------------
# Validation — runs on import
# ---------------------------------------------------------------------------

def _validate() -> None:
    """Validate all required config. Raises RuntimeError listing every missing key."""
    missing: list[str] = []

    # LLM provider must be set and valid
    if LLM_PROVIDER is None:
        missing.append("LLM_PROVIDER")
    elif LLM_PROVIDER not in _PROVIDER_KEY_MAP:
        raise RuntimeError(
            f"LLM_PROVIDER must be one of {list(_PROVIDER_KEY_MAP.keys())}, "
            f"got '{LLM_PROVIDER}'"
        )

    # Provider-specific API key
    if LLM_PROVIDER is not None and LLM_PROVIDER in _PROVIDER_KEY_MAP:
        key_name = _PROVIDER_KEY_MAP[LLM_PROVIDER]
        if _env(key_name) is None:
            missing.append(key_name)

    # Always-required keys
    _always_required = {
        "SUPABASE_URL": SUPABASE_URL,
        "SUPABASE_KEY": SUPABASE_KEY,
        "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
        "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID,
        "GMAIL_MCP_URL": GMAIL_MCP_URL,
        "INDEED_MCP_TOKEN": INDEED_MCP_TOKEN,
        "GITHUB_URL": GITHUB_URL,
        "APPLICANT_NAME": APPLICANT_NAME,
    }

    for key_name, value in _always_required.items():
        if value is None:
            missing.append(key_name)

    # Match threshold range check
    if not (0 <= MATCH_THRESHOLD <= 100):
        raise RuntimeError(
            f"MATCH_THRESHOLD must be between 0 and 100, got {MATCH_THRESHOLD}"
        )

    if missing:
        raise RuntimeError(
            "Missing required environment variables:\n  - " + "\n  - ".join(missing)
        )


_validate()

# Set LLM_API_KEY after validation passes (we know provider + key exist)
if LLM_PROVIDER is not None:
    LLM_API_KEY = _env(_PROVIDER_KEY_MAP[LLM_PROVIDER])

# ---------------------------------------------------------------------------
# Derived paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = _PROJECT_ROOT
CV_DIR: Path = _PROJECT_ROOT / "cv"
CHROMA_STORE_DIR: Path = CV_DIR / "chroma_store"
MASTER_CV_PATH: Path = CV_DIR / "master_cv.tex"
OUTPUT_DIR: Path = _PROJECT_ROOT / "output"

# Directory where fine-tuned models are saved (gitignored via nlp/models/).
FINETUNED_MODEL_DIR: Path = _PROJECT_ROOT / "nlp" / "models" / "match_scorer_finetuned"

# ---------------------------------------------------------------------------
# Fallback chain — built dynamically based on which API keys are present.
#
# Priority order:
#   1. Groq fallback models (4 models, separate quota buckets — same key)
#   2. Gemini
#   3. Mistral
#   4. OpenRouter (auto-routes across 29+ free models)
#   5. Anthropic (paid keys only — no meaningful free tier)
#   6. OpenAI (paid keys only — free tier is 3 RPM on GPT-3.5 only)
#   7. Ollama (local, unlimited, no API key required — always last)
#
# DeepSeek intentionally excluded — servers are in China, unsuitable
# for CV and personal job application data.
# ---------------------------------------------------------------------------

# Primary model string (the one LLM_PROVIDER resolves to)
_PRIMARY_MODEL: str | None = (
    PROVIDER_MODEL_MAP[LLM_PROVIDER]
    if LLM_PROVIDER and LLM_PROVIDER in PROVIDER_MODEL_MAP
    else None
)

FALLBACK_CHAIN: list[str] = []

# 1. Groq fallback models (separate quota buckets, same API key)
if _env("GROQ_API_KEY") is not None:
    for _groq_model in GROQ_FALLBACK_MODELS:
        if _groq_model != _PRIMARY_MODEL:
            FALLBACK_CHAIN.append(_groq_model)
    # Also include the primary Groq model as fallback if primary provider
    # is NOT groq (e.g. user's primary is gemini but has Groq key too)
    _groq_primary = PROVIDER_MODEL_MAP["groq"]
    if _groq_primary != _PRIMARY_MODEL:
        # Insert the main Groq model at the start of the Groq section
        FALLBACK_CHAIN.insert(0, _groq_primary)

# 2–6. Other cloud providers in priority order
_FALLBACK_PRIORITY: list[str] = ["gemini", "mistral", "openrouter", "anthropic", "openai"]

for _prov in _FALLBACK_PRIORITY:
    _model_str = PROVIDER_MODEL_MAP[_prov]
    _key_var = _PROVIDER_KEY_MAP.get(_prov)
    if _key_var and _env(_key_var) is not None and _model_str != _PRIMARY_MODEL:
        FALLBACK_CHAIN.append(_model_str)

# 7. Ollama is always the final fallback — local, no API key required
FALLBACK_CHAIN.append("ollama/llama3.2")
