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

_PROVIDER_KEY_MAP: dict[str, str] = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

# Resolved after validation
LLM_API_KEY: str | None = None

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
# Optional variables (not validated — used by setup scripts only)
# ---------------------------------------------------------------------------

# Direct Postgres connection string — used ONLY by db/create_tables.py.
# Not needed at runtime. See .env.example for format details.
SUPABASE_DB_URL: str | None = _env("SUPABASE_DB_URL")

# Supabase Management API token — used ONLY by db/create_tables.py as
# a fallback when direct Postgres connection fails.
# Generate at: https://supabase.com/dashboard/account/tokens
SUPABASE_ACCESS_TOKEN: str | None = _env("SUPABASE_ACCESS_TOKEN")

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
