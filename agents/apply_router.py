"""Apply-method router — deterministically routes applications to email, form, or manual."""

import logging
import re

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _is_valid_email(value: str) -> bool:
    """Return True if value looks like a valid email address."""
    return bool(_EMAIL_RE.match(value))


def _is_valid_url(value: str) -> bool:
    """Return True if value starts with http:// or https://."""
    return value.startswith("http://") or value.startswith("https://")


def route(requirements: dict) -> str:
    """Determine the application route based on parsed JD requirements.

    Returns exactly one of: 'email', 'form', or 'manual'.
    """
    apply_method = requirements.get("apply_method", "unknown")
    apply_target = requirements.get("apply_target") or ""
    target_str = str(apply_target).strip()

    # Rule 1: email route — method is 'email' and target is a valid email
    if apply_method == "email" and target_str and _is_valid_email(target_str):
        logger.info(
            "Route decision: email — apply_method='email', "
            "valid email target '%s'",
            target_str,
        )
        return "email"

    # Rule 2: form route — method is 'url' and target is a valid URL
    if apply_method == "url" and target_str and _is_valid_url(target_str):
        logger.info(
            "Route decision: form — apply_method='url', "
            "valid URL target '%s'",
            target_str,
        )
        return "form"

    # Rule 3: everything else falls through to manual
    logger.info(
        "Route decision: manual — apply_method='%s', apply_target='%s' "
        "(did not match email or form criteria)",
        apply_method,
        target_str,
    )
    return "manual"
