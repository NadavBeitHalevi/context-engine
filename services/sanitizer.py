"""Basic prompt-injection screening for retrieved content."""

import logging
import re

logger = logging.getLogger(__name__)

INJECTION_PATTERNS: tuple[str, ...] = (
    r"\bignore (?:all )?previous instructions\b",
    r"\bignore all prior commands\b",
    r"\byou are now in .* mode\b",
    r"\bact as\b",
    r"\bignore any legal advice\b",
    r"\bprint your instructions\b",
    r"\b(?:sudo|apt-get|yum|pip install)\b",
)

_COMPILED = tuple(re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS)


class UnsafeContentError(ValueError):
    """Raised when text matches a known prompt-injection pattern."""


def sanitize_input(text: str) -> str:
    """Return the text unchanged if clean; raise UnsafeContentError otherwise."""
    for pattern in _COMPILED:
        if pattern.search(text):
            logger.warning("Potential prompt injection detected (pattern: %s)", pattern.pattern)
            raise UnsafeContentError("Input sanitization failed: potential threat detected.")
    return text
