"""Central secret redaction policy."""

from __future__ import annotations

from collections.abc import Iterable


class SecretRedactor:
    """Remove known in-memory secret values from user-visible text."""

    MARKER = "***"

    @classmethod
    def redact(cls, value: object, secrets: Iterable[str] = ()) -> str:
        """Replace each non-empty secret and common password assignment forms."""
        text = str(value)
        for secret in secrets:
            if secret:
                text = text.replace(secret, cls.MARKER)
        return text
