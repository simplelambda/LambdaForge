"""Central secret redaction policy."""

from __future__ import annotations

import re
from collections.abc import Iterable


class SecretRedactor:
    """Remove known in-memory secret values from user-visible text."""

    MARKER = "***"
    _SECRET_KEY = re.compile(
        r"(?:password|passwd|passphrase|token|api[_-]?key|secret|authorization|credential)",
        re.IGNORECASE,
    )
    _ASSIGNMENT = re.compile(
        r"(?i)\b(password|passwd|passphrase|token|api[_-]?key|secret|authorization|credential)"
        r"(\s*[:=]\s*)([^\s,;]+)"
    )
    _OPTION = re.compile(
        r"(?i)(--?(?:password|passwd|passphrase|token|api[_-]?key|secret|authorization|credential)"
        r"\s+)(\S+)"
    )
    _URL_CREDENTIAL = re.compile(r"(?P<scheme>\b[a-z][a-z0-9+.-]*://[^\s/:@]+):[^\s@/]+@")
    _BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
    _PRIVATE_KEY = re.compile(
        r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----",
        re.DOTALL,
    )

    @classmethod
    def redact(cls, value: object, secrets: Iterable[str] = ()) -> str:
        """Replace explicit values and common credential forms in text and tracebacks."""
        text = str(value)
        for secret in secrets:
            if secret:
                text = text.replace(secret, cls.MARKER)
        text = cls._PRIVATE_KEY.sub(cls.MARKER, text)
        text = cls._URL_CREDENTIAL.sub(r"\g<scheme>:***@", text)
        text = cls._BEARER.sub(f"Bearer {cls.MARKER}", text)
        text = cls._OPTION.sub(lambda match: f"{match.group(1)}{cls.MARKER}", text)
        text = cls._ASSIGNMENT.sub(
            lambda match: f"{match.group(1)}{match.group(2)}{cls.MARKER}", text
        )
        return text

    @classmethod
    def is_secret_key(cls, value: str) -> bool:
        """Identify structured mapping fields whose values must never be serialized."""
        normalized = value.replace("-", "_")
        return cls._SECRET_KEY.fullmatch(normalized) is not None
