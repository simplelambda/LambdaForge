"""Redacted configuration secret."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SecretValue:
    """Hold a runtime secret without revealing it in text or serialized views."""

    value: str
    source: str

    def __repr__(self) -> str:
        return "SecretValue(***)"

    def __str__(self) -> str:
        return "***"
