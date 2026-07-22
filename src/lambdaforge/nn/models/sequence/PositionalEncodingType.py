"""Position encoding choices for Transformer sequence models."""

from __future__ import annotations

from enum import Enum


class PositionalEncodingType(str, Enum):
    """Supported position representations."""

    NONE = "none"
    SINUSOIDAL = "sinusoidal"
    LEARNED = "learned"

    @classmethod
    def _missing_(cls, value: object) -> PositionalEncodingType:
        allowed = ", ".join(repr(member.value) for member in cls)
        raise ValueError(f"positional_encoding must be one of: {allowed}. Got {value!r}.")
