"""Output policies shared by sequence encoders."""

from __future__ import annotations

from enum import Enum


class SequenceOutputMode(str, Enum):
    """Select how a sequence of hidden states is returned."""

    SEQUENCE = "sequence"
    FIRST = "first"
    LAST = "last"
    MEAN = "mean"
    MAX = "max"

    @classmethod
    def _missing_(cls, value: object) -> SequenceOutputMode:
        allowed = ", ".join(repr(member.value) for member in cls)
        raise ValueError(f"output_mode must be one of: {allowed}. Got {value!r}.")
