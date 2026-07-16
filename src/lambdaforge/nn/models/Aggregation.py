"""Supported sparse segment-reduction modes."""

from __future__ import annotations

from enum import Enum


class Aggregation(str, Enum):
    """Reduction applied to sparse messages that share a segment index."""

    SUM = "sum"
    MEAN = "mean"

    @classmethod
    def _missing_(cls, value: object) -> Aggregation:
        allowed = ", ".join(repr(member.value) for member in cls)
        raise ValueError(f"Aggregation must be one of: {allowed}. Got {value!r}.")
