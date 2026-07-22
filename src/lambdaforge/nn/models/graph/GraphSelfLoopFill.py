"""Typed policies for synthesizing aligned self-loop edge features."""

from __future__ import annotations

from enum import Enum


class GraphSelfLoopFill(str, Enum):
    """Choose how new self-loops receive edge features."""

    ZERO = "zero"
    MEAN = "mean"

    @classmethod
    def _missing_(cls, value: object) -> GraphSelfLoopFill:
        allowed = ", ".join(repr(member.value) for member in cls)
        raise ValueError(f"GraphSelfLoopFill must be one of: {allowed}. Got {value!r}.")
