"""Output policies for graph networks that update geometric coordinates."""

from __future__ import annotations

from enum import Enum


class EquivariantOutputMode(str, Enum):
    """Choose the public output returned by an equivariant graph stack."""

    FEATURES = "features"
    MAPPING = "mapping"

    @classmethod
    def _missing_(cls, value: object) -> EquivariantOutputMode:
        allowed = ", ".join(repr(member.value) for member in cls)
        raise ValueError(f"EquivariantOutputMode must be one of: {allowed}. Got {value!r}.")
