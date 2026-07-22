"""Reduction modes supported by :class:`EnsembleModel`."""

from __future__ import annotations

from enum import Enum


class EnsembleReduction(str, Enum):
    """Type-safe strategies for combining member predictions."""

    MEAN = "mean"
    SUM = "sum"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    STACK = "stack"
    CONCATENATE = "concatenate"
    WEIGHTED_MEAN = "weighted_mean"

    @classmethod
    def from_value(cls, value: EnsembleReduction | str) -> EnsembleReduction:
        """Normalize an enum or YAML-friendly string to one reduction value."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("reduction must be an EnsembleReduction or string.")
        normalized = value.strip().lower().replace("-", "_")
        aliases = {"concat": cls.CONCATENATE, "weighted": cls.WEIGHTED_MEAN}
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError as error:
            choices = ", ".join(member.value for member in cls)
            raise ValueError(
                f"Unknown ensemble reduction {value!r}; expected one of {choices}."
            ) from error
