"""Embedding merge modes for siamese models."""

from __future__ import annotations

from enum import Enum


class SiameseMerge(str, Enum):
    """Type-safe built-in operations for combining two embeddings."""

    ABSOLUTE_DIFFERENCE = "absolute_difference"
    DIFFERENCE = "difference"
    PRODUCT = "product"
    CONCATENATE = "concatenate"
    L1_DISTANCE = "l1_distance"
    L2_DISTANCE = "l2_distance"
    COSINE_SIMILARITY = "cosine_similarity"

    @classmethod
    def from_value(cls, value: SiameseMerge | str) -> SiameseMerge:
        """Normalize an enum or YAML-friendly string."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("merge must be a SiameseMerge or string.")
        normalized = value.strip().lower().replace("-", "_")
        aliases = {
            "abs_diff": cls.ABSOLUTE_DIFFERENCE,
            "concat": cls.CONCATENATE,
            "cosine": cls.COSINE_SIMILARITY,
            "l1": cls.L1_DISTANCE,
            "l2": cls.L2_DISTANCE,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError as error:
            choices = ", ".join(member.value for member in cls)
            raise ValueError(
                f"Unknown siamese merge {value!r}; expected one of {choices}."
            ) from error
