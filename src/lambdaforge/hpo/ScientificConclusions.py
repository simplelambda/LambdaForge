"""Shared vocabulary for exact scientific conclusions.

The adaptive controller, fixed sweeps and post-hoc analysis use the same tokens. A point estimate,
the stability of a descriptive conclusion and a formal sequential decision are deliberately
different objects.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SCIENTIFIC_CONCLUSION_KINDS = frozenset(
    {
        "PREFERRED",
        "PREFERRED_REGION",
        "PRACTICALLY_EQUIVALENT",
        "PRACTICAL_TOP_SET",
        "ALL_PRACTICALLY_EQUIVALENT",
        "BETTER_THAN_REFERENCE",
        "WORSE_THAN_REFERENCE",
        "EQUIVALENT_TO_REFERENCE",
        "PARTIALLY_ORDERED",
        "FLAT",
        "NO_MATERIAL_EFFECT",
        "CONTEXT_DEPENDENT",
        "UNRESOLVED",
    }
)
SCIENTIFIC_RELATION_KINDS = frozenset(
    {
        "MATERIALLY_BETTER",
        "PRACTICALLY_EQUIVALENT",
        "MATERIALLY_WORSE",
        "UNRESOLVED",
    }
)


@dataclass(frozen=True, slots=True)
class ScientificRelation:
    """One directed practical relation backed by an explicit confidence sequence."""

    left: Any
    right: Any
    relation: str
    estimate: float
    lower: float
    upper: float
    practical_margin: float | None

    def __post_init__(self) -> None:
        if self.relation not in SCIENTIFIC_RELATION_KINDS:
            raise ValueError(f"Unknown scientific relation: {self.relation!r}.")

    @property
    def resolved(self) -> bool:
        return self.relation != "UNRESOLVED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "left": self.left,
            "right": self.right,
            "relation": self.relation,
            "estimate": self.estimate,
            "confidence_sequence": {"lower": self.lower, "upper": self.upper},
            "practical_margin": self.practical_margin,
            "resolved": self.resolved,
        }


@dataclass(frozen=True, slots=True)
class ScientificConclusion:
    """An exact qualitative statement and the stability of that same statement."""

    kind: str
    values: tuple[Any, ...]
    token: str
    descriptive_stability: float

    def __post_init__(self) -> None:
        if self.kind not in SCIENTIFIC_CONCLUSION_KINDS:
            raise ValueError(f"Unknown scientific conclusion: {self.kind!r}.")
        if not 0.0 <= self.descriptive_stability <= 1.0:
            raise ValueError("descriptive_stability must be between zero and one.")

    @property
    def confidence(self) -> float:
        """Compatibility accessor for pre-0.16 analysis consumers."""
        return self.descriptive_stability

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "values": [_json_value(value) for value in self.values],
            "token": self.token,
            "descriptive_stability": self.descriptive_stability,
            "confidence": self.descriptive_stability,
        }


@dataclass(frozen=True, slots=True)
class ScientificQuestionState:
    """One question without conflating descriptive and sequential evidence."""

    conclusion: ScientificConclusion
    point_estimate: Mapping[str, Any]
    sequential_state: str | None = None
    sequential_coverage: float | None = None
    coverage_quality: float | None = None
    remaining_information_value: float | None = None

    @property
    def resolved(self) -> bool:
        return self.conclusion.kind != "UNRESOLVED" and self.sequential_state != "UNRESOLVED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "conclusion": self.conclusion.to_dict(),
            "point_estimate": dict(self.point_estimate),
            "sequential_decision": self.sequential_state,
            "sequential_coverage_level": self.sequential_coverage,
            "coverage_quality": self.coverage_quality,
            "remaining_information_value": self.remaining_information_value,
            "resolved": self.resolved,
        }


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    return str(value)


__all__ = [
    "SCIENTIFIC_CONCLUSION_KINDS",
    "SCIENTIFIC_RELATION_KINDS",
    "ScientificConclusion",
    "ScientificQuestionState",
    "ScientificRelation",
]
