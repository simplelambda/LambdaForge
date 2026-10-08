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

_RESOLVED_CONCLUSIONS = frozenset(
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
        "MATERIAL_INTERACTION",
        "WEAK_INTERACTION",
        "ADDITIVE",
    }
)


def formal_sequential_state(record: Mapping[str, Any] | None) -> str | None:
    """Read a formal decision; stopping for operational incompleteness is not resolution."""
    if record is None:
        return None
    formal = record.get("formal_sequential_evidence")
    if isinstance(formal, Mapping):
        return (
            "RESOLVED"
            if formal.get("state") == "RESOLVED" and record.get("stop") is True
            else "UNRESOLVED"
        )
    # Version-one historical records have no separate state. Preserve only explicit
    # supported scientific stops, never INCOMPLETE/budget/failure as formal approval.
    return (
        "RESOLVED"
        if record.get("stop") is True
        and record.get("conclusion") in _RESOLVED_CONCLUSIONS | {"SINGLE_CELL_COMPLETE"}
        else "UNRESOLVED"
    )


def scientific_status(scientific: Mapping[str, Any]) -> str:
    """Summarize material parameter and interaction questions consistently.

    An unresolved interaction is relevant only when its persisted materiality and remaining
    information make another feasible action scientifically worthwhile.  Parameter questions
    retain the conservative legacy behaviour when older records do not contain those fields.
    """
    resolution = _bounded_number(scientific.get("scientific_action_resolution"), default=0.0)
    questions: list[tuple[Mapping[str, Any], bool]] = []
    for key, interaction in (("parameter_questions", False), ("interaction_questions", True)):
        raw = scientific.get(key, ())
        if isinstance(raw, list | tuple):
            questions.extend((value, interaction) for value in raw if isinstance(value, Mapping))
    if not questions:
        return "unresolved"

    material_pending = 0
    resolved = 0
    considered = 0
    for question, interaction in questions:
        if question.get("feasible") is False or question.get("action_feasible") is False:
            continue
        kind = str(question.get("conclusion_kind", "UNRESOLVED"))
        remaining_raw = question.get("remaining_information_value")
        remaining_known = isinstance(remaining_raw, int | float) and not isinstance(
            remaining_raw, bool
        )
        remaining = _bounded_number(remaining_raw, default=0.0)
        materiality = _question_materiality(question, interaction=interaction)
        unresolved = kind in {"UNRESOLVED", "NO_CLEAR_PREFERENCE"} or bool(
            question.get("missing_evidence")
        )
        # Modern records explicitly state whether the pending interpretation is worth another
        # action.  Legacy parameter records remain conservative; legacy interactions are ignored
        # unless their own probability mass says they could be material.
        pending = (
            unresolved
            and materiality > 0.0
            and (
                remaining > resolution
                or (not remaining_known and (not interaction or materiality > 0.0))
            )
        )
        if pending:
            material_pending += 1
            considered += 1
        elif kind in _RESOLVED_CONCLUSIONS or (unresolved and remaining_known):
            resolved += 1
            considered += 1

    if material_pending:
        return "partially_resolved" if resolved else "unresolved"
    return "resolved" if considered else "unresolved"


def _bounded_number(value: Any, *, default: float) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return default
    return min(1.0, max(0.0, float(value)))


def _question_materiality(question: Mapping[str, Any], *, interaction: bool) -> float:
    raw = question.get("materiality")
    if isinstance(raw, int | float) and not isinstance(raw, bool):
        return min(1.0, max(0.0, float(raw)))
    if not interaction:
        return 1.0
    probabilities = question.get("probabilities")
    if not isinstance(probabilities, Mapping):
        return 0.0
    return min(
        1.0,
        max(
            0.0,
            float(probabilities.get("MATERIAL_INTERACTION", 0.0) or 0.0)
            + 0.5 * float(probabilities.get("WEAK_INTERACTION", 0.0) or 0.0),
        ),
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
    def descriptively_resolved(self) -> bool:
        return self.conclusion.kind != "UNRESOLVED"

    @property
    def formally_resolved(self) -> bool:
        return self.sequential_state == "RESOLVED"

    @property
    def resolved(self) -> bool:
        """Compatibility: applicable descriptive resolution, never formal approval."""
        return self.descriptively_resolved and (
            self.sequential_state is None or self.formally_resolved
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "conclusion": self.conclusion.to_dict(),
            "point_estimate": dict(self.point_estimate),
            "sequential_decision": self.sequential_state,
            "sequential_coverage_level": self.sequential_coverage,
            "coverage_quality": self.coverage_quality,
            "remaining_information_value": self.remaining_information_value,
            "resolved": self.resolved,
            "descriptively_resolved": self.descriptively_resolved,
            "formally_resolved": self.formally_resolved,
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
    "scientific_status",
    "formal_sequential_state",
]
