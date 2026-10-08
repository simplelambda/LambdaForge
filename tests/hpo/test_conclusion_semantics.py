"""Stability belongs to one exact displayed event, not its complement."""

import pytest

from lambdaforge.hpo.ScientificConclusions import (
    ScientificConclusion,
    ScientificQuestionState,
    formal_sequential_state,
)
from lambdaforge.hpo.ScientificDesign import ScientificQuestionAnalyzer
from lambdaforge.work.sweep_blocks import sweep_block_inventory


@pytest.mark.parametrize(
    "distribution", [{}, {'PREFERRED:"a"': 0.45, 'PREFERRED:"b"': 0.4, "UNRESOLVED": 0.15}]
)
def test_unresolved_stability_is_its_own_mass(distribution: dict[str, float]) -> None:
    exact = ScientificQuestionAnalyzer._exact_conclusion(  # noqa: SLF001
        distribution, levels=["a", "b"], practical_margin=0.01
    )
    assert exact.kind == "UNRESOLVED"
    assert exact.descriptive_stability == distribution.get("UNRESOLVED", 0)


def test_missing_sequential_state_is_not_formal_approval() -> None:
    question = ScientificQuestionState(
        ScientificConclusion("PREFERRED", ("a",), 'PREFERRED:"a"', 0.9), {"value": "b"}
    )
    assert question.descriptively_resolved
    assert question.resolved  # Historical exploratory use remains descriptive.
    assert not question.formally_resolved
    assert not question.to_dict()["formally_resolved"]
    assert question.to_dict()["point_estimate"] == {"value": "b"}
    assert question.to_dict()["conclusion"]["values"] == ["a"]


def test_exact_preference_and_equivalence_keep_their_own_event_mass() -> None:
    analyzer = ScientificQuestionAnalyzer
    preference = analyzer._exact_conclusion(  # noqa: SLF001
        {'PREFERRED:"b"': 0.7, 'PREFERRED:"a"': 0.3},
        levels=["a", "b"],
        practical_margin=0.01,
    )
    assert preference.values == ("b",)
    assert preference.confidence == 0.7
    equivalence = analyzer._exact_conclusion(  # noqa: SLF001
        {"PRACTICALLY_EQUIVALENT": 0.9, 'PREFERRED:"b"': 0.1},
        levels=["a", "b"],
        practical_margin=0.01,
    )
    assert equivalence.kind == "PRACTICALLY_EQUIVALENT"
    assert equivalence.confidence == 0.9
    without_margin = analyzer._exact_conclusion(  # noqa: SLF001
        {"PRACTICALLY_EQUIVALENT": 0.9, 'PREFERRED:"b"': 0.1},
        levels=["a", "b"],
        practical_margin=None,
    )
    assert without_margin.kind == "UNRESOLVED"
    assert without_margin.confidence == 0


def test_operational_stop_does_not_become_a_formal_scientific_decision() -> None:
    assert formal_sequential_state(None) is None
    assert formal_sequential_state({"stop": True, "conclusion": "INCOMPLETE"}) == "UNRESOLVED"
    assert formal_sequential_state({"stop": True, "conclusion": "PREFERRED"}) == "RESOLVED"
    assert (
        formal_sequential_state(
            {"stop": False, "formal_sequential_evidence": {"state": "RESOLVED"}}
        )
        == "UNRESOLVED"
    )


@pytest.mark.parametrize(
    "ordinals,commit,flags",
    [([0, 0], None, []), ([0, 2], 2, [2]), ([0, 1], 0, [0]), ([0, 1], 1, [])],
)
def test_sweep_recovery_rejects_conflicting_commitments(
    ordinals: list[int], commit: int | None, flags: list[int]
) -> None:
    with pytest.raises(ValueError):
        sweep_block_inventory(
            {
                "block_progress_version": 1,
                "blocks": [{"ordinal": n, "committed": n in flags} for n in ordinals],
                "lookahead": {"maximum_blocks": 1, "committed_ordinal": commit},
            }
        )
