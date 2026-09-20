from __future__ import annotations

import time
from typing import Any

import pytest

from lambdaforge.hpo.AdaptiveStatistics import AdaptiveSeedRacer
from lambdaforge.hpo.ScientificDesign import (
    ExperimentalDesignPolicy,
    ScientificQuestionAnalyzer,
    SeedNoiseModel,
)


def _candidate(
    trial: int,
    parameters: dict[str, Any],
    values: list[float],
    *,
    pruned: bool = False,
) -> dict[str, Any]:
    return {
        "trial": trial,
        "parameters": parameters,
        "state": "pruned" if pruned else "completed",
        "runs": [
            {
                "seed": seed,
                "state": "pruned" if pruned else "succeeded",
                "final_objective": None if pruned else value,
                "best_observed_objective": value,
                "censored": pruned,
            }
            for seed, value in enumerate(values, start=1)
        ],
    }


def _analyze(
    candidates: list[dict[str, Any]],
    *,
    margin: float | None = 0.05,
    mode: str = "max",
    pool: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return ScientificQuestionAnalyzer.analyze(
        candidates,
        {"metric": "score", "mode": mode},
        practical_margin=margin,
        fingerprint="sha256:test-scientific-design",
        candidate_pool=pool,
        final=True,
    )


def test_flat_parameter_can_be_a_high_confidence_positive_conclusion() -> None:
    candidates = [
        _candidate(index, {"width": width}, [0.60, 0.61, 0.59])
        for index, width in enumerate((16, 32, 64, 128), start=1)
    ]

    question = _analyze(candidates)["parameter_questions"][0]

    assert question["conclusion_kind"] == "PRACTICALLY_EQUIVALENT"
    assert question["confidence"] >= 0.8
    assert question["evidence"]["coverage_is_not_confidence"] is True


def test_small_stable_advantage_within_margin_is_a_weak_preference() -> None:
    candidates = [
        _candidate(1, {"optimizer": "adam"}, [0.60, 0.61, 0.59]),
        _candidate(2, {"optimizer": "lion"}, [0.62, 0.63, 0.61]),
    ]

    question = _analyze(candidates)["parameter_questions"][0]

    assert question["conclusion_kind"] in {"WEAK_PREFERENCE", "PRACTICALLY_EQUIVALENT"}
    assert question["practically_equivalent_probability"] >= 0.8


def test_scientific_uncertainty_tracks_which_value_is_preferred_not_only_state_kind() -> None:
    analysis = _analyze(
        [
            _candidate(1, {"optimizer": "adam"}, [0.60]),
            _candidate(2, {"optimizer": "adam"}, [0.40]),
            _candidate(3, {"optimizer": "lion"}, [0.60]),
            _candidate(4, {"optimizer": "lion"}, [0.40]),
        ],
        margin=None,
    )
    question = analysis["parameter_questions"][0]

    assert question["conclusion_kind"] == "PREFERRED"
    assert set(question["conclusion_distribution"]) == {
        'PREFERRED:"adam"',
        'PREFERRED:"lion"',
    }
    assert question["entropy"] > 0.5
    assert analysis["scientific_uncertainty"] > 0.5


def test_large_effect_and_min_mode_select_the_scientifically_correct_value() -> None:
    maximum = _analyze(
        [
            _candidate(1, {"depth": "shallow"}, [0.3, 0.31]),
            _candidate(2, {"depth": "deep"}, [0.8, 0.79]),
        ],
        margin=0.02,
    )["parameter_questions"][0]
    minimum = _analyze(
        [
            _candidate(1, {"depth": "shallow"}, [0.3, 0.31]),
            _candidate(2, {"depth": "deep"}, [0.8, 0.79]),
        ],
        margin=0.02,
        mode="min",
    )["parameter_questions"][0]

    assert maximum["best_value_probability"]["deep"] > 0.8
    assert minimum["best_value_probability"]["shallow"] > 0.8


def test_joint_effect_is_reported_as_context_dependent() -> None:
    candidates: list[dict[str, Any]] = []
    trial = 0
    # Replicated candidate-level support is required for a stable interaction conclusion; one
    # observation in each cell is coverage, not confidence.
    for left, right in ((0, 0), (0, 1), (1, 0), (1, 1)):
        for replicate in range(3):
            trial += 1
            value = (0.9 if left == right else 0.1) + replicate * 0.001
            candidates.append(_candidate(trial, {"left": left, "right": right}, [value, value]))

    analysis = _analyze(candidates, margin=0.1)
    interaction = analysis["interaction_questions"][0]

    assert interaction["conclusion_kind"] == "MATERIAL_INTERACTION"
    assert interaction["confidence"] >= 0.8
    assert any(
        question["conclusion_kind"] == "CONTEXT_DEPENDENT"
        for question in analysis["parameter_questions"]
    )


def test_seed_noise_never_uses_between_candidate_spread() -> None:
    unresolved = SeedNoiseModel.fit({1: {1: 0.0}, 2: {1: 100.0}})
    calibrated = SeedNoiseModel.fit({1: {1: 0.0, 2: 0.2}, 2: {1: 100.0, 2: 100.2}})

    assert unresolved.variance is None
    assert calibrated.calibrated
    assert calibrated.variance == pytest.approx(0.0, abs=1e-12)
    assert calibrated.to_dict()["between_candidate_spread_used_as_seed_noise"] is False


def test_seed_racing_values_incumbent_replication_and_shared_seeds() -> None:
    racer = AdaptiveSeedRacer(mode="max", margin=0.02, probability_threshold=0.05)
    outcomes = {
        1: {1: 0.80, 2: 0.78},
        2: {1: 0.77, 3: 0.76},
        3: {1: 0.20},
    }

    decisions = racer.decisions(
        outcomes,
        eligible=(1, 2, 3),
        available_seeds={1: (1, 2, 3), 2: (1, 2, 3), 3: (1, 2, 3)},
    )

    by_trial = {decision.trial: decision for decision in decisions}
    assert 1 in by_trial
    assert by_trial[1].purpose in {"ADD_SHARED_SEED", "REPLICATE_INCUMBENT"}
    assert by_trial[1].recommended_seed == 3
    assert 3 not in by_trial


def test_experimental_design_can_choose_a_matched_scientific_probe() -> None:
    pool = {
        1: {"feature": False, "width": 32},
        2: {"feature": False, "width": 64},
        3: {"feature": True, "width": 32},
        4: {"feature": True, "width": 64},
    }
    candidates = [
        _candidate(1, pool[1], [0.60, 0.62]),
        _candidate(2, pool[2], [0.61, 0.59]),
    ]
    scientific = _analyze(candidates, margin=0.05, pool=pool)
    outcomes = {1: {1: 0.60, 2: 0.62}, 2: {1: 0.61, 2: 0.59}}

    ranked = ExperimentalDesignPolicy(pool, mode="max", practical_margin=0.05).rank(
        outcomes,
        selected=(1, 2),
        scientific_state=scientific,
    )

    assert ranked
    assert ranked[0].trial in {3, 4}
    assert ranked[0].information_value > 0
    assert ranked[0].counterfactual_match_quality > 0
    assert ranked[0].purpose in {
        "RESOLVE_PARAMETER",
        "RESOLVE_INTERACTION",
        "EXPLORE_COVERAGE",
        "COVER_PARAMETER_VALUE",
        "COVER_INTERACTION_CELL",
        "OPTIMIZE",
    }


def test_scientific_analysis_is_deterministic_and_tracks_pruned_regions() -> None:
    candidates = [
        _candidate(1, {"x": 0}, [0.7, 0.72]),
        _candidate(2, {"x": 1}, [0.2], pruned=True),
        _candidate(3, {"x": 2}, [0.75, 0.74]),
    ]

    first = _analyze(candidates)
    second = _analyze(candidates)

    assert first == second
    question = first["parameter_questions"][0]
    assert question["support"]["pruned_candidates"] == 1
    assert first["practical_optimal_region"]["members"]


def test_one_pruned_seed_censors_completed_siblings_from_response_fitting() -> None:
    analysis = _analyze(
        [
            {
                "trial": 1,
                "parameters": {"x": 0},
                "state": "pruned",
                "runs": [
                    {"seed": 1, "state": "succeeded", "final_objective": 0.8},
                    {
                        "seed": 2,
                        "state": "pruned",
                        "best_observed_objective": 0.2,
                        "censored": True,
                    },
                ],
            },
            _candidate(2, {"x": 1}, [0.7]),
        ]
    )

    support = analysis["parameter_questions"][0]["support"]
    assert support["completed_candidates"] == 1
    assert support["pruned_candidates"] == 1


def test_practical_region_distinguishes_constrained_and_flexible_parameters() -> None:
    analysis = _analyze(
        [
            _candidate(1, {"depth": 1, "normalization": False}, [0.80, 0.81]),
            _candidate(2, {"depth": 1, "normalization": True}, [0.79, 0.80]),
            _candidate(3, {"depth": 4, "normalization": False}, [0.20, 0.21]),
            _candidate(4, {"depth": 4, "normalization": True}, [0.20, 0.19]),
        ],
        margin=0.05,
    )

    flexibility = analysis["practical_optimal_region"]["flexibility"]
    assert flexibility["depth"]["classification"] in {
        "strongly-constrained",
        "moderately-constrained",
    }
    assert flexibility["normalization"]["classification"] == "flexible"


def test_new_interaction_evidence_reopens_a_previously_flat_question() -> None:
    initial = [
        _candidate(index, {"left": left, "right": 0}, [0.5, 0.5])
        for index, left in enumerate((0, 1, 0, 1), start=1)
    ]
    before = _analyze(initial, margin=0.1)
    before_left = next(
        question for question in before["parameter_questions"] if question["parameter"] == "left"
    )

    expanded = list(initial)
    for left, right in ((0, 0), (0, 1), (1, 0), (1, 1)):
        value = 0.9 if left == right else 0.1
        expanded.append(
            _candidate(len(expanded) + 1, {"left": left, "right": right}, [value, value])
        )
    after = _analyze(expanded, margin=0.1)
    after_left = next(
        question for question in after["parameter_questions"] if question["parameter"] == "left"
    )

    assert before_left["conclusion_kind"] == "PRACTICALLY_EQUIVALENT"
    assert before_left["entropy"] == 0.0
    assert after_left["conclusion_kind"] == "CONTEXT_DEPENDENT"
    assert after_left["entropy"] > 0.5
    assert after["scientific_uncertainty"] > before["scientific_uncertainty"]


def test_insufficient_and_conditional_support_remain_explicit() -> None:
    pool = {
        1: {"family": "linear"},
        2: {"family": "tree", "depth": 2},
        3: {"family": "tree", "depth": 4},
    }
    analysis = _analyze(
        [
            _candidate(1, pool[1], [0.5]),
            _candidate(2, pool[2], [0.6]),
        ],
        pool=pool,
    )
    depth = next(
        question for question in analysis["parameter_questions"] if question["parameter"] == "depth"
    )

    assert depth["conclusion_kind"] == "UNRESOLVED"
    assert depth["confidence_label"] == "low"
    assert depth["support"]["inactive_completed_candidates"] == 1
    assert depth["support"]["inactive_authored_candidates"] == 1
    assert depth["missing_evidence"]


def test_controller_value_moves_between_interaction_probe_and_optimization() -> None:
    pool = {
        index + 1: {"x": x / 4, "feature": feature}
        for index, (x, feature) in enumerate(
            (x, feature) for x in range(5) for feature in (False, True)
        )
    }
    selected = (1, len(pool))
    outcomes = {1: {1: 0.2}, len(pool): {1: 0.3}}
    evidence = [
        _candidate(1, pool[1], [0.2]),
        _candidate(len(pool), pool[len(pool)], [0.3]),
    ]
    state = _analyze(evidence, margin=0.02, pool=pool)
    policy = ExperimentalDesignPolicy(pool, mode="max", practical_margin=0.02)

    evidence_first = policy.rank(
        outcomes,
        selected=selected,
        scientific_state={
            **state,
            "optimization_opportunity": 0.01,
            "scientific_uncertainty": 1.0,
        },
    )[0]
    optimization_first = policy.rank(
        outcomes,
        selected=selected,
        scientific_state={
            **state,
            "optimization_opportunity": 1.0,
            "scientific_uncertainty": 0.01,
        },
    )[0]

    assert evidence_first.purpose in {"RESOLVE_INTERACTION", "COVER_INTERACTION_CELL"}
    assert evidence_first.target_questions
    assert optimization_first.purpose == "OPTIMIZE"
    assert optimization_first.optimization_value > 0


def test_large_proposal_pool_keeps_live_scientific_analysis_bounded() -> None:
    pool = {
        trial: {
            "width": (32, 64, 128, 256)[trial % 4],
            "depth": 1 + trial % 5,
            "dropout": (trial % 20) / 40,
            "optimizer": ("adam", "lion")[trial % 2],
            "residual": bool(trial % 2),
            "heads": (2, 4, 8)[trial % 3],
        }
        for trial in range(1, 4097)
    }
    candidates = [
        _candidate(trial, pool[trial], [0.4 + (trial % 11) / 100]) for trial in range(1, 23)
    ]

    started = time.perf_counter()
    analysis = ScientificQuestionAnalyzer.analyze(
        candidates,
        {"metric": "score", "mode": "max"},
        practical_margin=0.01,
        fingerprint="sha256:large-live-pool",
        candidate_pool=pool,
        final=False,
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 10.0
    assert analysis["evidence"]["candidate_pool_size"] == 4096
    assert analysis["evidence"]["reference_points"] <= 32
    assert analysis["evidence"]["matching_support_points"] <= 4
    assert analysis["evidence"]["live_cost_is_pool_bounded"] is True


def test_scientific_shortlist_rotates_and_always_keeps_sampler_proposal() -> None:
    pool = {trial: {"x": trial / 4096, "family": trial % 4} for trial in range(1, 4097)}
    policy = ExperimentalDesignPolicy(pool, mode="max", practical_margin=0.01)
    state = {
        "optimization_opportunity": 0.5,
        "scientific_uncertainty": 0.5,
        "optimization_weight": 0.5,
        "information_weight": 0.5,
        "parameter_questions": [],
        "interaction_questions": [],
        "evidence": {"resamples": 32},
    }
    outcomes = {1: {1: 0.4}, 2: {1: 0.5}}

    first = policy.rank(
        outcomes,
        selected=(1, 2),
        scientific_state=state,
        required_candidates=(4096,),
        decision_key="decision-1",
    )
    second = policy.rank(
        outcomes,
        selected=(1, 2),
        scientific_state=state,
        required_candidates=(4096,),
        decision_key="decision-2",
    )

    assert len(first) == 32
    assert len(second) == 32
    assert any(value.trial == 4096 for value in first)
    assert {value.trial for value in first} != {value.trial for value in second}
