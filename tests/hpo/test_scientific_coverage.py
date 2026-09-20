from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.hpo.AdaptiveSearch import AdaptiveSearchPolicy
from lambdaforge.hpo.InitialDesign import CoverageState, InitialDesignPlanner
from lambdaforge.hpo.ParameterSpace import ParameterSpace
from lambdaforge.hpo.ScientificDesign import ExperimentalDesignPolicy
from lambdaforge.hpo.SurvivalModel import SurvivalModel, SurvivalObservation
from lambdaforge.work.runner import _adaptive_parallelism, _request_early_stops


def _pool(
    schema: dict[str, dict[str, object]],
) -> tuple[ParameterSpace, dict[int, dict[str, object]]]:
    names = list(schema)
    values = [tuple(schema[name]["values"]) for name in names]  # type: ignore[arg-type]
    candidates = {
        index: dict(zip(names, combination, strict=True))
        for index, combination in enumerate(itertools.product(*values), 1)
    }
    return ParameterSpace.from_schema(schema, tuple(candidates.values())), candidates


def test_initial_design_is_independent_of_resource_parallelism() -> None:
    space, candidates = _pool(
        {
            "left": {"values": [False, True]},
            "right": {"values": [False, True]},
            "width": {"values": [1, 2, 4]},
        }
    )
    expected = InitialDesignPlanner.plan(candidates, space, candidate_budget=12).trials

    for policy in (
        AdaptiveSearchPolicy(runs_per_gpu=2, max_parallel=4, candidate_budget=12),
        AdaptiveSearchPolicy(runs_per_gpu=10, max_parallel=20, candidate_budget=12),
        AdaptiveSearchPolicy(runs_per_gpu=None, max_parallel=None, candidate_budget=12),
    ):
        assert (
            InitialDesignPlanner.plan(
                candidates,
                space,
                candidate_budget=policy.candidate_budget,
                startup_trials=policy.startup_trials,
            ).trials
            == expected
        )


def test_initial_design_size_follows_authored_rank_and_discrete_support() -> None:
    small_space, small = _pool(
        {"left": {"values": [False, True]}, "right": {"values": [False, True]}}
    )
    large_schema = {
        **{f"flag_{index}": {"values": [False, True]} for index in range(10)},
        **{f"number_{index}": {"values": [0, 1, 2]} for index in range(5)},
        "kind": {"values": ["a", "b", "c"]},
    }
    # A bounded Sobol-like authored pool is enough to identify the larger basis without a full
    # 2^10 * 3^6 factorial allocation in the test.
    large = {
        index + 1: {
            **{f"flag_{flag}": bool((index >> flag) & 1) for flag in range(10)},
            **{f"number_{number}": (index + number) % 3 for number in range(5)},
            "kind": ("a", "b", "c")[index % 3],
        }
        for index in range(64)
    }
    large_space = ParameterSpace.from_schema(large_schema, tuple(large.values()))

    small_plan = InitialDesignPlanner.plan(small, small_space, candidate_budget=len(small))
    large_plan = InitialDesignPlanner.plan(large, large_space, candidate_budget=len(large))

    assert large_plan.full_rank > small_plan.full_rank
    assert len(large_plan.anchors) > len(small_plan.anchors)


def test_boolean_and_conditional_activity_obligations_are_protected() -> None:
    candidates = {
        1: {"enabled": False},
        2: {"enabled": True, "depth": 1},
        3: {"enabled": True, "depth": 2},
        4: {"enabled": True, "depth": 3},
    }
    schema = {
        "enabled": {"values": [False, True]},
        "depth": {"values": [1, 2, 3], "when": {"enabled": True}},
    }
    space = ParameterSpace.from_schema(schema, tuple(candidates.values()))
    plan = InitialDesignPlanner.plan(candidates, space, candidate_budget=4)
    keys = {key for anchor in plan.anchors for key in anchor.obligations}

    assert "enabled:value:False" in keys
    assert "enabled:value:True" in keys
    assert "depth:inactive" in keys
    assert "depth:active" in keys


def test_coverage_distinguishes_pruned_search_from_complete_response() -> None:
    space, candidates = _pool({"feature": {"values": [False, True]}})
    plan = InitialDesignPlanner.plan(candidates, space, candidate_budget=2)
    coverage = CoverageState.summarize(
        plan,
        candidates,
        {1: "pruned", 2: "completed"},
        space,
    )
    false_value = next(
        value for value in coverage["obligations"] if value["target"] == {"feature": False}
    )

    assert false_value["search_coverage"] is True
    assert false_value["response_coverage"] is False
    assert false_value["censored_pruned"] == 1


def test_opportunistic_fill_does_not_change_protected_anchor_count() -> None:
    space, candidates = _pool(
        {
            "left": {"values": [False, True]},
            "right": {"values": [False, True]},
            "width": {"values": [1, 2, 3]},
        }
    )
    plan = InitialDesignPlanner.plan(candidates, space, candidate_budget=12, startup_trials=4)
    extras = InitialDesignPlanner.opportunistic(candidates, space, reference=plan.trials, count=4)

    assert len(plan.anchors) == 4
    assert len(extras) == 4
    assert set(extras).isdisjoint(plan.trials)


def test_hard_infeasible_anchor_replacement_preserves_its_obligations() -> None:
    space, candidates = _pool(
        {
            "enabled": {"values": [False, True]},
            "context": {"values": ["a", "b", "c"]},
        }
    )
    plan = InitialDesignPlanner.plan(candidates, space, candidate_budget=6, startup_trials=2)
    old = plan.anchors[0]
    replacement = plan.replacement(
        old.trial,
        candidates=candidates,
        parameter_space=space,
    )

    assert replacement is not None
    obligations = {value.key: value for value in plan.obligations}
    assert any(replacement in obligations[key].candidate_trials for key in old.obligations)


def test_replacement_anchor_can_be_replaced_again_without_losing_obligations() -> None:
    space, candidates = _pool(
        {
            "enabled": {"values": [False, True]},
            "context": {"values": ["a", "b", "c", "d"]},
        }
    )
    plan = InitialDesignPlanner.plan(candidates, space, candidate_budget=8, startup_trials=2)
    old = plan.anchors[0]
    first = plan.replacement(old.trial, candidates=candidates, parameter_space=space)

    assert first is not None
    second = plan.replacement(
        first,
        candidates=candidates,
        parameter_space=space,
        unavailable=(*plan.trials, first),
        obligation_keys=old.obligations,
    )

    assert second is not None
    assert second not in {*plan.trials, first}


def test_coverage_summarizes_pair_cells_without_fabricating_responses() -> None:
    space, candidates = _pool(
        {"left": {"values": [False, True]}, "right": {"values": [False, True]}}
    )
    plan = InitialDesignPlanner.plan(candidates, space, candidate_budget=4)
    coverage = CoverageState.summarize(
        plan,
        candidates,
        {1: "completed", 2: "pruned", 3: "active"},
        space,
    )

    pair = coverage["interactions"][0]
    assert pair["authored_cells"] == 4
    assert pair["observed_cells"] == 3
    assert pair["response_cells"] == 1
    assert pair["censored_cells"] == 1
    assert pair["unexplored_cells"] == 1


def test_coverage_prioritizes_scientifically_relevant_interactions() -> None:
    space, candidates = _pool(
        {
            "alpha": {"values": [False, True]},
            "beta": {"values": [False, True]},
            "gamma": {"values": [False, True]},
        }
    )
    plan = InitialDesignPlanner.plan(candidates, space, candidate_budget=8)
    coverage = CoverageState.summarize(
        plan,
        candidates,
        {1: "completed"},
        space,
        relevant_interactions=(("beta", "gamma"),),
    )

    assert [value["parameters"] for value in coverage["interactions"]] == [["beta", "gamma"]]


def test_coverage_design_targets_an_unobserved_boolean_context() -> None:
    candidates = {
        1: {"enabled": True, "context": "a"},
        2: {"enabled": True, "context": "b"},
        3: {"enabled": False, "context": "a"},
        4: {"enabled": False, "context": "b"},
    }
    scientific = {
        "natural_utility_scale": 1.0,
        "optimization_opportunity": 0.1,
        "scientific_uncertainty": 0.9,
        "parameter_questions": [
            {
                "parameter": "enabled",
                "kind": "categorical",
                "entropy": 1.0,
                "authored_values": [False, True],
                "support": {"completed_candidates": 2},
                "conclusion_distribution": {"UNRESOLVED": 1.0},
            }
        ],
        "interaction_questions": [],
        "evidence": {"resamples": 8},
    }
    ranked = ExperimentalDesignPolicy(
        candidates,
        mode="max",
        practical_margin=None,
        parameter_space={
            "enabled": {"values": [False, True]},
            "context": {"values": ["a", "b"]},
        },
    ).rank({1: {1: 0.8}, 2: {1: 0.7}}, selected=(1, 2), scientific_state=scientific)

    assert ranked[0].trial in {3, 4}
    assert ranked[0].purpose == "COVER_PARAMETER_VALUE"
    assert ranked[0].to_dict()["action"] == "COVER_PARAMETER_VALUE"


def test_pending_candidate_does_not_count_as_matched_response_context() -> None:
    candidates = {
        1: {"enabled": True, "context": "a"},
        2: {"enabled": True, "context": "b"},
        3: {"enabled": False, "context": "a"},
        4: {"enabled": False, "context": "b"},
        5: {"enabled": False, "context": "c"},
    }
    scientific = {
        "natural_utility_scale": 1.0,
        "optimization_opportunity": 0.0,
        "scientific_uncertainty": 1.0,
        "parameter_questions": [
            {
                "parameter": "enabled",
                "kind": "categorical",
                "entropy": 1.0,
                "authored_values": [False, True],
                "support": {"completed_candidates": 2},
                "conclusion_distribution": {"UNRESOLVED": 1.0},
            }
        ],
        "interaction_questions": [],
        "evidence": {"resamples": 8},
    }
    ranked = ExperimentalDesignPolicy(
        candidates,
        mode="max",
        practical_margin=None,
        parameter_space={
            "enabled": {"values": [False, True]},
            "context": {"values": ["a", "b", "c"]},
        },
    ).rank(
        {1: {1: 0.8}, 2: {1: 0.7}},
        selected=(1, 2, 3),
        scientific_state=scientific,
    )

    assert ranked[0].trial in {4, 5}
    assert ranked[0].purpose == "COVER_PARAMETER_VALUE"


def test_pruned_candidate_reduces_search_coverage_debt_without_becoming_response() -> None:
    candidates = {
        1: {"enabled": True, "context": "a"},
        2: {"enabled": True, "context": "b"},
        3: {"enabled": False, "context": "a"},
        4: {"enabled": False, "context": "b"},
    }
    scientific = {
        "natural_utility_scale": 1.0,
        "optimization_opportunity": 0.0,
        "scientific_uncertainty": 1.0,
        "parameter_questions": [
            {
                "parameter": "enabled",
                "kind": "categorical",
                "entropy": 1.0,
                "authored_values": [False, True],
                "support": {"completed_candidates": 2},
                "conclusion_distribution": {"UNRESOLVED": 1.0},
            }
        ],
        "interaction_questions": [],
    }
    policy = ExperimentalDesignPolicy(
        candidates,
        mode="max",
        practical_margin=None,
        parameter_space={
            "enabled": {"values": [False, True]},
            "context": {"values": ["a", "b"]},
        },
    )
    no_pruned_coverage = policy.rank(
        {1: {1: 0.8}, 2: {1: 0.7}},
        selected=(1, 2, 3),
        scientific_state=scientific,
    )
    with_pruned_coverage = policy.rank(
        {1: {1: 0.8}, 2: {1: 0.7}},
        selected=(1, 2, 3),
        scientific_state=scientific,
        search_covered=(3,),
    )

    assert with_pruned_coverage[0].information_value < no_pruned_coverage[0].information_value


def test_auto_concurrency_has_finite_host_ceiling_and_round_trips() -> None:
    policy = AdaptiveSearchPolicy.from_search(
        {
            "trials": 20,
            "runs_per_gpu": "auto",
            "max_parallel": "auto",
            "x": {"values": [0, 1]},
        }
    )
    resources = ResourceRequest.from_mapping({"cpu": 6, "gpu": 2})

    assert policy.runs_per_gpu is None
    assert policy.max_parallel is None
    assert policy.to_dict()["runs_per_gpu"] == "auto"
    assert _adaptive_parallelism(resources, policy) == 6


def test_survival_uses_log_and_conditional_parameter_space_geometry() -> None:
    candidates = {
        1: {"lr": 1e-5, "enabled": False},
        2: {"lr": 1e-4, "enabled": False},
        3: {"lr": 1e-3, "enabled": False},
        4: {"lr": 1e-4, "enabled": True, "depth": 1},
    }
    schema = {
        "lr": {"range": [1e-5, 1e-3], "scale": "log"},
        "enabled": {"values": [False, True]},
        "depth": {"values": [1, 2], "when": {"enabled": True}},
    }
    model = SurvivalModel(candidates, parameter_space=schema)
    estimates = model.predict_all((SurvivalObservation(2, False),))

    assert estimates[1].effective_samples == pytest.approx(estimates[3].effective_samples)
    assert model.parameter_space.distance(
        candidates[1], candidates[4]
    ) > model.parameter_space.distance(candidates[1], candidates[2])


def test_scientific_continuation_bypasses_competitive_pruning(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "\n".join(
            [
                '{"name":"score","value":0.1,"step":3}',
                '{"name":"score","value":0.1,"step":4}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    stop = tmp_path / "stop"
    _request_early_stops(
        (
            {
                "trial_index": 1,
                "seed": 4,
                "hpo_metrics_path": metrics,
                "hpo_stop_path": stop,
                "hpo_scientific_continuation": True,
            },
        ),
        metric="score",
        mode="max",
        min_step=3,
    )

    assert not stop.exists()
