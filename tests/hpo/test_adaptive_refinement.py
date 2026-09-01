"""Scientific invariants for refined adaptive-HPO evidence and decisions."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from lambdaforge.hpo.AdaptiveSampler import AdaptiveSampler, CandidateObservation
from lambdaforge.hpo.AdaptiveSearch import AdaptiveSearchPolicy
from lambdaforge.hpo.BayesianSampler import BayesianSampler
from lambdaforge.hpo.CurveEvidence import CompletedCurve, audit_pruner
from lambdaforge.hpo.ObjectiveUtility import ObjectiveUtility, pareto_front
from lambdaforge.hpo.StudyInsights import StudyInsightAnalyzer
from lambdaforge.hpo.SurvivalModel import SurvivalModel, SurvivalObservation
from lambdaforge.work import WorkConfig
from lambdaforge.work.runner import _request_scheduler_preemption, _run_termination


def test_composite_utility_normalizes_fixed_ranges_and_weights() -> None:
    objective = ObjectiveUtility.normalize(
        {
            "metrics": {
                "quality": {"mode": "max", "weight": 3, "range": [0, 1]},
                "latency": {"mode": "min", "weight": 1, "range": [0, 100]},
            },
            "aggregation": "weighted_mean",
        }
    )

    evaluated = ObjectiveUtility(objective).evaluate({"quality": 0.8, "latency": 20})

    assert evaluated is not None
    assert evaluated["weights"] == {"quality": 0.75, "latency": 0.25}
    assert evaluated["value"] == pytest.approx(0.8)


@pytest.mark.parametrize("aggregation", ["geometric", "chebyshev"])
def test_composite_utility_aggregations_are_bounded(aggregation: str) -> None:
    objective = ObjectiveUtility.normalize(
        {
            "metrics": {
                "first": {"mode": "max", "weight": 1, "range": [0, 1]},
                "second": {"mode": "min", "weight": 1, "range": [0, 10]},
            },
            "aggregation": aggregation,
        }
    )
    evaluated = ObjectiveUtility(objective).evaluate({"first": 0.7, "second": 3})
    assert evaluated is not None
    assert 0 <= evaluated["value"] <= 1


def test_chebyshev_cannot_hide_a_zero_quality_high_weight_component() -> None:
    objective = ObjectiveUtility.normalize(
        {
            "metrics": {
                "critical": {"mode": "max", "weight": 3, "range": [0, 1]},
                "secondary": {"mode": "max", "weight": 1, "range": [0, 1]},
            },
            "aggregation": "chebyshev",
        }
    )
    evaluated = ObjectiveUtility(objective).evaluate({"critical": 0.0, "secondary": 1.0})
    assert evaluated is not None
    assert evaluated["value"] == 0.0


def test_balance_sensitive_utility_can_prefer_a_consistently_good_candidate() -> None:
    objective = ObjectiveUtility.normalize(
        {
            "metrics": {
                "auprc": {"mode": "max", "weight": 0.5, "range": [0, 1]},
                "balanced_accuracy": {
                    "mode": "max",
                    "weight": 0.5,
                    "range": [0, 1],
                },
            },
            "aggregation": "geometric",
        }
    )
    evaluator = ObjectiveUtility(objective)

    high_primary = evaluator.evaluate({"auprc": 0.80, "balanced_accuracy": 0.55})
    balanced = evaluator.evaluate({"auprc": 0.75, "balanced_accuracy": 0.70})

    assert high_primary is not None and balanced is not None
    assert balanced["value"] > high_primary["value"]


def test_zero_weight_metric_does_not_zero_geometric_utility() -> None:
    objective = ObjectiveUtility.normalize(
        {
            "metrics": {
                "quality": {"mode": "max", "weight": 1, "range": [0, 1]},
                "diagnostic_only": {"mode": "max", "weight": 0, "range": [0, 1]},
            },
            "aggregation": "geometric",
        }
    )

    evaluated = ObjectiveUtility(objective).evaluate({"quality": 0.8, "diagnostic_only": 0.0})

    assert evaluated is not None
    assert evaluated["value"] == pytest.approx(0.8)


def test_composite_observation_never_mixes_different_checkpoints() -> None:
    objective = ObjectiveUtility.normalize(
        {
            "metrics": {
                "a": {"mode": "max", "weight": 1, "range": [0, 1]},
                "b": {"mode": "max", "weight": 1, "range": [0, 1]},
            }
        }
    )
    records = (
        {"name": "a", "value": 1.0, "step": 1},
        {"name": "b", "value": 0.9, "step": 2},
        {"name": "a", "value": 0.4, "step": 2},
    )

    observation = ObjectiveUtility(objective).observation(records, fallback={"a": 1, "b": 1})

    assert observation is not None
    assert observation["best_step"] == 2
    assert observation["components"]["a"]["raw"] == 0.4
    assert observation["components"]["b"]["raw"] == 0.9


def test_composite_observation_rejects_an_incomplete_metric_vector() -> None:
    objective = ObjectiveUtility.normalize(
        {
            "metrics": {
                "a": {"mode": "max", "weight": 1, "range": [0, 1]},
                "b": {"mode": "max", "weight": 1, "range": [0, 1]},
            }
        }
    )

    assert (
        ObjectiveUtility(objective).observation(
            ({"name": "a", "value": 0.9, "step": 3},), fallback={"a": 0.9, "b": 0.8}
        )
        is None
    )


def test_fixed_normalization_does_not_change_when_a_later_extreme_appears() -> None:
    objective = ObjectiveUtility.normalize(
        {
            "metrics": {
                "a": {"mode": "max", "weight": 1, "range": [0, 10]},
                "b": {"mode": "max", "weight": 1, "range": [0, 10]},
            },
            "aggregation": "weighted_mean",
        }
    )
    evaluator = ObjectiveUtility(objective)

    before = evaluator.evaluate({"a": 5, "b": 5})
    evaluator.evaluate({"a": 10_000, "b": -10_000})
    after = evaluator.evaluate({"a": 5, "b": 5})

    assert before == after


def test_pareto_diagnostic_uses_raw_metric_directions() -> None:
    objective = ObjectiveUtility.normalize(
        {
            "metrics": {
                "quality": {"mode": "max", "weight": 1, "range": [0, 1]},
                "latency": {"mode": "min", "weight": 1, "range": [0, 10]},
            }
        }
    )
    candidates = (
        {"trial": 1, "raw_metrics": {"quality": 0.8, "latency": 5}},
        {"trial": 2, "raw_metrics": {"quality": 0.9, "latency": 4}},
        {"trial": 3, "raw_metrics": {"quality": 0.7, "latency": 2}},
    )
    assert pareto_front(candidates, objective) == (2, 3)


def test_survival_model_is_smoothed_joint_and_uncertain() -> None:
    model = SurvivalModel({1: {"x": 0, "kind": "a"}, 2: {"x": 1, "kind": "b"}})
    estimates = model.predict_all((SurvivalObservation(1, False),))

    assert estimates[1].probability < estimates[2].probability
    assert estimates[1].lower < estimates[1].probability < estimates[1].upper
    assert estimates[2].effective_samples < estimates[1].effective_samples


def test_retrospective_pruner_audit_reports_savings_and_false_prunes() -> None:
    curves = (
        CompletedCurve(1, 1, ((1, 0.80), (2, 0.85), (3, 0.90), (4, 0.91)), 40, True),
        CompletedCurve(2, 1, ((1, 0.10), (2, 0.12), (3, 0.14), (4, 0.15)), 40, True),
        CompletedCurve(3, 1, ((1, 0.50), (2, 0.55), (3, 0.60), (4, 0.65)), 40, True),
    )

    report = audit_pruner(
        curves,
        mode="max",
        min_step=1,
        confirmations=1,
        probability_threshold=0.45,
        margin=0.0,
    )

    assert report["completed_candidates"] == 3
    assert report["calibration_candidates"] >= 2
    assert report["simulated_performance_prunes"] >= 1
    assert report["epochs_saved"] > 0
    assert report["gpu_seconds_saved"] > 0
    assert report["probability_brier_score"] is not None


def test_pruning_statistics_count_candidates_not_seeds() -> None:
    candidates = (
        {
            "trial": 1,
            "parameters": {"width": 16},
            "state": "pruned",
            "runs": [{"state": "pruned"}, {"state": "pruned"}, {"state": "pruned"}],
        },
        {
            "trial": 2,
            "parameters": {"width": 32},
            "state": "observed",
            "selection_objective": 0.8,
            "runs": [{"state": "succeeded"}],
        },
    )
    analysis = StudyInsightAnalyzer.analyze(candidates, {"metric": "score", "mode": "max"})
    signal = analysis["parameters"][0]["pruning_signal"]
    assert signal["observations"] == 2
    assert signal["pruned"] == 1
    assert signal["rate_lower"] < signal["smoothed_rate"] < signal["rate_upper"]


def test_joint_insights_detect_xor_without_claiming_a_marginal_direction() -> None:
    candidates = []
    trial = 0
    for repeat in range(2):
        for left in (False, True):
            for right in (False, True):
                trial += 1
                candidates.append(
                    {
                        "trial": trial,
                        "parameters": {"left": left, "right": right, "repeat": repeat},
                        "state": "observed",
                        "selection_objective": float(left != right),
                        "runs": [{"state": "succeeded"}],
                    }
                )
    analysis = StudyInsightAnalyzer.analyze(candidates, {"metric": "score", "mode": "max"})
    left = next(value for value in analysis["parameters"] if value["parameter"] == "left")
    relationship = next(
        value for value in left["joint_relationships"] if value["parameter"] == "right"
    )
    assert relationship["status"] == "predictive"
    assert relationship["gain"] > 0
    assert left["status"] != "actionable"
    assert "depends on right" in left["interaction_context"]


def test_mixed_knn_sampler_uses_xor_interaction_not_marginal_direction() -> None:
    candidates = {
        1: {"left": 0, "right": 0, "replicate": 0},
        2: {"left": 0, "right": 1, "replicate": 0},
        3: {"left": 1, "right": 0, "replicate": 0},
        4: {"left": 1, "right": 1, "replicate": 0},
        5: {"left": 0, "right": 0, "replicate": 1},
        6: {"left": 0, "right": 1, "replicate": 1},
    }
    observations = tuple(
        CandidateObservation(
            trial,
            1.0 if candidates[trial]["left"] == candidates[trial]["right"] else 0.0,
        )
        for trial in (1, 2, 3, 4)
    )
    proposed = AdaptiveSampler(candidates, mode="max").propose(
        observations,
        selected=(1, 2, 3, 4),
        count=1,
    )
    assert proposed == (5,)


def test_fallback_pending_distance_respects_the_pending_run_fidelity() -> None:
    candidates = {
        1: {"x": 0.0},
        2: {"x": 0.9},
        3: {"x": 0.85},
        4: {"x": 0.5},
    }
    sampler = AdaptiveSampler(candidates, mode="max")

    full_pending = sampler.propose(
        (CandidateObservation(1, 0.5),),
        selected=(1, 2),
        pending=(2,),
        pending_fidelity={2: 1.0},
        count=1,
    )
    early_pending = sampler.propose(
        (CandidateObservation(1, 0.5),),
        selected=(1, 2),
        pending=(2,),
        pending_fidelity={2: 0.1},
        count=1,
    )

    assert full_pending == (4,)
    assert early_pending == (3,)


def test_botorch_numeric_surrogate_accepts_explicit_mixed_fidelity_evidence() -> None:
    if not BayesianSampler.available():
        pytest.skip("optional BoTorch provider is not installed")
    candidates = {index: {"width": float(index)} for index in range(1, 8)}
    observations = tuple(
        CandidateObservation(
            index,
            value=float(index) / 10,
            standard_error=0.02,
            fidelity=0.25 if index < 4 else 1.0,
        )
        for index in range(1, 6)
    )

    proposed = BayesianSampler(candidates, mode="max").propose(
        observations,
        selected=(1, 2, 3, 4, 5, 6),
        pending=(6,),
        count=1,
    )

    assert proposed == (7,)


def test_removed_ambiguous_policy_fields_fail_with_migration() -> None:
    with pytest.raises(ValueError, match="fidelity.reduction_factor"):
        AdaptiveSearchPolicy.from_search({"reduction_factor": 2})


def test_failure_wins_over_a_coincident_cooperative_stop_request(tmp_path) -> None:
    stop = tmp_path / "run.stop"
    stop.write_text("candidate pruning requested", encoding="utf-8")

    termination, detail = _run_termination(
        status="failed",
        failure={"type": "CUDAOutOfMemoryError", "message": "CUDA out of memory"},
        stop_path=stop,
    )

    assert termination == "resource_failed"
    assert detail["failure_type"] == "CUDAOutOfMemoryError"


def _resumable_active_specification(tmp_path: Path) -> dict[str, object]:
    execution = tmp_path / "execution"
    checkpoint = execution / "runs" / "run-one" / "checkpoints" / "epoch.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"safe checkpoint")
    manifest = execution / "hpo-control" / "trial.checkpoint.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"checkpoint_root": str(checkpoint.parent), "run_id": "run-one"}),
        encoding="utf-8",
    )
    metrics = execution / "hpo-control" / "trial.metrics.jsonl"
    metrics.write_text(
        "".join(
            json.dumps({"name": "score", "value": value, "step": step}) + "\n"
            for step, value in ((1, 0.4), (2, 0.5), (4, 0.55))
        ),
        encoding="utf-8",
    )
    return {
        "execution_dir": execution,
        "trial_index": 1,
        "seed": 7,
        "hpo_phase": "search",
        "hpo_fidelity": {"current": 0, "target": 10, "maximum": 10},
        "hpo_metrics_path": metrics,
        "hpo_stop_path": execution / "hpo-control" / "trial.stop",
        "hpo_checkpoint_manifest_path": manifest,
        "hpo_dispatched_monotonic": time.monotonic() - 120,
        "hpo_scheduler_action": "START_NEW",
        "hpo_scheduler_priority": 0.01,
    }


def test_scheduler_preemption_requires_checkpoint_and_material_priority_gain(tmp_path) -> None:
    specification = _resumable_active_specification(tmp_path)
    alternatives = (
        {"action": "START_NEW", "trial": 9, "score": 1.0},
        {"action": "ADD_SEED", "trial": 4, "score": 0.1},
    )

    evidence = _request_scheduler_preemption(
        (specification,),
        alternatives=alternatives,
        selected_action="START_NEW",
        selected_trial=9,
        metric="score",
        min_step=2,
    )

    assert evidence is not None
    assert evidence["termination_type"] == "scheduler_preempted"
    assert evidence["checkpoint_required"] is True
    stop = specification["hpo_stop_path"]
    assert isinstance(stop, Path)
    assert stop.is_file()
    persisted = json.loads(stop.with_name("trial.stop.evidence.json").read_text(encoding="utf-8"))
    assert persisted["new_action"] == "ADD_SEED"


def test_scheduler_preemption_hysteresis_ignores_small_priority_change(tmp_path) -> None:
    specification = _resumable_active_specification(tmp_path)

    evidence = _request_scheduler_preemption(
        (specification,),
        alternatives=(
            {"action": "START_NEW", "trial": 9, "score": 1.0},
            {"action": "ADD_SEED", "trial": 4, "score": 0.011},
        ),
        selected_action="START_NEW",
        selected_trial=9,
        metric="score",
        min_step=2,
    )

    assert evidence is None
    stop = specification["hpo_stop_path"]
    assert isinstance(stop, Path)
    assert not stop.exists()


def test_confirmation_is_never_scheduler_preempted(tmp_path) -> None:
    specification = _resumable_active_specification(tmp_path)
    specification["hpo_phase"] = "confirmation"

    evidence = _request_scheduler_preemption(
        (specification,),
        alternatives=(
            {"action": "START_NEW", "trial": 9, "score": 1.0},
            {"action": "ADD_SEED", "trial": 4, "score": 0.1},
        ),
        selected_action="START_NEW",
        selected_trial=9,
        metric="score",
        min_step=2,
    )

    assert evidence is None


def test_unscored_startup_coverage_is_not_opportunistically_preempted(tmp_path) -> None:
    specification = _resumable_active_specification(tmp_path)
    specification.pop("hpo_scheduler_priority")

    evidence = _request_scheduler_preemption(
        (specification,),
        alternatives=(
            {"action": "START_NEW", "trial": 9, "score": 1.0},
            {"action": "ADD_SEED", "trial": 4, "score": 0.9},
        ),
        selected_action="START_NEW",
        selected_trial=9,
        metric="score",
        min_step=2,
    )

    assert evidence is None


def test_stop_request_arriving_after_target_does_not_censor_completed_run(tmp_path) -> None:
    stop = tmp_path / "run.stop"
    stop.write_text("scheduler requested pause", encoding="utf-8")
    stop.with_name("run.stop.evidence.json").write_text(
        json.dumps({"termination_type": "scheduler_preempted"}), encoding="utf-8"
    )

    termination, detail = _run_termination(
        status="succeeded",
        failure=None,
        stop_path=stop,
        completed_step=10,
        target_step=10,
    )

    assert termination == "completed"
    assert detail["stop_request_observed_after_target"] == "scheduler_preempted"


def test_candidate_budget_is_separate_from_rich_proposal_pool(tmp_path) -> None:
    config = WorkConfig.from_mapping(
        {
            "name": "pool",
            "run": "tests.work_cases.AdaptiveScoreWork",
            "search": {
                "trials": 3,
                "proposal_pool_size": 24,
                "confirmation_seeds": [],
                "quality": {"range": [0, 1]},
            },
            "objective": {"metric": "score", "mode": "max"},
        },
        source=tmp_path / "study.yaml",
    )
    definition = config.levels[0].runs[0]
    assert definition.search_policy is not None
    assert definition.search_policy.candidate_budget == 3
    assert len(definition.variants) == 24
    assert definition.run_count == 3
