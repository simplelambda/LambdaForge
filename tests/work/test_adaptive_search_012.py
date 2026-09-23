"""Focused adaptive study behavior without requiring a GPU or external provider."""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.hpo.AdaptiveResources import ActiveResourceCommitment, ResourcePrediction
from lambdaforge.hpo.AdaptiveSampler import AdaptiveSampler, CandidateObservation
from lambdaforge.hpo.AdaptiveSearch import AdaptiveSearchPolicy
from lambdaforge.hpo.AdaptiveStatistics import AdaptiveSeedRacer
from lambdaforge.hpo.BayesianSampler import BayesianSampler
from lambdaforge.hpo.SobolSearch import SobolSearch
from lambdaforge.work import WorkConfig, WorkRunner
from lambdaforge.work.models import WorkResources, WorkResult
from lambdaforge.work.runner import (
    _adaptive_parallelism,
    _admissible_gpu_slots,
    _candidate_score,
    _execute_cpu_isolated_runs,
    _execute_gpu_admitted_runs,
    _gpu_memory_inventory,
    _objective_observation,
    _request_early_stops,
    _resource_frontier_room,
    _retry_failed_result,
    _validate_gpu_memory_capacity,
)


def _completed_pruner_calibration(tmp_path: Path) -> tuple[WorkResult, WorkResult]:
    """Return two complete curves: the minimum retrospective comparison evidence."""
    results: list[WorkResult] = []
    for trial, values in (
        (1001, (0.50, 0.54, 0.57, 0.59, 0.61, 0.62)),
        (1002, (0.44, 0.48, 0.51, 0.53, 0.55, 0.56)),
    ):
        run_dir = tmp_path / f"calibration-{trial}"
        run_dir.mkdir()
        (run_dir / "metrics.jsonl").write_text(
            "".join(
                json.dumps({"name": "score", "value": value, "step": step}) + "\n"
                for step, value in enumerate(values, 1)
            ),
            encoding="utf-8",
        )
        results.append(
            WorkResult(
                name="work",
                work_class="tests.work_cases.AdaptiveScoreWork",
                execution_id="execution-calibration",
                run_id=f"run-{trial}",
                attempt_id="attempt-1",
                attempt_number=1,
                scientific_fingerprint=f"sha256:{trial}",
                status="succeeded",
                run_dir=run_dir,
                created_at_utc="2026-01-01T00:00:00+00:00",
                started_at_utc="2026-01-01T00:00:00+00:00",
                finished_at_utc="2026-01-01T00:00:06+00:00",
                duration_seconds=6.0,
                seed=4,
                trial={"index": trial, "parameters": {"width": trial}},
                parameters={},
                inputs=(),
                requested_resources=WorkResources(1, 0, 0, 0, None, 0, 1),
                metrics={"score": values[-1]},
                termination_type="completed",
            )
        )
    return results[0], results[1]


def test_hpo_separates_current_curve_value_from_best_checkpoint(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "".join(
            json.dumps({"name": "score", "value": value, "step": step, "split": None}) + "\n"
            for step, value in ((1, 0.5), (2, 0.9), (3, 0.6))
        ),
        encoding="utf-8",
    )

    observation = _objective_observation(
        {"metric": "score", "mode": "max"},
        paths=(metrics,),
        fallback={},
    )

    assert observation == {
        "metric": "score",
        "mode": "max",
        "current": 0.6,
        "current_step": 3,
        "best": 0.9,
        "best_step": 2,
        "selection": "best-observed-checkpoint",
    }


def test_objective_guardrail_is_evaluated_at_the_primary_best_epoch(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.jsonl"
    rows = (
        ("auprc", 0.65, 1),
        ("accuracy", 0.50, 1),
        ("auprc", 0.60, 2),
        ("accuracy", 0.80, 2),
    )
    metrics.write_text(
        "".join(
            json.dumps({"name": name, "value": value, "step": step, "split": None}) + "\n"
            for name, value, step in rows
        ),
        encoding="utf-8",
    )

    observation = _objective_observation(
        {
            "metric": "auprc",
            "mode": "max",
            "constraints": {"accuracy": {"min": 0.6}},
        },
        paths=(metrics,),
        fallback={},
    )

    assert observation is not None
    assert observation["best_step"] == 1
    assert observation["constraints"]["accuracy"]["value"] == 0.5
    assert observation["constraints"]["accuracy"]["satisfied"] is False
    assert observation["feasible"] is False


def test_objective_guardrails_are_validated_and_preserved(tmp_path: Path) -> None:
    base = {
        "name": "guarded",
        "run": "tests.work_cases.AdaptiveScoreWork",
        "search": {"quality": {"values": [1.0, 2.0]}},
        "objective": {
            "metric": "score",
            "mode": "max",
            "constraints": {"accuracy": {"min": 0.55}, "latency": {"max": 2.0}},
        },
    }

    config = WorkConfig.from_mapping(base, source=tmp_path / "guarded.yaml")

    assert config.levels[0].runs[0].objective == base["objective"]
    repeated = WorkConfig.from_mapping(
        {
            **base,
            "objective": {
                "metric": "score",
                "mode": "max",
                "constraints": {"score": {"min": 0.1, "seed_aggregation": "worst"}},
            },
        },
        source=tmp_path / "repeated.yaml",
    )
    assert repeated.levels[0].runs[0].objective["constraints"]["score"] == {
        "min": 0.1,
        "seed_aggregation": "worst",
    }


def test_adaptive_sampler_proposes_only_after_observing_the_startup_region() -> None:
    candidates = {index + 1: {"x": index / 5} for index in range(6)}
    sampler = AdaptiveSampler(candidates, mode="max")
    startup = sampler.initial(2)

    left = sampler.propose(
        (
            CandidateObservation(startup[0], 1.0),
            CandidateObservation(startup[1], 0.0),
        ),
        selected=startup,
        count=1,
    )
    right = sampler.propose(
        (
            CandidateObservation(startup[0], 0.0),
            CandidateObservation(startup[1], 1.0),
        ),
        selected=startup,
        count=1,
    )

    assert startup == (1, 6)
    assert left != right
    assert candidates[left[0]]["x"] < candidates[right[0]]["x"]


def test_adaptive_sampler_uses_pruned_only_trials_as_censored_avoidance() -> None:
    candidates = {index: {"x": (index - 1) / 4} for index in range(1, 6)}
    sampler = AdaptiveSampler(candidates, mode="max")

    proposed = sampler.propose(
        (CandidateObservation(1, 0.5),),
        selected=(1, 5),
        censored=(5,),
        count=1,
    )

    assert proposed == (3,)


def test_bayesian_sampler_uses_seed_uncertainty_for_joint_noisy_acquisition() -> None:
    if not BayesianSampler.available():
        pytest.skip("optional BoTorch provider is not installed")
    candidates = {
        index: {"width": index * 16, "optimizer": "adamw" if index % 2 else "sgd"}
        for index in range(1, 8)
    }
    observations = tuple(
        CandidateObservation(index, float(index), standard_error=0.05 * index)
        for index in range(1, 6)
    )

    proposed = BayesianSampler(candidates, mode="max").propose(
        observations,
        selected=(1, 2, 3, 4, 5),
        count=2,
    )

    assert len(proposed) == 2
    assert len(set(proposed)) == 2
    assert set(proposed).isdisjoint({1, 2, 3, 4, 5})


def test_policy_bounds_startup_and_failure_retries() -> None:
    policy = AdaptiveSearchPolicy.from_search(
        {
            "startup_trials": 12,
            "failure_retries": 2,
            "early_stopping": {"enabled": True, "min_step": 4, "confirmations": 3},
        }
    )

    assert policy.startup_trials == 12
    assert policy.failure_retries == 2
    assert policy.early_stopping_min_step == 4
    assert policy.early_stopping_confirmations == 3
    with pytest.raises(ValueError, match="failure_retries"):
        AdaptiveSearchPolicy(failure_retries=4)
    with pytest.raises(ValueError, match="confirmations"):
        AdaptiveSearchPolicy.from_search({"early_stopping": {"enabled": True, "confirmations": 0}})


def test_objective_search_enables_full_adaptive_defaults_and_exhaustive_is_explicit(
    tmp_path: Path,
) -> None:
    base = {
        "name": "default-adaptive",
        "run": "tests.work_cases.AdaptiveScoreWork",
        "seeds": [4, 7, 32, 54],
        "search": {"quality": {"values": [1.0, 2.0]}},
        "objective": {"metric": "score", "mode": "max"},
    }

    adaptive = WorkConfig.from_mapping(base, source=tmp_path / "adaptive.yaml")
    policy = adaptive.levels[0].runs[0].search_policy
    assert policy is not None
    assert policy.min_seeds == 1
    assert policy.early_stopping
    assert policy.sampler == "auto"
    # The authored trial count is consumed by default. Record-only convergence is opt-in because
    # a streak without a new maximum says nothing about mixed-space coverage.
    assert policy.convergence_patience == 0
    assert policy.confirmation_auto
    assert policy.confirmation_seeds == ()

    explicit_convergence = WorkConfig.from_mapping(
        {
            **base,
            "search": {
                "quality": {"values": [1.0, 2.0]},
                "convergence_patience": 8,
                "min_improvement": 0.01,
            },
        },
        source=tmp_path / "explicit-convergence.yaml",
    )
    explicit_policy = explicit_convergence.levels[0].runs[0].search_policy
    assert explicit_policy is not None
    assert explicit_policy.convergence_patience == 8
    assert explicit_policy.min_improvement == 0.01

    exhaustive = WorkConfig.from_mapping(
        {**base, "search": {"strategy": "exhaustive", "quality": [1.0, 2.0]}},
        source=tmp_path / "sweep.yaml",
    )
    assert exhaustive.levels[0].runs[0].search_policy is None
    assert exhaustive.planned_runs == 8


def test_empty_confirmation_seed_list_explicitly_disables_confirmation(
    tmp_path: Path,
) -> None:
    config = WorkConfig.from_mapping(
        {
            "name": "no-confirmation",
            "run": "tests.work_cases.AdaptiveScoreWork",
            "search": {
                "confirmation_seeds": [],
                "quality": {"values": [1.0, 2.0]},
            },
            "objective": {"metric": "score", "mode": "max"},
        },
        source=tmp_path / "no-confirmation.yaml",
    )

    policy = config.levels[0].runs[0].search_policy
    assert policy is not None
    assert policy.confirmation_seeds == ()


def test_exhaustive_sweep_is_exact_including_finite_conditions(tmp_path: Path) -> None:
    config = WorkConfig.from_mapping(
        {
            "name": "exact-sweep",
            "run": "tests.work_cases.ConditionalSweepWork",
            "search": {
                "strategy": "exhaustive",
                "model": {"values": ["linear", "tree"]},
                "depth": {"values": [2, 4], "when": {"model": "tree"}},
            },
        },
        source=tmp_path / "sweep.yaml",
    )

    assert [dict(value) for value in config.levels[0].runs[0].variants] == [
        {"model": "linear"},
        {"model": "tree", "depth": 2},
        {"model": "tree", "depth": 4},
    ]


def test_exhaustive_sweep_rejects_continuous_ranges_and_trial_caps(tmp_path: Path) -> None:
    base = {
        "name": "invalid-sweep",
        "run": "tests.work_cases.AdaptiveScoreWork",
    }
    with pytest.raises(ValueError, match="cannot be exhaustive"):
        WorkConfig.from_mapping(
            {
                **base,
                "search": {
                    "strategy": "exhaustive",
                    "quality": {"range": [0.0, 1.0]},
                },
            },
            source=tmp_path / "range.yaml",
        )
    with pytest.raises(ValueError, match="trials"):
        WorkConfig.from_mapping(
            {
                **base,
                "search": {
                    "strategy": "exhaustive",
                    "trials": 2,
                    "quality": {"values": [0.0, 1.0]},
                },
            },
            source=tmp_path / "trials.yaml",
        )


def test_sobol_pool_is_reproducible_mixed_and_conditional() -> None:
    space = {
        "model": {"type": "choice", "values": ["linear", "tree"]},
        "depth": {"type": "int", "low": 1, "high": 5, "when": {"model": "tree"}},
        "rate": {"type": "loguniform", "low": 1e-4, "high": 1e-2},
    }

    first = SobolSearch(space, seed=7).trials(12)
    second = SobolSearch(space, seed=7).trials(12)

    assert [trial.parameters for trial in first] == [trial.parameters for trial in second]
    assert len({trial.fingerprint for trial in first}) == 12
    assert all(
        ("depth" in trial.parameters) == (trial.parameters["model"] == "tree") for trial in first
    )


def test_seed_racer_uses_shared_evidence_and_drops_dominated_candidates() -> None:
    racer = AdaptiveSeedRacer(mode="max", probability_threshold=0.05, margin=0.02)
    outcomes = {
        1: {1: 0.90, 2: 0.91, 3: 0.89},
        2: {1: 0.89, 2: 0.90, 3: 0.88},
        3: {1: 0.20, 2: 0.19, 3: 0.21},
    }

    decisions = racer.decisions(outcomes)

    assert {decision.trial for decision in decisions} == {1, 2}
    assert all(decision.completed_seeds == 3 for decision in decisions)


def test_adaptive_specifications_do_not_duplicate_the_candidate_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planning memory must grow with Runs, not pool x Runs x pool."""
    config = WorkConfig.from_mapping(
        {
            "name": "compact-planning",
            "run": "tests.work_cases.AdaptiveScoreWork",
            "seeds": [1, 2, 3],
            "search": {
                "strategy": "adaptive",
                "trials": 4,
                "proposal_pool_size": 32,
                "quality": {"range": [0.0, 1.0]},
            },
            "objective": {"metric": "score", "mode": "max"},
            "resources": {"cpu": 1},
        },
        source=tmp_path / "compact-planning.yaml",
    )
    captured: list[Mapping[str, Any]] = []

    class PlanningCaptured(RuntimeError):
        pass

    def capture(specifications: Sequence[Mapping[str, Any]]) -> tuple[WorkResult, ...]:
        captured.extend(specifications)
        raise PlanningCaptured

    monkeypatch.setattr("lambdaforge.work.runner._execute_group", capture)

    with pytest.raises(PlanningCaptured):
        WorkRunner().run(config)

    assert len(captured) == 32 * 3
    assert len({id(specification["definition"]) for specification in captured}) == 1
    assert "variants" not in captured[0]["definition"]
    assert captured[0]["definition"]["has_variants"] is True


def test_fresh_confirmation_seeds_select_the_final_candidate(
    tmp_path: Path,
) -> None:
    config = WorkConfig.from_mapping(
        {
            "name": "confirmation-study",
            "run": "tests.work_cases.AdaptiveScoreWork",
            "seeds": [1, 2],
            "search": {
                "strategy": "adaptive",
                "trials": 2,
                "min_seeds": 1,
                "max_parallel": 2,
                "confirmation_top_k": 1,
                "confirmation_seeds": [101, 102],
                "quality": {"values": [1.0, 3.0]},
            },
            "objective": {"metric": "score", "mode": "max"},
            "resources": {"cpu": 2},
        },
        source=tmp_path / "confirmation.yaml",
    )

    result = WorkRunner().run(config)

    confirmation = [run for run in result.runs if run.study_phase == "confirmation"]
    assert {run.seed for run in confirmation} == {101, 102}
    assert {run.trial["parameters"]["quality"] for run in confirmation} == {3.0}
    assert result.summary["best"]["selection_basis"] == "fresh-confirmation-mean"
    assert result.summary["best"]["confirmation_seeds"] == [101, 102]


def test_incomplete_confirmation_is_explicit_and_cannot_select_survivors(
    tmp_path: Path,
) -> None:
    config = WorkConfig.from_mapping(
        {
            "name": "incomplete-confirmation",
            "run": "tests.work_cases.ConfirmationFailureWork",
            "seeds": [1],
            "search": {
                "strategy": "adaptive",
                "trials": 1,
                "min_seeds": 1,
                "failure_retries": 0,
                "confirmation_top_k": 1,
                "confirmation_seeds": [101, 102],
                "quality": {"values": [1.0]},
            },
            "objective": {"metric": "score", "mode": "max"},
            "resources": {"cpu": 1},
        },
        source=tmp_path / "incomplete-confirmation.yaml",
    )

    result = WorkRunner().run(config)

    assert result.status == "failed"
    assert result.summary["best"] is None
    assert result.summary["confirmation"]["status"] == "incomplete"
    assert result.summary["confirmation"]["confirmation_incomplete"] is True
    candidate = result.summary["candidates"][0]
    assert candidate["confirmation_incomplete"] is True
    assert candidate["confirmation_complete"] is False


def test_multi_fidelity_resumes_only_a_competitive_configuration(
    tmp_path: Path,
) -> None:
    config = WorkConfig.from_mapping(
        {
            "name": "fidelity-study",
            "run": "tests.work_cases.FidelityScoreWork",
            "seeds": [7],
            "search": {
                "strategy": "adaptive",
                "min_seeds": 1,
                "seed_probability_threshold": 0.25,
                "confirmation_seeds": [],
                "fidelity": {"min": 1, "max": 3, "reduction_factor": 3},
                "quality": {"values": [0.0, 2.0]},
            },
            "objective": {"metric": "score", "mode": "max"},
            "resources": {"cpu": 2},
        },
        source=tmp_path / "fidelity.yaml",
    )

    result = WorkRunner().run(config)

    assert result.status == "succeeded"
    assert len(result.runs) == 3
    continued = [run for run in result.runs if run.fidelity["target"] == 3]
    assert len(continued) == 1
    assert continued[0].resumed_from_checkpoint
    assert continued[0].trial["parameters"] == {"quality": 2.0}
    assert result.summary["best"]["parameters"] == {"quality": 2.0}
    decisions = [
        json.loads(line)
        for line in (result.execution_dir / "hpo-control" / "decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {decision["action"] for decision in decisions} >= {
        "INITIALIZE",
        "START_NEW",
        "PROMOTE_FIDELITY",
        "FINISH",
    }
    state = json.loads(
        (result.execution_dir / "hpo-control" / "state.json").read_text(encoding="utf-8")
    )
    assert state["completed_runs"] == len(result.runs)
    assert any(run["fidelity"].get("target") == 3 for run in state["runs"])


def test_adaptive_search_allocates_more_seeds_only_to_promising_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_path = tmp_path / "live-study"
    monkeypatch.setenv("LAMBDAFORGE_STUDY_PATH", str(study_path))
    monkeypatch.setenv("LAMBDAFORGE_PROGRESS_PATH", str(tmp_path / "progress.json"))
    source = tmp_path / "study.yaml"
    config = WorkConfig.from_mapping(
        {
            "name": "adaptive-study",
            "run": "tests.work_cases.AdaptiveScoreWork",
            "seeds": [1, 2, 3, 4],
            "search": {
                "strategy": "adaptive",
                "trials": 3,
                "max_parallel": 2,
                "min_seeds": 1,
                "confirmation_seeds": [],
                "quality": {"values": [1.0, 2.0, 3.0]},
            },
            "objective": {"metric": "score", "mode": "max"},
            "resources": {"cpu": 2},
        },
        source=source,
    )

    result = WorkRunner().run(config)

    assert result.status == "succeeded"
    assert len(result.runs) <= 8
    assert result.summary["planned_runs"] == 12
    assert result.summary["completed_runs"] == len(result.runs)
    assert result.summary["best"]["parameters"] == {"quality": 3.0}
    seeds_by_quality: dict[float, set[int | None]] = {}
    for run in result.runs:
        quality = float(run.trial["parameters"]["quality"])
        seeds_by_quality.setdefault(quality, set()).add(run.seed)
    # One shared repeat calibrates within-candidate seed noise.  Once that repeat shows the
    # incumbent is stable, the controller must not spend the remaining seeds mechanically.
    assert seeds_by_quality[3.0] == {1, 2}
    assert len(seeds_by_quality[1.0]) == 1
    decisions = [
        json.loads(line)
        for line in (result.execution_dir / "hpo-control" / "decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    calibration = [
        decision for decision in decisions if decision.get("action") == "CALIBRATE_SEED_NOISE"
    ]
    assert len(calibration) == 1
    repeat_actions = {
        "CALIBRATE_SEED_NOISE",
        "ADD_SHARED_SEED",
        "REPLICATE_INCUMBENT",
        "ADD_SEED",
    }
    assert all(
        float(decision["controller_value"]) > 0.0
        for decision in decisions
        if decision.get("action") in repeat_actions
    )
    telemetry = json.loads((study_path / "summary.json").read_text(encoding="utf-8"))
    assert telemetry["strategy"] == "adaptive"
    assert telemetry["counts"]["scheduled_runs"] == len(result.runs)
    assert telemetry["counts"]["completed_runs"] == len(result.runs)
    assert len(telemetry["candidates"]) == 3
    observed_runs = [run for candidate in telemetry["candidates"] for run in candidate["runs"]]
    assert all(run["log_path"] for run in observed_runs)
    controller = json.loads((study_path / "controller.json").read_text(encoding="utf-8"))
    assert controller["last"]["action"] == "FINISH"
    assert controller["last"]["reason"] == "candidate-budget-reached"


def test_default_adaptive_search_consumes_candidate_budget_on_a_flat_objective(
    tmp_path: Path,
) -> None:
    config = WorkConfig.from_mapping(
        {
            "name": "flat-adaptive-study",
            "run": "tests.work_cases.FlatAdaptiveScoreWork",
            "seeds": [1],
            "search": {
                "strategy": "adaptive",
                "trials": 10,
                "startup_trials": 1,
                "max_parallel": 2,
                "confirmation_seeds": [],
                "early_stopping": False,
                "choice": {"values": list(range(10))},
            },
            "objective": {"metric": "score", "mode": "max"},
            "resources": {"cpu": 2},
        },
        source=tmp_path / "flat.yaml",
    )

    result = WorkRunner().run(config)

    assert result.status == "succeeded"
    assert len({run.trial["index"] for run in result.runs}) == 10
    decisions = [
        json.loads(line)
        for line in (result.execution_dir / "hpo-control" / "decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert not any(decision["action"] == "STOP_PROPOSING" for decision in decisions)
    assert decisions[-1]["action"] == "FINISH"
    assert decisions[-1]["reason"] == "candidate-budget-reached"


def test_positive_convergence_patience_remains_an_explicit_early_stop(
    tmp_path: Path,
) -> None:
    config = WorkConfig.from_mapping(
        {
            "name": "explicit-convergence-study",
            "run": "tests.work_cases.FlatAdaptiveScoreWork",
            "seeds": [1],
            "search": {
                "strategy": "adaptive",
                "trials": 10,
                "startup_trials": 1,
                "max_parallel": 1,
                "confirmation_seeds": [],
                "early_stopping": False,
                "convergence_patience": 3,
                "choice": {"values": list(range(10))},
            },
            "objective": {"metric": "score", "mode": "max"},
            "resources": {"cpu": 1},
        },
        source=tmp_path / "explicit-convergence.yaml",
    )

    result = WorkRunner().run(config)
    decisions = [
        json.loads(line)
        for line in (result.execution_dir / "hpo-control" / "decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert len({run.trial["index"] for run in result.runs}) < 10
    assert any(decision["action"] == "STOP_PROPOSING" for decision in decisions)
    assert decisions[-1]["action"] == "FINISH"
    assert decisions[-1]["reason"] == "explicit-record-convergence"


def test_adaptive_controller_restores_durable_runs_without_reexecuting_them(
    tmp_path: Path,
) -> None:
    config = WorkConfig.from_mapping(
        {
            "name": "restartable-adaptive-study",
            "run": "tests.work_cases.FlatAdaptiveScoreWork",
            "seeds": [1],
            "search": {
                "strategy": "adaptive",
                "trials": 2,
                "startup_trials": 2,
                "max_parallel": 1,
                "confirmation_seeds": [],
                "early_stopping": False,
                "choice": {"values": [0, 1]},
            },
            "objective": {"metric": "score", "mode": "max"},
            "resources": {"cpu": 1},
        },
        source=tmp_path / "restartable.yaml",
    )
    first = WorkRunner().run(config)
    original_attempts = [run.attempt_id for run in first.runs]
    (first.execution_dir / "result.json").unlink()

    restored = WorkRunner().run(config)

    assert [run.attempt_id for run in restored.runs] == original_attempts
    decisions = [
        json.loads(line)
        for line in (restored.execution_dir / "hpo-control" / "decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(value["action"] == "RESTORE_CONTROLLER" for value in decisions)


def test_bayesian_provider_failure_is_audited_before_deterministic_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_provider(*args: Any, **kwargs: Any) -> tuple[int, ...]:
        del args, kwargs
        raise RuntimeError("controlled provider fit failure")

    monkeypatch.setattr("lambdaforge.work.runner.BayesianSampler.propose", fail_provider)
    config = WorkConfig.from_mapping(
        {
            "name": "provider-fallback",
            "run": "tests.work_cases.AdaptiveScoreWork",
            "seeds": [1],
            "search": {
                "strategy": "adaptive",
                "sampler": "botorch",
                "trials": 3,
                "startup_trials": 2,
                "confirmation_seeds": [],
                "quality": {"values": [1.0, 2.0, 3.0]},
            },
            "objective": {"metric": "score", "mode": "max"},
            "resources": {"cpu": 2},
        },
        source=tmp_path / "fallback.yaml",
    )

    result = WorkRunner().run(config)
    decisions = [
        json.loads(line)
        for line in (result.execution_dir / "hpo-control" / "decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    fallback = next(value for value in decisions if value["action"] == "SURROGATE_FALLBACK")
    assert fallback["error_type"] == "RuntimeError"
    assert fallback["error"] == "controlled provider fit failure"
    assert any(
        value["action"] == "PROPOSE" and value["backend"] == "mixed-knn" for value in decisions
    )
    assert result.status == "succeeded"


def test_gpu_packing_can_learn_automatically_or_keep_an_explicit_safety_floor(
    tmp_path: Path,
) -> None:
    base = {
        "name": "gpu-study",
        "run": "tests.work_cases.AdaptiveScoreWork",
        "seeds": [1, 2],
        "search": {
            "strategy": "adaptive",
            "runs_per_gpu": 4,
            "quality": {"values": [1.0, 2.0]},
        },
        "objective": {"metric": "score", "mode": "max"},
        "resources": {"gpu": 2},
    }
    automatic = WorkConfig.from_mapping(base, source=tmp_path / "study.yaml")
    automatic_policy = automatic.levels[0].runs[0].search_policy
    assert automatic_policy is not None
    assert automatic_policy.runs_per_gpu == 4
    assert automatic.resources.gpu_memory_bytes == 0

    configured = {**base, "resources": {"gpu": 2, "gpu_memory": "16GiB"}}
    config = WorkConfig.from_mapping(configured, source=tmp_path / "study.yaml")
    policy = config.levels[0].runs[0].search_policy
    assert policy is not None
    assert policy.runs_per_gpu == 4
    assert config.resources.gpu_count == 2


def test_gpu_parallelism_respects_the_hard_host_cpu_ceiling() -> None:
    resources = ResourceRequest(gpu_count=2, gpu_memory_bytes=30 * 1024**3)
    policy = AdaptiveSearchPolicy(runs_per_gpu=3)

    assert _adaptive_parallelism(resources, policy) == 1


def test_gpu_admission_uses_any_device_that_fits_and_waits_without_failing() -> None:
    gib = 1024**3
    memory = ((20 * gib, 80 * gib), (45 * gib, 80 * gib))

    slots = _admissible_gpu_slots(
        memory,
        active=(0, 0),
        last_launch=(float("-inf"), float("-inf")),
        required_bytes=30 * gib,
        runs_per_gpu=3,
        now=100.0,
        launch_stagger_seconds=5.0,
        usable=(0, 1),
    )

    assert slots == (1,)
    assert (
        _admissible_gpu_slots(
            ((20 * gib, 80 * gib), (25 * gib, 80 * gib)),
            active=(0, 1),
            last_launch=(0.0, 90.0),
            required_bytes=30 * gib,
            runs_per_gpu=3,
            now=100.0,
            launch_stagger_seconds=5.0,
            usable=(0, 1),
        )
        == ()
    )


def test_gpu_admission_staggers_same_device_launches_and_respects_packing_cap() -> None:
    gib = 1024**3
    memory = ((70 * gib, 80 * gib),)

    too_soon = _admissible_gpu_slots(
        memory,
        active=(1,),
        last_launch=(98.0,),
        required_bytes=20 * gib,
        runs_per_gpu=3,
        now=100.0,
        launch_stagger_seconds=5.0,
        usable=(0,),
    )
    ready = _admissible_gpu_slots(
        memory,
        active=(1,),
        last_launch=(90.0,),
        required_bytes=20 * gib,
        runs_per_gpu=3,
        now=100.0,
        launch_stagger_seconds=5.0,
        usable=(0,),
    )
    full = _admissible_gpu_slots(
        memory,
        active=(3,),
        last_launch=(90.0,),
        required_bytes=20 * gib,
        runs_per_gpu=3,
        now=100.0,
        launch_stagger_seconds=5.0,
        usable=(0,),
    )
    live_threshold_allows_another_run = _admissible_gpu_slots(
        memory,
        active=(2,),
        last_launch=(90.0,),
        required_bytes=30 * gib,
        runs_per_gpu=4,
        now=100.0,
        launch_stagger_seconds=5.0,
        usable=(0,),
    )

    assert too_soon == ()
    assert ready == (0,)
    assert full == ()
    assert live_threshold_allows_another_run == (0,)


def test_gpu_admission_does_not_double_count_active_run_thresholds() -> None:
    gib = 1024**3

    slots = _admissible_gpu_slots(
        ((21 * gib, 80 * gib),),
        active=(3,),
        last_launch=(0.0,),
        required_bytes=20 * gib,
        runs_per_gpu=5,
        now=100.0,
        launch_stagger_seconds=5.0,
        usable=(0,),
    )

    assert slots == (0,)


def test_cuda_oom_is_requeued_as_a_new_attempt_before_becoming_terminal(
    tmp_path: Path,
) -> None:
    result = WorkResult(
        name="study",
        work_class="tests.work_cases.AdaptiveScoreWork",
        execution_id="execution-test",
        run_id="run-test",
        attempt_id="attempt-0001",
        attempt_number=1,
        scientific_fingerprint="sha256:test",
        status="failed",
        run_dir=tmp_path,
        created_at_utc="2026-01-01T00:00:00+00:00",
        started_at_utc="2026-01-01T00:00:00+00:00",
        finished_at_utc="2026-01-01T00:00:01+00:00",
        duration_seconds=1.0,
        seed=4,
        trial={"index": 1, "parameters": {}},
        parameters={},
        inputs=(),
        requested_resources=WorkResources(1, 0, 1, 20 * 1024**3, None, 0, 1),
        failure={"type": "CUDAOutOfMemoryError", "message": "CUDA out of memory"},
        gpu_index=0,
        gpu_token="gpu-a",
        termination_type="resource_failed",
    )
    policy = AdaptiveSearchPolicy(failure_retries=1)
    specification = {
        "trial_index": 1,
        "seed": 4,
        "gpu_index": 0,
        "gpu_slot": "gpu-a",
    }

    retry = _retry_failed_result(
        specification,
        result,
        policy=policy,
        telemetry=None,
    )

    assert retry is not None
    assert retry["controller_retry"] == 1
    assert "gpu_index" not in retry
    assert retry["gpu_slot"] is None
    assert _retry_failed_result(retry, result, policy=policy, telemetry=None) is None


def test_gpu_cold_start_completes_sequential_learning_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gib = 1024**3
    attempts: dict[int, int] = {}
    events: list[str] = []
    submitted: list[Future[Any]] = []

    def outcome(trial: int, *, oom: bool = False) -> WorkResult:
        return WorkResult(
            name="study",
            work_class="tests.work_cases.AdaptiveScoreWork",
            execution_id="execution-test",
            run_id=f"run-{trial}",
            attempt_id=f"attempt-{attempts[trial]:04d}",
            attempt_number=attempts[trial],
            scientific_fingerprint=f"sha256:{trial}",
            status="failed" if oom else "succeeded",
            run_dir=tmp_path,
            created_at_utc="2026-01-01T00:00:00+00:00",
            started_at_utc="2026-01-01T00:00:00+00:00",
            finished_at_utc="2026-01-01T00:00:01+00:00",
            duration_seconds=1.0,
            seed=trial,
            trial={"index": trial, "parameters": {}},
            parameters={},
            inputs=(),
            requested_resources=WorkResources(1, 0, 1, 20 * gib, None, 0, 1),
            failure=(
                {"type": "CUDAOutOfMemoryError", "message": "CUDA out of memory"} if oom else None
            ),
            gpu_index=0,
            gpu_token="gpu-a",
            termination_type="resource_failed" if oom else "completed",
        )

    class ControlledPool:
        def __init__(self, *args: Any, initargs: tuple[str] = (), **kwargs: Any) -> None:
            del args, kwargs
            self.slot = initargs[0]

        def submit(self, function: Any, value: dict[str, Any]) -> Future[Any]:
            del function
            trial = int(value["trial_index"])
            attempts[trial] = attempts.get(trial, 0) + 1
            future: Future[Any] = Future()
            future.trial = trial  # type: ignore[attr-defined]
            future.attempt = attempts[trial]  # type: ignore[attr-defined]
            submitted.append(future)
            events.append(f"submit-{trial}-{attempts[trial]}")
            return future

        def shutdown(self, **kwargs: Any) -> None:
            del kwargs

    def controlled_wait(
        futures: Sequence[Future[Any]],
        **kwargs: Any,
    ) -> tuple[set[Future[Any]], set[Future[Any]]]:
        del kwargs
        current = set(futures)
        selected = next(iter(current))
        trial = int(selected.trial)  # type: ignore[attr-defined]
        selected.set_result(outcome(trial))
        events.append(f"finish-{trial}-1")
        return {selected}, current - {selected}

    monkeypatch.setattr(
        "lambdaforge.work.runner._gpu_memory_inventory",
        lambda count: ((70 * gib, 80 * gib),),
    )
    monkeypatch.setattr("lambdaforge.work.runner.ProcessPoolExecutor", ControlledPool)
    monkeypatch.setattr("lambdaforge.work.runner.wait", controlled_wait)
    monkeypatch.setattr("lambdaforge.work.runner._GPU_LAUNCH_STAGGER_SECONDS", 0.0)
    results: list[WorkResult] = []
    executors: list[Any] = []

    _execute_gpu_admitted_runs(
        [{"trial_index": trial, "seed": trial} for trial in (1, 2, 3)],
        resources=ResourceRequest(gpu_count=1, gpu_memory_bytes=20 * gib),
        policy=AdaptiveSearchPolicy(
            runs_per_gpu=3,
            early_stopping=False,
            failure_retries=0,
        ),
        parallelism=3,
        visible_gpus=("gpu-a",),
        objective_metric="score",
        objective_mode="max",
        telemetry=None,
        results=results,
        executors=executors,
    )

    assert attempts == {1: 1, 2: 1, 3: 1}
    assert events.index("finish-1-1") < events.index("submit-2-1")
    assert len(results) == 3
    assert all(result.ok for result in results)
    output = capsys.readouterr().out
    assert "commitment=80.0GiB" in output


def test_gpu_admission_rejects_only_a_bound_impossible_on_every_device() -> None:
    gib = 1024**3

    _validate_gpu_memory_capacity(((5 * gib, 24 * gib), (10 * gib, 40 * gib)), 30 * gib)
    with pytest.raises(ValueError, match="no Run can ever be admitted"):
        _validate_gpu_memory_capacity(((5 * gib, 24 * gib), (10 * gib, 20 * gib)), 30 * gib)


def test_gpu_dispatch_fails_explicitly_when_every_action_is_device_infeasible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deterministic impossibility terminates instead of polling forever."""
    gib = 1024**3
    monkeypatch.setattr(
        "lambdaforge.work.runner._gpu_memory_inventory",
        lambda count: ((80 * gib, 80 * gib),),
    )
    monkeypatch.setattr(
        "lambdaforge.work.runner.ResourceDemandModel.predict",
        lambda self, **kwargs: ResourcePrediction(
            candidate_key=str(kwargs["candidate_key"]),
            predicted_peak_bytes=81 * gib,
            lower_bytes=81 * gib,
            upper_bytes=81 * gib,
            known_lower_bound_bytes=81 * gib,
            predicted_time_to_envelope_seconds=None,
            predicted_duration_seconds=None,
            compatible_history_count=1,
            exact_history_count=0,
            support="censored-only",
            calibration="poor",
            backend="test",
            samples=(81 * gib,),
        ),
    )

    with pytest.raises(RuntimeError, match="instead of waiting or retrying forever"):
        _execute_gpu_admitted_runs(
            [{"trial_index": 1, "seed": 4}],
            resources=ResourceRequest(gpu_count=1),
            policy=AdaptiveSearchPolicy(runs_per_gpu=10, early_stopping=False),
            parallelism=10,
            visible_gpus=("gpu-a",),
            objective_metric="score",
            objective_mode="max",
            telemetry=None,
            results=[],
            executors=[],
        )


def test_gpu_dispatch_requests_a_bounded_alternative_when_frontier_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gib = 1024**3
    probes = itertools.chain(
        (((40 * gib, 80 * gib),),) * 3,
        itertools.repeat(((80 * gib, 80 * gib),)),
    )
    submitted: list[int] = []
    frontier_calls = 0

    def predict(self: Any, **kwargs: Any) -> ResourcePrediction:
        peak = int(kwargs["parameters"].get("memory_gib", 70)) * gib
        return ResourcePrediction(
            str(kwargs["candidate_key"]),
            peak,
            peak,
            peak,
            0,
            None,
            1.0,
            4,
            0,
            "near-compatible",
            "moderate",
            "test",
            (peak,),
        )

    class ImmediatePool:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def submit(self, function: Any, value: dict[str, Any]) -> Future[Any]:
            del function
            submitted.append(int(value["trial_index"]))
            future: Future[Any] = Future()
            future.set_result(SimpleNamespace(trial={"index": value["trial_index"]}))
            return future

        def shutdown(self, **kwargs: Any) -> None:
            del kwargs

    def expand(queued: Any, pending: Any) -> list[dict[str, Any]]:
        nonlocal frontier_calls
        del queued, pending
        frontier_calls += 1
        return [{"trial_index": 2, "seed": 2, "trial_parameters": {"memory_gib": 10}}]

    monkeypatch.setattr("lambdaforge.work.runner._gpu_memory_inventory", lambda count: next(probes))
    monkeypatch.setattr("lambdaforge.work.runner.ResourceDemandModel.predict", predict)
    monkeypatch.setattr("lambdaforge.work.runner.ProcessPoolExecutor", ImmediatePool)
    monkeypatch.setattr("lambdaforge.work.runner._GPU_LAUNCH_STAGGER_SECONDS", 0.0)

    results: list[Any] = []
    _execute_gpu_admitted_runs(
        [{"trial_index": 1, "seed": 1, "trial_parameters": {"memory_gib": 70}}],
        resources=ResourceRequest(gpu_count=1),
        policy=AdaptiveSearchPolicy(runs_per_gpu=2, early_stopping=False),
        parallelism=2,
        visible_gpus=("gpu-a",),
        objective_metric="score",
        objective_mode="max",
        telemetry=None,
        results=results,
        executors=[],
        on_resource_blocked=expand,
    )

    assert frontier_calls == 1
    assert submitted == [2, 1]
    assert len(results) == 2


def test_gpu_round_uses_resource_planner_and_releases_every_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gib = 1024**3
    samples = (
        ((10 * gib, 80 * gib), (40 * gib, 80 * gib)),
        ((10 * gib, 80 * gib), (40 * gib, 80 * gib)),
        ((35 * gib, 80 * gib), (10 * gib, 80 * gib)),
        ((35 * gib, 80 * gib), (35 * gib, 80 * gib)),
    )
    observations = itertools.chain(samples, itertools.repeat(samples[-1]))
    submitted: list[tuple[str, int]] = []
    shut_down: list[str] = []

    class ImmediatePool:
        def __init__(self, *args: Any, initargs: tuple[str] = (), **kwargs: Any) -> None:
            del args, kwargs
            self.slot = initargs[0]

        def submit(self, function: Any, value: dict[str, Any]) -> Future[Any]:
            del function
            submitted.append((self.slot, int(value["trial_index"])))
            future: Future[Any] = Future()
            future.set_result(SimpleNamespace(trial={"index": value["trial_index"]}))
            return future

        def shutdown(self, **kwargs: Any) -> None:
            del kwargs
            shut_down.append(self.slot)

    monkeypatch.setattr(
        "lambdaforge.work.runner._gpu_memory_inventory",
        lambda count: next(observations),
    )
    monkeypatch.setattr("lambdaforge.work.runner.ProcessPoolExecutor", ImmediatePool)
    monkeypatch.setattr("lambdaforge.work.runner._GPU_LAUNCH_STAGGER_SECONDS", 0.0)
    results: list[Any] = []
    executors: list[Any] = []

    _execute_gpu_admitted_runs(
        [{"trial_index": index, "seed": index} for index in (1, 2, 3)],
        resources=ResourceRequest(gpu_count=2, gpu_memory_bytes=30 * gib),
        policy=AdaptiveSearchPolicy(runs_per_gpu=2, early_stopping=False),
        parallelism=4,
        visible_gpus=("gpu-a", "gpu-b"),
        objective_metric="score",
        objective_mode="max",
        telemetry=None,
        results=results,
        executors=executors,
    )

    assert submitted[0] == ("gpu-b", 1)
    assert submitted == [("gpu-b", 1), ("gpu-a", 2), ("gpu-a", 3)]
    assert len(submitted) == len(results) == 3
    assert shut_down == [slot for slot, _trial in submitted]
    assert executors == []


def test_gpu_resource_trajectory_keeps_sampling_when_dispatch_queue_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fully occupied study must still observe peaks after its launch ramp."""
    gib = 1024**3
    future: Future[Any] = Future()
    probes = 0
    updates = 0

    class DeferredPool:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def submit(self, function: Any, value: dict[str, Any]) -> Future[Any]:
            del function, value
            return future

        def shutdown(self, **kwargs: Any) -> None:
            del kwargs

    def inventory(count: int) -> tuple[tuple[int, int], ...]:
        nonlocal probes
        assert count == 1
        probes += 1
        if probes == 3:
            future.set_result(SimpleNamespace(trial={"index": 1}))
        return ((60 * gib, 80 * gib),)

    from lambdaforge.work import runner as runner_module

    original_update = runner_module._update_active_resource_commitments

    def update(*args: Any, **kwargs: Any) -> None:
        nonlocal updates
        updates += 1
        original_update(*args, **kwargs)

    monkeypatch.setattr("lambdaforge.work.runner._gpu_memory_inventory", inventory)
    monkeypatch.setattr("lambdaforge.work.runner.ProcessPoolExecutor", DeferredPool)
    monkeypatch.setattr("lambdaforge.work.runner._GPU_LAUNCH_STAGGER_SECONDS", 0.0)
    monkeypatch.setattr("lambdaforge.work.runner._GPU_RESOURCE_SAMPLE_SECONDS", 0.0)
    monkeypatch.setattr("lambdaforge.work.runner._update_active_resource_commitments", update)

    _execute_gpu_admitted_runs(
        [{"trial_index": 1, "seed": 1}],
        resources=ResourceRequest(gpu_count=1),
        policy=AdaptiveSearchPolicy(runs_per_gpu=1, early_stopping=False),
        parallelism=1,
        visible_gpus=("gpu-a",),
        objective_metric="score",
        objective_mode="max",
        telemetry=None,
        results=[],
        executors=[],
    )

    assert probes >= 3
    assert updates >= 1


def test_gpu_dispatch_refills_an_empty_runtime_queue_before_active_run_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deferred controller actions must not leave a physically usable GPU idle for hours."""
    gib = 1024**3
    future: Future[Any] = Future()

    class FrontierRequested(RuntimeError):
        pass

    class DeferredPool:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def submit(self, function: Any, value: dict[str, Any]) -> Future[Any]:
            del function, value
            return future

        def shutdown(self, **kwargs: Any) -> None:
            del kwargs

    def refill(queued: Any, pending: Any) -> list[dict[str, Any]]:
        assert not queued
        assert len(pending) == 1
        raise FrontierRequested("resource readiness requested a scientific action")

    monkeypatch.setattr(
        "lambdaforge.work.runner._gpu_memory_inventory",
        lambda count: ((70 * gib, 80 * gib),) * count,
    )
    monkeypatch.setattr("lambdaforge.work.runner.ProcessPoolExecutor", DeferredPool)
    monkeypatch.setattr(
        "lambdaforge.work.runner.wait",
        lambda futures, **kwargs: (set(), set(futures)),
    )
    monkeypatch.setattr("lambdaforge.work.runner._GPU_LAUNCH_STAGGER_SECONDS", 0.0)
    monkeypatch.setattr("lambdaforge.work.runner._GPU_RESOURCE_SAMPLE_SECONDS", 0.0)

    with pytest.raises(FrontierRequested, match="resource readiness"):
        _execute_gpu_admitted_runs(
            [{"trial_index": 1, "seed": 1}],
            resources=ResourceRequest(gpu_count=2),
            policy=AdaptiveSearchPolicy(runs_per_gpu=1, early_stopping=False),
            parallelism=2,
            visible_gpus=("gpu-a", "gpu-b"),
            objective_metric="score",
            objective_mode="max",
            telemetry=None,
            results=[],
            executors=[],
            on_resource_blocked=refill,
        )


def test_three_gpu_dispatch_launches_baselines_then_live_packing_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the runtime modes: three baselines then a pre-completion 1→2 probe."""
    gib = 1024**3
    submitted_modes: list[tuple[int, str, int]] = []
    clock = itertools.count(0.0, 2_000.0)

    class ProbeLaunched(RuntimeError):
        pass

    class DeferredPool:
        def __init__(self, *args: Any, initargs: tuple[str, ...] = (), **kwargs: Any) -> None:
            del args, kwargs
            self.token = initargs[0]

        def submit(self, function: Any, value: dict[str, Any]) -> Future[Any]:
            del function
            submitted_modes.append(
                (
                    int(value["trial_index"]),
                    str(value["resource_admission_mode"]),
                    int(value["gpu_index"]),
                )
            )
            if len(submitted_modes) == 4:
                raise ProbeLaunched("controlled one-to-two probe launched")
            return Future()

        def shutdown(self, **kwargs: Any) -> None:
            del kwargs

    def publish_live_baselines(
        memory: Any,
        *,
        pending: Any,
        commitments: dict[Any, ActiveResourceCommitment],
        **kwargs: Any,
    ) -> tuple[Any, ...]:
        del memory, pending, kwargs
        for future, commitment in tuple(commitments.items()):
            commitments[future] = replace(
                commitment,
                current_bytes=15 * gib,
                running_peak_bytes=15 * gib,
                future_peak_samples=(15 * gib, 24 * gib, 141 * gib),
                remaining_seconds=600.0,
                resource_state="PLATEAU_UNCONFIRMED",
                checkpoint_resumable=True,
                evidence_cycles=1,
                growth_hazard=0.25,
            )
        return ()

    monkeypatch.setattr(
        "lambdaforge.work.runner._gpu_memory_inventory",
        lambda count: ((125 * gib, 141 * gib),) * count,
    )
    monkeypatch.setattr(
        "lambdaforge.work.runner._gpu_hardware_labels",
        lambda count, memory: ("H200-141",) * count,
    )
    monkeypatch.setattr("lambdaforge.work.runner.ProcessPoolExecutor", DeferredPool)
    monkeypatch.setattr(
        "lambdaforge.work.runner.wait", lambda futures, **kwargs: (set(), set(futures))
    )
    monkeypatch.setattr(
        "lambdaforge.work.runner._update_active_resource_commitments",
        publish_live_baselines,
    )
    monkeypatch.setattr("lambdaforge.work.runner.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("lambdaforge.work.runner._GPU_LAUNCH_STAGGER_SECONDS", 0.0)
    monkeypatch.setattr("lambdaforge.work.runner._GPU_RESOURCE_SAMPLE_SECONDS", 0.0)

    with pytest.raises(ProbeLaunched, match="one-to-two"):
        _execute_gpu_admitted_runs(
            [{"trial_index": index, "seed": index} for index in range(1, 7)],
            resources=ResourceRequest(gpu_count=3),
            policy=AdaptiveSearchPolicy(runs_per_gpu=50, max_parallel=150, early_stopping=False),
            parallelism=150,
            visible_gpus=("gpu-a", "gpu-b", "gpu-c"),
            objective_metric="score",
            objective_mode="max",
            telemetry=None,
            results=[],
            executors=[],
        )

    assert {gpu for _trial, _mode, gpu in submitted_modes[:3]} == {0, 1, 2}
    assert {mode for _trial, mode, _gpu in submitted_modes[:3]} == {"BASELINE_ADMISSION"}
    assert submitted_modes[3][1] == "EXPLORATORY_ADMISSION"
    assert submitted_modes[3][2] in {0, 1, 2}


def test_gpu_dispatch_can_expand_frontier_twice_without_a_terminal_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blocked first extension must not close refill until a Run terminates."""
    gib = 1024**3
    active_future: Future[Any] = Future()
    frontier_calls = 0

    class SecondFrontierRequested(RuntimeError):
        pass

    class DeferredPool:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def submit(self, function: Any, value: dict[str, Any]) -> Future[Any]:
            del function, value
            return active_future

        def shutdown(self, **kwargs: Any) -> None:
            del kwargs

    def refill(queued: Any, pending: Any) -> list[dict[str, Any]]:
        nonlocal frontier_calls
        assert len(pending) == 1
        frontier_calls += 1
        if frontier_calls == 1:
            assert not queued
            return [
                {
                    "trial_index": 2,
                    "seed": 2,
                    "trial_parameters": {"memory_gib": 100},
                }
            ]
        raise SecondFrontierRequested("requested another bounded frontier")

    def predict(self: Any, **kwargs: Any) -> ResourcePrediction:
        del self
        peak = int(kwargs["parameters"].get("memory_gib", 10)) * gib
        return ResourcePrediction(
            str(kwargs["candidate_key"]),
            peak,
            peak,
            peak,
            peak,
            None,
            100.0,
            1,
            1,
            "exact-candidate",
            "good",
            "test",
            (peak,),
        )

    monkeypatch.setattr(
        "lambdaforge.work.runner._gpu_memory_inventory",
        lambda count: ((70 * gib, 80 * gib),) * count,
    )
    monkeypatch.setattr("lambdaforge.work.runner.ResourceDemandModel.predict", predict)
    monkeypatch.setattr("lambdaforge.work.runner.ProcessPoolExecutor", DeferredPool)
    monkeypatch.setattr(
        "lambdaforge.work.runner.wait", lambda futures, **kwargs: (set(), set(futures))
    )
    monkeypatch.setattr("lambdaforge.work.runner._GPU_LAUNCH_STAGGER_SECONDS", 0.0)
    monkeypatch.setattr("lambdaforge.work.runner._GPU_RESOURCE_SAMPLE_SECONDS", 0.0)

    with pytest.raises(SecondFrontierRequested, match="another bounded frontier"):
        _execute_gpu_admitted_runs(
            [{"trial_index": 1, "seed": 1, "trial_parameters": {"memory_gib": 10}}],
            resources=ResourceRequest(gpu_count=2),
            policy=AdaptiveSearchPolicy(runs_per_gpu=2, early_stopping=False),
            parallelism=4,
            visible_gpus=("gpu-a", "gpu-b"),
            objective_metric="score",
            objective_mode="max",
            telemetry=None,
            results=[],
            executors=[],
            on_resource_blocked=refill,
        )

    assert frontier_calls == 2


def test_gpu_run_failure_still_exits_its_cuda_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gib = 1024**3
    shut_down: list[str] = []

    class FailingPool:
        def __init__(self, *args: Any, initargs: tuple[str] = (), **kwargs: Any) -> None:
            del args, kwargs
            self.slot = initargs[0]

        def submit(self, function: Any, value: dict[str, Any]) -> Future[Any]:
            del function, value
            future: Future[Any] = Future()
            future.set_exception(RuntimeError("scientific failure"))
            return future

        def shutdown(self, **kwargs: Any) -> None:
            del kwargs
            shut_down.append(self.slot)

    monkeypatch.setattr(
        "lambdaforge.work.runner._gpu_memory_inventory",
        lambda count: ((70 * gib, 80 * gib),),
    )
    monkeypatch.setattr("lambdaforge.work.runner.ProcessPoolExecutor", FailingPool)
    monkeypatch.setattr("lambdaforge.work.runner._GPU_LAUNCH_STAGGER_SECONDS", 0.0)
    executors: list[Any] = []
    results: list[WorkResult] = []

    _execute_gpu_admitted_runs(
        [
            {
                "trial_index": 1,
                "seed": 4,
                "trial_parameters": {"quality": 1.0},
                "parameters": {"quality": 1.0},
                "execution_id": "execution-test",
                "execution_dir": tmp_path,
                "definition": {
                    "name": "study",
                    "work_class": "tests.work_cases.AdaptiveScoreWork",
                    "resources": ResourceRequest(gpu_count=1, gpu_memory_bytes=20 * gib).to_dict(),
                },
            }
        ],
        resources=ResourceRequest(gpu_count=1, gpu_memory_bytes=20 * gib),
        policy=AdaptiveSearchPolicy(runs_per_gpu=3, early_stopping=False),
        parallelism=3,
        visible_gpus=("gpu-a",),
        objective_metric="score",
        objective_mode="max",
        telemetry=None,
        results=results,
        executors=executors,
    )

    assert shut_down == ["gpu-a"]
    assert executors == []
    assert len(results) == 1
    assert results[0].status == "failed"
    assert results[0].gpu_index == 0


def test_resource_frontier_bounds_waiting_actions_without_capping_active_work() -> None:
    queued = tuple({"trial_index": value} for value in range(1, 17))
    active = tuple({"trial_index": value} for value in range(17, 23))

    assert _resource_frontier_room(queued, active, parallelism=150) == 0
    assert _resource_frontier_room(queued[:2], active[:1], parallelism=4) == 1
    assert _resource_frontier_room((), active, parallelism=150) == 16


def test_killed_cpu_worker_is_retried_without_aborting_an_unrelated_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: dict[int, int] = {}
    shutdowns = 0

    class IsolatedPool:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def submit(self, function: Any, value: dict[str, Any]) -> Future[Any]:
            del function
            trial = int(value["trial_index"])
            attempts[trial] = attempts.get(trial, 0) + 1
            future: Future[Any] = Future()
            if trial == 1 and attempts[trial] == 1:
                future.set_exception(BrokenProcessPool("worker terminated abruptly"))
            else:
                future.set_result(SimpleNamespace(trial={"index": trial}))
            return future

        def shutdown(self, **kwargs: Any) -> None:
            nonlocal shutdowns
            del kwargs
            shutdowns += 1

    monkeypatch.setattr("lambdaforge.work.runner.ProcessPoolExecutor", IsolatedPool)
    results: list[Any] = []
    executors: list[Any] = []

    _execute_cpu_isolated_runs(
        [{"trial_index": 1, "seed": 1}, {"trial_index": 2, "seed": 2}],
        policy=AdaptiveSearchPolicy(
            max_parallel=2,
            early_stopping=False,
            failure_retries=1,
        ),
        parallelism=2,
        objective_metric="score",
        objective_mode="max",
        telemetry=None,
        results=results,
        executors=executors,
    )

    assert attempts == {1: 2, 2: 1}
    assert len(results) == 2
    assert shutdowns == 3
    assert executors == []


def test_cpu_dispatch_replaces_stale_queued_work_before_the_straggler_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted: list[int] = []
    second: Future[Any] = Future()

    class ControlledPool:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def submit(self, function: Any, value: dict[str, Any]) -> Future[Any]:
            del function
            trial = int(value["trial_index"])
            submitted.append(trial)
            if trial == 2:
                return second
            future: Future[Any] = Future()
            future.set_result(SimpleNamespace(trial={"index": trial}))
            return future

        def shutdown(self, **kwargs: Any) -> None:
            del kwargs

    def refill(
        result: Any,
        queued: Sequence[Mapping[str, Any]],
        pending: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        del result
        if 4 not in submitted:
            assert len(pending) == 1
            assert [int(value["trial_index"]) for value in queued] == [3]
            second.set_result(SimpleNamespace(trial={"index": 2}))
            return [{"trial_index": 4, "seed": 4}]
        return []

    monkeypatch.setattr("lambdaforge.work.runner.ProcessPoolExecutor", ControlledPool)
    results: list[Any] = []
    executors: list[Any] = []

    _execute_cpu_isolated_runs(
        [
            {"trial_index": 1, "seed": 1},
            {"trial_index": 2, "seed": 2},
            {"trial_index": 3, "seed": 3},
        ],
        policy=AdaptiveSearchPolicy(max_parallel=2, early_stopping=False),
        parallelism=2,
        objective_metric="score",
        objective_mode="max",
        telemetry=None,
        results=results,
        executors=executors,
        on_result=refill,
    )

    assert submitted == [1, 2, 4]
    assert len(results) == 3


def test_gpu_memory_probe_uses_an_ephemeral_child_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        observed.update(command=command, kwargs=kwargs)
        return SimpleNamespace(stdout="[[100, 200], [300, 400]]\n")

    monkeypatch.setattr("lambdaforge.work.runner.subprocess.run", run)

    assert _gpu_memory_inventory(2) == ((100, 200), (300, 400))
    assert observed["command"][:2] == [__import__("sys").executable, "-c"]
    assert observed["kwargs"]["timeout"] == 20


def test_transient_gpu_probe_failure_pauses_admission_without_losing_active_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gib = 1024**3
    observations: Any = iter(
        (
            ((70 * gib, 80 * gib),),
            ((70 * gib, 80 * gib),),
            RuntimeError("temporary CUDA query failure"),
            ((70 * gib, 80 * gib),),
        )
    )
    submitted: list[int] = []
    wait_calls = 0

    def inventory(_count: int) -> tuple[tuple[int, int], ...]:
        value = next(observations)
        if isinstance(value, Exception):
            raise value
        return value

    class ControlledPool:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def submit(self, function: Any, value: dict[str, Any]) -> Future[Any]:
            del function
            submitted.append(int(value["trial_index"]))
            return Future()

        def shutdown(self, **kwargs: Any) -> None:
            del kwargs

    def controlled_wait(futures: Any, **kwargs: Any) -> tuple[set[Any], set[Any]]:
        nonlocal wait_calls
        del kwargs
        wait_calls += 1
        values = set(futures)
        if wait_calls == 1:
            return set(), values
        for future in values:
            if not future.done():
                future.set_result(SimpleNamespace(trial={"index": submitted[-1]}))
        return values, set()

    monkeypatch.setattr("lambdaforge.work.runner._gpu_memory_inventory", inventory)
    monkeypatch.setattr("lambdaforge.work.runner.ProcessPoolExecutor", ControlledPool)
    monkeypatch.setattr("lambdaforge.work.runner.wait", controlled_wait)
    monkeypatch.setattr("lambdaforge.work.runner._GPU_ADMISSION_POLL_SECONDS", 0.0)
    monkeypatch.setattr("lambdaforge.work.runner._GPU_LAUNCH_STAGGER_SECONDS", 0.0)
    results: list[Any] = []
    executors: list[Any] = []

    _execute_gpu_admitted_runs(
        [{"trial_index": 1, "seed": 4}, {"trial_index": 2, "seed": 7}],
        resources=ResourceRequest(gpu_count=1, gpu_memory_bytes=20 * gib),
        policy=AdaptiveSearchPolicy(runs_per_gpu=1, early_stopping=False),
        parallelism=1,
        visible_gpus=("gpu-a",),
        objective_metric="score",
        objective_mode="max",
        telemetry=None,
        results=results,
        executors=executors,
    )

    assert submitted == [1, 2]
    assert len(results) == 2
    assert executors == []


def test_gpu_probe_error_retains_bounded_child_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    def fail(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise subprocess.CalledProcessError(
            1,
            ("python", "-c", "probe"),
            stderr="AssertionError: only 1 CUDA device(s) visible",
        )

    monkeypatch.setattr("lambdaforge.work.runner.subprocess.run", fail)

    with pytest.raises(RuntimeError, match="only 1 CUDA device"):
        _gpu_memory_inventory(2)


def test_early_stopping_requests_only_the_current_bottom_fraction(tmp_path: Path) -> None:
    specifications = []
    for index, value in enumerate((0.9, 0.8, 0.2, 0.1), 1):
        metrics = tmp_path / f"trial-{index}.jsonl"
        metrics.write_text(
            json.dumps({"name": "score", "value": value, "step": 5, "split": None}) + "\n",
            encoding="utf-8",
        )
        specifications.append(
            {"hpo_metrics_path": metrics, "hpo_stop_path": tmp_path / f"trial-{index}.stop"}
        )

    _request_early_stops(
        specifications,
        metric="score",
        mode="max",
        min_step=3,
        confirmations=1,
        probability_threshold=0.25,
        historical_results=_completed_pruner_calibration(tmp_path),
    )

    assert not Path(specifications[0]["hpo_stop_path"]).exists()
    assert Path(specifications[2]["hpo_stop_path"]).is_file()
    assert Path(specifications[3]["hpo_stop_path"]).is_file()


def test_early_stopping_waits_for_retrospective_curve_calibration(tmp_path: Path) -> None:
    specifications = []
    for trial, value in enumerate((0.95, 0.05), 1):
        metrics = tmp_path / f"uncalibrated-{trial}.jsonl"
        metrics.write_text(
            "".join(
                json.dumps({"name": "score", "value": value, "step": step}) + "\n"
                for step in range(1, 7)
            ),
            encoding="utf-8",
        )
        specifications.append(
            {
                "trial_index": trial,
                "hpo_metrics_path": metrics,
                "hpo_stop_path": tmp_path / f"uncalibrated-{trial}.stop",
            }
        )

    _request_early_stops(
        specifications,
        metric="score",
        mode="max",
        min_step=3,
        confirmations=1,
        probability_threshold=0.99,
    )

    assert not any(Path(value["hpo_stop_path"]).exists() for value in specifications)


def test_probabilistic_curve_pruning_keeps_a_slow_improving_run(tmp_path: Path) -> None:
    curves = (
        (0.10, 0.20, 0.30, 0.40, 0.50),
        (0.61, 0.63, 0.64, 0.65, 0.65),
        (0.10, 0.10, 0.10, 0.10, 0.10),
    )
    specifications = []
    for trial, curve in enumerate(curves, 1):
        metrics = tmp_path / f"curve-{trial}.jsonl"
        metrics.write_text(
            "".join(
                json.dumps({"name": "score", "value": value, "step": step, "split": None}) + "\n"
                for step, value in enumerate(curve, 1)
            ),
            encoding="utf-8",
        )
        specifications.append(
            {"hpo_metrics_path": metrics, "hpo_stop_path": tmp_path / f"curve-{trial}.stop"}
        )

    _request_early_stops(
        specifications,
        metric="score",
        mode="max",
        min_step=3,
        confirmations=1,
        probability_threshold=0.1,
        historical_results=_completed_pruner_calibration(tmp_path),
    )

    assert not Path(specifications[0]["hpo_stop_path"]).exists()
    assert Path(specifications[2]["hpo_stop_path"]).is_file()


def test_pruning_without_fidelity_does_not_invent_a_min_step_forecast_horizon(
    tmp_path: Path,
) -> None:
    """A currently leading curve must not be pruned by a long local-linear extrapolation."""
    curves = (
        (0.44, 0.45, 0.46, 0.46, 0.46),
        (0.55, 0.54, 0.53, 0.50, 0.48),
    )
    specifications = []
    for trial, curve in enumerate(curves, 1):
        metrics = tmp_path / f"forecast-{trial}.jsonl"
        metrics.write_text(
            "".join(
                json.dumps({"name": "score", "value": value, "step": step}) + "\n"
                for step, value in enumerate(curve, 40)
            ),
            encoding="utf-8",
        )
        specifications.append(
            {
                "trial_index": trial,
                "seed": 4,
                "hpo_metrics_path": metrics,
                "hpo_stop_path": tmp_path / f"forecast-{trial}.stop",
            }
        )

    _request_early_stops(
        specifications,
        metric="score",
        mode="max",
        min_step=40,
        confirmations=1,
        probability_threshold=0.02,
        margin=0.015,
    )

    assert not Path(specifications[1]["hpo_stop_path"]).exists()


def test_default_pruning_requires_two_distinct_uncompetitive_steps(tmp_path: Path) -> None:
    specifications = []
    for trial, value in enumerate((0.9, 0.1), 1):
        metrics = tmp_path / f"stable-{trial}.jsonl"
        metrics.write_text(
            "".join(
                json.dumps({"name": "score", "value": value, "step": step, "split": None}) + "\n"
                for step in range(1, 6)
            ),
            encoding="utf-8",
        )
        specifications.append(
            {"hpo_metrics_path": metrics, "hpo_stop_path": tmp_path / f"stable-{trial}.stop"}
        )

    calibration = _completed_pruner_calibration(tmp_path)
    _request_early_stops(
        specifications,
        metric="score",
        mode="max",
        min_step=3,
        historical_results=calibration,
    )

    weak_stop = Path(specifications[1]["hpo_stop_path"])
    assert not weak_stop.exists()
    for trial, value in enumerate((0.91, 0.11), 1):
        with Path(specifications[trial - 1]["hpo_metrics_path"]).open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(
                json.dumps({"name": "score", "value": value, "step": 6, "split": None}) + "\n"
            )

    _request_early_stops(
        specifications,
        metric="score",
        mode="max",
        min_step=3,
        historical_results=calibration,
    )

    assert weak_stop.is_file()
    assert "confirmations=2/2" in weak_stop.read_text(encoding="utf-8")


def test_completed_historical_candidate_can_prune_a_lone_active_straggler(
    tmp_path: Path,
) -> None:
    historical_dir = tmp_path / "historical"
    historical_dir.mkdir()
    (historical_dir / "metrics.jsonl").write_text(
        "".join(
            json.dumps({"name": "score", "value": 0.9, "step": step, "split": None}) + "\n"
            for step in range(1, 7)
        ),
        encoding="utf-8",
    )
    historical = WorkResult(
        name="work",
        work_class="tests.work_cases.AdaptiveScoreWork",
        execution_id="execution-1",
        run_id="run-historical",
        attempt_id="attempt-1",
        attempt_number=1,
        scientific_fingerprint="sha256:historical",
        status="succeeded",
        run_dir=historical_dir,
        created_at_utc="2026-01-01T00:00:00+00:00",
        started_at_utc="2026-01-01T00:00:00+00:00",
        finished_at_utc="2026-01-01T00:00:06+00:00",
        duration_seconds=6.0,
        seed=4,
        trial={"index": 1, "parameters": {"width": 128}},
        parameters={},
        inputs=(),
        requested_resources=WorkResources(1, 0, 0, 0, None, 0, 1),
        metrics={"score": 0.9},
        termination_type="completed",
    )
    active_metrics = tmp_path / "active.jsonl"
    active_metrics.write_text(
        "".join(
            json.dumps({"name": "score", "value": 0.1, "step": step, "split": None}) + "\n"
            for step in range(1, 7)
        ),
        encoding="utf-8",
    )
    prior_rung_dir = tmp_path / "prior-rung"
    prior_rung_dir.mkdir()
    (prior_rung_dir / "metrics.jsonl").write_text(
        "".join(
            json.dumps({"name": "score", "value": 0.1, "step": step, "split": None}) + "\n"
            for step in range(1, 4)
        ),
        encoding="utf-8",
    )
    prior_same_seed = replace(
        historical,
        run_id="run-prior-rung",
        run_dir=prior_rung_dir,
        trial={"index": 2, "parameters": {"width": 64}},
        metrics={"score": 0.1},
        fidelity={"current": 0, "target": 3, "maximum": 9},
    )
    calibration_peer = _completed_pruner_calibration(tmp_path)[0]
    stop = tmp_path / "active.stop"

    _request_early_stops(
        (
            {
                "trial_index": 2,
                "seed": 4,
                "hpo_metrics_path": active_metrics,
                "hpo_stop_path": stop,
            },
        ),
        metric="score",
        mode="max",
        min_step=3,
        confirmations=1,
        probability_threshold=0.25,
        historical_results=(historical, prior_same_seed, calibration_peer),
    )

    assert stop.is_file()
    evidence = json.loads((tmp_path / "active.stop.evidence.json").read_text())
    assert evidence["reference_candidate"] == 1
    assert evidence["comparison_method"] == "independent-candidate-posterior"
    assert evidence["candidate_seed_evidence"] == 1


def test_pruned_runs_are_not_promoted_by_a_partial_objective(tmp_path: Path) -> None:
    pruned = WorkResult(
        name="work",
        work_class="tests.work_cases.AdaptiveScoreWork",
        execution_id="execution-1",
        run_id="run-1",
        attempt_id="attempt-1",
        attempt_number=1,
        scientific_fingerprint="sha256:test",
        status="succeeded",
        run_dir=tmp_path,
        created_at_utc="2026-01-01T00:00:00+00:00",
        started_at_utc="2026-01-01T00:00:00+00:00",
        finished_at_utc="2026-01-01T00:00:01+00:00",
        duration_seconds=1.0,
        seed=1,
        trial={"index": 1, "parameters": {}},
        parameters={},
        inputs=(),
        requested_resources=WorkResources(1, 0, 0, 0, None, 0, 1),
        metrics={"score": 0.99},
        pruned=True,
        prune_reason="adaptive-early-stopping",
    )

    assert _candidate_score((pruned,), "score", "max", 1.0) is None
    completed = replace(
        pruned,
        pruned=False,
        prune_reason=None,
        metrics={"score": 0.6},
        objective_observation={
            "metric": "score",
            "mode": "max",
            "current": 0.6,
            "current_step": 8,
            "best": 0.9,
            "best_step": 5,
            "selection": "best-observed-checkpoint",
        },
    )
    assert _candidate_score((completed,), "score", "max", 0.0) == 0.9
    assert _candidate_score((completed, pruned), "score", "max", 0.0) is None
