"""Focused adaptive study behavior without requiring a GPU or external provider."""

from __future__ import annotations

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
    _validate_gpu_memory_capacity,
)


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
    with pytest.raises(ValueError, match="cannot repeat"):
        WorkConfig.from_mapping(
            {
                **base,
                "objective": {
                    "metric": "score",
                    "mode": "max",
                    "constraints": {"score": {"min": 0.1}},
                },
            },
            source=tmp_path / "invalid.yaml",
        )


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
    assert policy.min_seeds == 3
    assert policy.early_stopping
    assert policy.sampler == "auto"
    assert policy.convergence_patience == 8
    assert len(policy.confirmation_seeds) == 3
    assert set(policy.confirmation_seeds).isdisjoint({4, 7, 32, 54})

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
    assert seeds_by_quality[3.0] == {1, 2, 3, 4}
    assert len(seeds_by_quality[1.0]) == 1
    telemetry = json.loads((study_path / "summary.json").read_text(encoding="utf-8"))
    assert telemetry["strategy"] == "adaptive"
    assert telemetry["counts"]["scheduled_runs"] == len(result.runs)
    assert telemetry["counts"]["completed_runs"] == len(result.runs)
    assert len(telemetry["candidates"]) == 3
    observed_runs = [run for candidate in telemetry["candidates"] for run in candidate["runs"]]
    assert all(run["log_path"] for run in observed_runs)


def test_gpu_packing_is_explicit_and_requires_a_per_run_memory_bound(tmp_path: Path) -> None:
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
    try:
        WorkConfig.from_mapping(base, source=tmp_path / "study.yaml")
    except ValueError as error:
        assert "resources.gpu_memory" in str(error)
    else:
        raise AssertionError("Unsafe GPU packing was accepted without gpu_memory.")

    configured = {**base, "resources": {"gpu": 2, "gpu_memory": "16GiB"}}
    config = WorkConfig.from_mapping(configured, source=tmp_path / "study.yaml")
    policy = config.levels[0].runs[0].search_policy
    assert policy is not None
    assert policy.runs_per_gpu == 4
    assert config.resources.gpu_count == 2


def test_gpu_parallelism_is_a_maximum_not_an_immediate_free_memory_requirement() -> None:
    resources = ResourceRequest(gpu_count=2, gpu_memory_bytes=30 * 1024**3)
    policy = AdaptiveSearchPolicy(runs_per_gpu=3)

    assert _adaptive_parallelism(resources, policy) == 6


def test_gpu_admission_uses_any_device_that_fits_and_waits_without_failing() -> None:
    gib = 1024**3
    memory = ((20 * gib, 80 * gib), (45 * gib, 80 * gib))

    slots = _admissible_gpu_slots(
        memory,
        active=(0, 0),
        last_launch=(float("-inf"), float("-inf")),
        admission_budget=(20 * gib, 45 * gib),
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
            admission_budget=(20 * gib, 45 * gib),
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
        admission_budget=(70 * gib,),
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
        admission_budget=(70 * gib,),
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
        admission_budget=(70 * gib,),
        required_bytes=20 * gib,
        runs_per_gpu=3,
        now=100.0,
        launch_stagger_seconds=5.0,
        usable=(0,),
    )
    budget_exhausted = _admissible_gpu_slots(
        memory,
        active=(2,),
        last_launch=(90.0,),
        admission_budget=(70 * gib,),
        required_bytes=30 * gib,
        runs_per_gpu=4,
        now=100.0,
        launch_stagger_seconds=5.0,
        usable=(0,),
    )

    assert too_soon == ()
    assert ready == (0,)
    assert full == ()
    assert budget_exhausted == ()


def test_gpu_admission_rejects_only_a_bound_impossible_on_every_device() -> None:
    gib = 1024**3

    _validate_gpu_memory_capacity(((5 * gib, 24 * gib), (10 * gib, 40 * gib)), 30 * gib)
    with pytest.raises(ValueError, match="no Run can ever be admitted"):
        _validate_gpu_memory_capacity(((5 * gib, 24 * gib), (10 * gib, 20 * gib)), 30 * gib)


def test_gpu_round_dispatches_to_available_devices_and_retries_queued_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gib = 1024**3
    observations = iter(
        (
            ((10 * gib, 80 * gib), (40 * gib, 80 * gib)),
            ((10 * gib, 80 * gib), (40 * gib, 80 * gib)),
            ((35 * gib, 80 * gib), (10 * gib, 80 * gib)),
            ((35 * gib, 80 * gib), (35 * gib, 80 * gib)),
        )
    )
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
    assert ("gpu-a", 2) in submitted
    assert len(submitted) == len(results) == 3
    assert shut_down == [slot for slot, _trial in submitted]
    assert executors == []


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


def test_cpu_dispatch_refills_a_free_slot_before_the_straggler_finishes(
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
        result: Any, queued: int, pending: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        del result, queued
        if 3 not in submitted:
            assert len(pending) == 1
            second.set_result(SimpleNamespace(trial={"index": 2}))
            return [{"trial_index": 3, "seed": 3}]
        return []

    monkeypatch.setattr("lambdaforge.work.runner.ProcessPoolExecutor", ControlledPool)
    results: list[Any] = []
    executors: list[Any] = []

    _execute_cpu_isolated_runs(
        [{"trial_index": 1, "seed": 1}, {"trial_index": 2, "seed": 2}],
        policy=AdaptiveSearchPolicy(max_parallel=2, early_stopping=False),
        parallelism=2,
        objective_metric="score",
        objective_mode="max",
        telemetry=None,
        results=results,
        executors=executors,
        on_result=refill,
    )

    assert submitted == [1, 2, 3]
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
    )

    assert not Path(specifications[0]["hpo_stop_path"]).exists()
    assert Path(specifications[2]["hpo_stop_path"]).is_file()
    assert Path(specifications[3]["hpo_stop_path"]).is_file()


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
    )

    assert not Path(specifications[0]["hpo_stop_path"]).exists()
    assert Path(specifications[2]["hpo_stop_path"]).is_file()


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

    _request_early_stops(specifications, metric="score", mode="max", min_step=3)

    weak_stop = Path(specifications[1]["hpo_stop_path"])
    assert not weak_stop.exists()
    for trial, value in enumerate((0.91, 0.11), 1):
        with Path(specifications[trial - 1]["hpo_metrics_path"]).open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(
                json.dumps({"name": "score", "value": value, "step": 6, "split": None}) + "\n"
            )

    _request_early_stops(specifications, metric="score", mode="max", min_step=3)

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
        historical_results=(historical, prior_same_seed),
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
