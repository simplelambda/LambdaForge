"""Long-study regressions across telemetry, scientific allocation and resource history."""

from __future__ import annotations

import importlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from lambdaforge.hpo.AdaptiveResources import (
    ActiveResourceCommitment,
    AdmissionDecision,
    BoundedResourceTrajectory,
    ResourceDemandModel,
    ResourceHistoryStore,
)
from lambdaforge.hpo.AdaptiveStatistics import AdaptiveSeedRacer
from lambdaforge.hpo.ParameterSpace import ParameterSpace
from lambdaforge.hpo.ScientificDesign import ScientificQuestionAnalyzer, SeedNoiseModel
from lambdaforge.study_projection import interactive_study
from lambdaforge.work.models import atomic_json
from lambdaforge.work.study import StudyTelemetry
from tests.hpo.test_adaptive_resources import GIB, gpu, observation, prediction


def test_durable_semantics_survive_thousands_of_decisions_and_reload(tmp_path: Path) -> None:
    telemetry = StudyTelemetry(tmp_path / "study")
    objective = {"metric": "score", "mode": "max", "practical_margin": 0.015}
    space = {"learning_rate": {"range": [0.00001, 0.1], "scale": "log"}}
    telemetry.initialize(
        name="long",
        execution_id="execution-long",
        strategy="adaptive",
        objective=objective,
        specifications=(),
    )
    initialization = {
        "action": "INITIALIZE",
        "objective": objective,
        "policy": {"practical_equivalence_margin": 0.015, "parameter_space": space},
        "seed_stream_metadata": {"role": "replicate"},
    }
    telemetry.controller_decision(initialization)
    for index in range(1, 1051):
        telemetry.controller_decision({"action": "WAIT", "decision": index})
    reloaded = StudyTelemetry(tmp_path / "study")
    reloaded.refresh()
    source = json.loads((tmp_path / "study/summary.json").read_text())
    assert all(value["action"] != "INITIALIZE" for value in source["controller"]["recent"])
    assert len(source["controller"]["recent"]) == 25
    assert source["controller"]["initialization"] == initialization
    scientific = source["hpo_analysis"]["scientific_understanding"]
    assert scientific["practical_margin"] == 0.015
    projected = interactive_study(source)
    from lambdaforge.analysis.StudyAnalysis import StudyAnalysis

    assert StudyAnalysis._search_policy(projected)["parameter_space"] == space
    assert ParameterSpace.from_schema(space).descriptor("learning_rate").scale == "log"


def test_legacy_semantics_are_recovered_once_from_durable_history(tmp_path: Path) -> None:
    telemetry = StudyTelemetry(tmp_path / "study")
    telemetry.initialize(
        name="old",
        execution_id="e",
        strategy="adaptive",
        objective={"metric": "score", "mode": "max"},
        specifications=(),
    )
    initialization = {
        "action": "INITIALIZE",
        "policy": {
            "practical_equivalence_margin": 0.015,
            "parameter_space": {"lr": {"range": [1e-5, 0.1], "scale": "log"}},
        },
    }
    (tmp_path / "study/controller-history.jsonl").write_text(json.dumps(initialization) + "\n")
    atomic_json(
        tmp_path / "study/controller.json",
        {
            "controller_telemetry_version": 1,
            "history_count": 1050,
            "recent": [{"action": "WAIT"}],
        },
    )
    telemetry.refresh()
    assert (
        json.loads((tmp_path / "study/controller.json").read_text())["initialization"]
        == initialization
    )
    (tmp_path / "study/controller-history.jsonl").unlink()
    telemetry.refresh()
    summary = json.loads((tmp_path / "study/summary.json").read_text())
    assert summary["hpo_analysis"]["scientific_understanding"]["practical_margin"] == 0.015


def test_required_scientific_configuration_fails_explicitly_if_lost(tmp_path: Path) -> None:
    from lambdaforge.analysis.StudyAnalysis import StudyAnalysis

    source = interactive_study(
        {"controller": {"scientific_configuration_required": True, "recent": []}}
    )
    with pytest.raises(ValueError, match="scientific"):
        StudyAnalysis._search_policy(source)


def test_summary_reuses_the_controller_scientific_snapshot(tmp_path: Path) -> None:
    objective = {"metric": "score", "mode": "max"}
    space = {"lr": {"range": [1e-5, 0.1], "scale": "log"}}
    candidates = [
        {
            "trial": trial,
            "parameters": {"lr": lr},
            "runs": [
                {"seed": seed, "state": "succeeded", "final_objective": 0.6 + seed / 10000}
                for seed in range(3)
            ],
        }
        for trial, lr in enumerate((1e-5, 1e-3, 0.1), 1)
    ]
    scientific = ScientificQuestionAnalyzer.analyze(
        candidates,
        objective,
        practical_margin=0.015,
        fingerprint="e",
        parameter_space=space,
    )
    state_path = tmp_path / "job/work/execution/hpo-control/state.json"
    atomic_json(state_path, {"scientific_understanding": scientific})
    telemetry = StudyTelemetry(tmp_path / "study")
    telemetry.initialize(
        name="same", execution_id="e", strategy="adaptive", objective=objective, specifications=()
    )
    telemetry.controller_decision(
        {
            "action": "INITIALIZE",
            "control_state_path": str(state_path),
            "policy": {
                "parameter_space": space,
                "practical_equivalence_margin": 0.015,
            },
        }
    )
    for index in range(30):
        telemetry.controller_decision({"action": "WAIT", "decision": index})
    telemetry.refresh()
    source = json.loads((tmp_path / "study/summary.json").read_text())
    assert source["hpo_analysis"]["scientific_understanding"] == scientific


def test_one_repeat_identifies_noise_but_retains_wide_variance_uncertainty() -> None:
    noise = SeedNoiseModel.fit({1: {11: 0.65, 22: 0.6512}, 2: {11: 0.64}})
    diagnostic = noise.to_dict()
    assert diagnostic["status"] == "provisional"
    assert diagnostic["residual_degrees_of_freedom"] == 1
    assert not noise.calibrated
    assert diagnostic["sigma_range"][1] > 10 * diagnostic["sigma_range"][0]
    decisions = AdaptiveSeedRacer(mode="max", margin=0.015).decisions(
        {1: {11: 0.65, 22: 0.6512}, 2: {11: 0.64}},
        eligible=[1, 2],
        scientific={
            "unresolved_questions": [{"remaining_information_value": 0.9} for _ in range(40)]
        },
    )
    assert decisions and max(item.global_calibration_value for item in decisions) > 0.9
    assert any(item.purpose == "CALIBRATE_SEED_NOISE" for item in decisions)


def test_distributed_shared_seed_evidence_shrinks_noise_uncertainty_and_value() -> None:
    sparse = {1: {11: 0.65, 22: 0.6512}, 2: {11: 0.64}}
    rich = {
        trial: {seed: 0.6 + trial / 100 + (-1) ** seed * 0.001 for seed in range(24)}
        for trial in range(1, 7)
    }
    weak, strong = SeedNoiseModel.fit(sparse), SeedNoiseModel.fit(rich)
    assert strong.calibrated
    assert strong.relative_uncertainty < weak.relative_uncertainty
    low, high = strong.to_dict()["sigma_range"]
    assert high / low < weak.to_dict()["sigma_range"][1] / weak.to_dict()["sigma_range"][0]
    racer = AdaptiveSeedRacer(mode="max", margin=0.015)
    scientific = {"unresolved_questions": [{"remaining_information_value": 0.9} for _ in range(40)]}
    weak_value = max(
        item.global_calibration_value for item in racer.decisions(sparse, scientific=scientific)
    )
    strong_value = max(
        item.global_calibration_value for item in racer.decisions(rich, scientific=scientific)
    )
    assert strong_value < weak_value
    # With equal incremental cost the same common planner currency ranks a shared seed above
    # a weak interaction first, then a material interaction above further noise calibration.
    assert weak_value > 0.5 > strong_value


def test_noisy_context_is_not_hidden_by_the_pooled_predictive_fallback() -> None:
    outcomes = {
        trial: {seed: 0.5 + (-1) ** seed * (0.2 if trial == 1 else 0.001) for seed in range(24)}
        for trial in range(1, 6)
    }
    noise = SeedNoiseModel.fit(outcomes)
    assert noise.to_dict()["heterogeneity_status"] == "possible"
    assert noise.predictive_variance >= max(noise.context_variances)


def test_terminal_sampled_envelopes_end_whole_gpu_cold_start(tmp_path: Path) -> None:
    store = ResourceHistoryStore(tmp_path)
    for index in range(24):
        store.append(
            replace(
                observation(f"small-{index}", (10 + index % 3) * GIB),
                total_bytes=140 * GIB,
                hardware="large-gpu",
                peak_is_exact=False,
                measurement_quality="sampled-physical",
                sampled_complete=True,
                sampled_peak_upper_bytes=(12 + index % 3) * GIB,
            )
        )
    model = ResourceDemandModel(ResourceHistoryStore(tmp_path).load())
    prediction = model.predict(
        candidate_key="small-1",
        compatibility_key="compatible",
        parameters={},
        hardware="large-gpu",
        total_bytes=140 * GIB,
    )
    assert prediction.exact_history_count == 0
    assert prediction.sampled_history_count > 0
    assert prediction.upper_bytes < 30 * GIB
    incomplete = replace(store.load()[0], sampled_complete=False, state="failed")
    censored = ResourceDemandModel((incomplete,)).predict(
        candidate_key=incomplete.candidate_key,
        compatibility_key="compatible",
        parameters={},
        hardware="large-gpu",
        total_bytes=140 * GIB,
    )
    assert censored.sampled_history_count == 0
    assert censored.upper_bytes >= 140 * GIB


def test_throughput_learning_distinguishes_safe_memory_from_useful_packing() -> None:
    model = ResourceDemandModel(
        tuple(
            observation(str(concurrency), 10 * GIB, co_runners=concurrency - 1, throughput=rate)
            for concurrency, rate in ((1, 10), (2, 9), (3, 5))
        )
    )
    assert model.aggregate_throughput("compatible", 1) == 10
    assert model.aggregate_throughput("compatible", 2) == 18
    assert model.aggregate_throughput("compatible", 3) == 15
    from lambdaforge.hpo.AdaptiveResources import CandidateResourceAction, GPUPlacementPlanner

    action = CandidateResourceAction(
        "new",
        {"trial_index": 4, "resource_compatibility_key": "compatible"},
        0.8,
        prediction("new", 10 * GIB),
    )
    one = (ActiveResourceCommitment("one", 10 * GIB),)
    admitted, _blocked = GPUPlacementPlanner(model).place(
        (action,),
        (gpu(free=60, active=one),),
        max_launches=1,
    )
    assert admitted
    two = (*one, ActiveResourceCommitment("two", 10 * GIB))
    admitted, blocked = GPUPlacementPlanner(model).place(
        (action,),
        (gpu(free=60, active=two),),
        max_launches=1,
    )
    assert not admitted
    assert blocked[0].devices[0]["reason"] == "predicted-aggregate-throughput-would-not-improve"


def test_repeated_blocks_coalesce_but_preserve_counts_across_restart(tmp_path: Path) -> None:
    decision = AdmissionDecision(
        "candidate",
        1,
        "RESOURCE_BLOCKED",
        None,
        "envelope does not fit",
        0.2,
        {"upper_bytes": 30 * GIB},
        (),
        0.5,
        100,
    )
    store = ResourceHistoryStore(tmp_path)
    for _ in range(100):
        store.record_decisions((decision,))
    assert len((tmp_path / "admission-decisions.jsonl").read_text().splitlines()) == 1
    restarted = ResourceHistoryStore(tmp_path)
    restarted.record_decisions((replace(decision, prediction={"upper_bytes": 20 * GIB}),))
    summary = json.loads((tmp_path / "blocked-summary.json").read_text())["candidate"]
    assert summary["blocked_count"] == 101
    assert summary["representative"]["prediction"]["upper_bytes"] == 20 * GIB
    assert summary["first_seen"] <= summary["last_seen"]


def test_live_nvml_peak_survives_missing_exit_sample_and_terminal_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = importlib.import_module("lambdaforge.work.runner")
    from lambdaforge.work.models import WorkResources, WorkResult

    future = object()
    specification = {"trial_parameters": {"width": 32}, "hpo_dispatched_monotonic": 0.0}
    metadata = {
        future: {
            "candidate_key": "one",
            "compatibility_key": "compatible",
            "hardware": "large",
            "total_bytes": 140 * GIB,
        }
    }
    trajectories: dict[object, BoundedResourceTrajectory] = {}
    commitments = {future: ActiveResourceCommitment("one", 140 * GIB)}
    monkeypatch.setattr(
        runner,
        "_read_live_resource_snapshot",
        lambda _: {
            "step": 2,
            "cuda_allocated_bytes": 9 * GIB,
            "cuda_max_allocated_bytes": 10 * GIB,
            "cuda_reserved_bytes": 10 * GIB,
            "phase": "training",
        },
    )
    for now, physical in ((10.0, {future: 11 * GIB}), (20.0, {future: 12 * GIB}), (30.0, {})):
        monkeypatch.setattr(
            runner, "_gpu_process_memory", lambda *_, value=physical: (value, "nvml-process-exact")
        )
        runner._update_active_resource_commitments(
            ((128 * GIB, 140 * GIB),),
            pending={future: (specification, 0, None)},
            commitments=commitments,
            trajectories=trajectories,
            metadata=metadata,
            model=ResourceDemandModel(),
            now=now,
        )
    assert metadata[future]["measurement_provenance"] != "nvml-process-exact"
    metadata[future]["trajectory"] = trajectories[future]
    result = WorkResult(
        name="small",
        work_class="project.Train",
        execution_id="e",
        run_id="r",
        attempt_id="a",
        attempt_number=1,
        scientific_fingerprint="sha256:e",
        status="succeeded",
        run_dir=tmp_path,
        created_at_utc="2026-01-01T00:00:00+00:00",
        started_at_utc="2026-01-01T00:00:00+00:00",
        finished_at_utc="2026-01-01T00:00:30+00:00",
        duration_seconds=30,
        seed=1,
        trial={"index": 1},
        parameters={},
        inputs=(),
        requested_resources=WorkResources(1, 0, 1, 0, None, 0, 1),
    )
    observed = runner._resource_observation_for_result(
        specification,
        result,
        metadata=metadata[future],
        memory_failure=False,
    )
    assert observed.physical_peak_bytes == 12 * GIB
    assert observed.allocated_peak_bytes == 10 * GIB
    assert observed.sampled_complete and not observed.peak_is_exact
    store = ResourceHistoryStore(tmp_path / "resources")
    store.append(observed)
    prediction_after = ResourceDemandModel(store.load()).predict(
        candidate_key="one",
        compatibility_key="compatible",
        parameters={"width": 32},
        hardware="large",
        total_bytes=140 * GIB,
    )
    assert prediction_after.sampled_history_count == 1
    assert prediction_after.upper_bytes < 30 * GIB
    interrupted = runner._resource_observation_for_result(
        specification,
        replace(result, status="failed"),
        metadata=metadata[future],
        memory_failure=False,
    )
    assert not interrupted.sampled_complete
    assert interrupted.sampled_peak_upper_bytes is None


def test_generic_metric_and_progress_throughput_are_available_without_named_training_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lambdaforge.work.runtime import MetricCollection, ProgressReporter

    clock = [0.0]
    monkeypatch.setattr("lambdaforge.work.runtime.time.monotonic", lambda: clock[0])
    metrics = MetricCollection(tmp_path)
    progress = ProgressReporter(tmp_path)
    clock[0] = 10.0
    metrics.log("score", 0.6, step=1)
    progress.update(completed=20, total=100)
    assert json.loads((tmp_path / "metric-progress.json").read_text())["throughput"] == 0.1
    assert json.loads((tmp_path / "progress.json").read_text())["throughput"] == 2.0


def test_live_and_report_analysis_keep_semantics_and_expose_precision() -> None:
    candidates = [
        {
            "trial": trial,
            "parameters": {"lr": 10 ** (-5 + trial)},
            "runs": [
                {"seed": seed, "state": "succeeded", "final_objective": 0.6 + seed / 10000}
                for seed in range(5)
            ],
        }
        for trial in range(1, 8)
    ]
    kwargs = {
        "practical_margin": 0.015,
        "fingerprint": "precision",
        "parameter_space": {"lr": {"range": [1e-5, 100], "scale": "log"}},
    }
    live = ScientificQuestionAnalyzer.analyze(
        candidates, {"metric": "score", "mode": "max"}, **kwargs
    )
    report = ScientificQuestionAnalyzer.analyze(
        candidates, {"metric": "score", "mode": "max"}, final=True, **kwargs
    )
    assert live["evidence"]["resamples"] == 32
    assert report["evidence"]["resamples"] > 32
    assert live["practical_margin"] == report["practical_margin"] == 0.015
    assert (
        live["parameter_questions"][0]["conclusion_kind"]
        == report["parameter_questions"][0]["conclusion_kind"]
    )
    assert "agreement" in report["evidence"]["confidence_semantics"]
