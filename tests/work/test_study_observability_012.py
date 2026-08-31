"""Compact, isolated observability for concurrent scientific study Runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from lambdaforge.cli.LiveJobMonitor import (
    StudyEpochRenderer,
    StudyInsightRenderer,
    StudyParameterInsightRenderer,
    StudyRenderer,
    StudyRunRenderer,
)
from lambdaforge.controlplane import ClusterCatalog, ClusterProfile, JobService, JobState, JobStore
from lambdaforge.controlplane.jobs import JobRecord
from lambdaforge.hpo.StudyInsights import StudyInsightAnalyzer
from lambdaforge.training.callbacks.AdaptiveHpoCallback import AdaptiveHpoCallback
from lambdaforge.work.study import StudyTelemetry


def _specification(trial: int = 1, seed: int = 4) -> dict[str, object]:
    return {
        "trial_index": trial,
        "seed": seed,
        "trial_parameters": {"width": 128, "dropout": 0.2},
    }


def test_study_telemetry_folds_live_metrics_without_copying_run_evidence(
    tmp_path: Path,
) -> None:
    study = StudyTelemetry(tmp_path / "study", progress_path=tmp_path / "progress.json")
    specification = _specification()
    run_dir = tmp_path / "job" / "work" / "runs" / "run-1" / "attempts" / "attempt-1"
    run_dir.mkdir(parents=True)
    metrics = run_dir / "metrics.jsonl"
    training = run_dir / "training-metrics.jsonl"
    training.write_text(
        json.dumps({"kind": "chart-filter", "include": ["val_*"]})
        + "\n"
        + "\n".join(
            json.dumps({"name": "val_loss", "value": value, "step": step, "split": None})
            for step, value in ((1, 0.8), (2, 0.5))
        )
        + "\n",
        encoding="utf-8",
    )

    study.initialize(
        name="training",
        execution_id="execution-1",
        strategy="adaptive",
        objective={"metric": "val_loss", "mode": "min"},
        specifications=(specification,),
    )
    study.schedule((specification,))
    study.run_started(
        specification,
        run_dir=run_dir,
        metrics_path=metrics,
        training_metrics_path=training,
    )
    snapshot = study.refresh()

    run = snapshot["candidates"][0]["runs"][0]
    assert run["key"] == "trial-00001-seed-4"
    assert run["state"] == "running"
    assert run["latest_step"] == 2
    assert run["latest_metrics"]["val_loss"] == 0.5
    assert run["best_step"] == 2
    assert run["best_objective"] == 0.5
    candidate = snapshot["candidates"][0]
    assert candidate["current_objective"] == 0.5
    assert candidate["best_objective"] == 0.5
    assert candidate["best_seed"] == 4
    assert candidate["best_is_provisional"] is True
    assert snapshot["counts"]["active_runs"] == 1
    assert json.loads((tmp_path / "progress.json").read_text())["completed"] == 0
    assert training.read_text(encoding="utf-8").count("val_loss") == 2


def test_study_telemetry_preserves_best_epoch_outside_the_bounded_tail(
    tmp_path: Path,
) -> None:
    study = StudyTelemetry(tmp_path / "study")
    specification = _specification()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metrics = run_dir / "metrics.jsonl"
    training = run_dir / "training-metrics.jsonl"
    metrics.touch()
    training.write_text(
        json.dumps({"name": "score", "value": 0.9, "step": 1, "split": None}) + "\n",
        encoding="utf-8",
    )
    study.initialize(
        name="training",
        execution_id="execution-1",
        strategy="adaptive",
        objective={"metric": "score", "mode": "max"},
        specifications=(specification,),
    )
    study.schedule((specification,))
    study.run_started(
        specification,
        run_dir=run_dir,
        metrics_path=metrics,
        training_metrics_path=training,
    )
    assert study.refresh()["candidates"][0]["runs"][0]["best_step"] == 1

    with training.open("a", encoding="utf-8") as stream:
        for step in range(2, 7002):
            stream.write(
                json.dumps({"name": "score", "value": 0.1, "step": step, "split": None}) + "\n"
            )

    run = study.refresh()["candidates"][0]["runs"][0]
    assert run["latest_step"] == 7001
    assert run["best_step"] == 1
    assert run["best_objective"] == 0.9


def test_terminal_objective_keeps_current_distinct_from_best_despite_duplicate_streams(
    tmp_path: Path,
) -> None:
    study = StudyTelemetry(tmp_path / "study")
    specification = _specification()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    training = run_dir / "training-metrics.jsonl"
    training.write_text(
        "".join(
            json.dumps({"name": "score", "value": value, "step": step, "split": None}) + "\n"
            for step, value in ((3, 0.6), (2, 0.9))
        ),
        encoding="utf-8",
    )
    study.initialize(
        name="training",
        execution_id="execution-1",
        strategy="adaptive",
        objective={"metric": "score", "mode": "max"},
        specifications=(),
    )
    study.schedule((specification,))
    study._write_run(
        "trial-00001-seed-4",
        {
            "trial": 1,
            "seed": 4,
            "state": "succeeded",
            "metrics": {"score": 0.6},
            "metrics_path": str(run_dir / "metrics.jsonl"),
            "training_metrics_path": str(training),
            "objective_observation": {
                "metric": "score",
                "mode": "max",
                "current": 0.6,
                "current_step": 3,
                "best": 0.9,
                "best_step": 2,
            },
        },
    )

    run = study.refresh()["candidates"][0]["runs"][0]

    assert run["latest_metrics"]["score"] == 0.6
    assert run["latest_step"] == 3
    assert run["best_objective"] == 0.9
    assert run["best_step"] == 2


def test_adaptive_study_publishes_only_candidates_that_were_actually_proposed(
    tmp_path: Path,
) -> None:
    study = StudyTelemetry(tmp_path / "study")
    first = _specification()
    second = {**first, "trial_index": 2, "trial_parameters": {"width": 256}}
    study.initialize(
        name="training",
        execution_id="execution-1",
        strategy="adaptive",
        objective={"metric": "score", "mode": "max"},
        specifications=(),
        planned_runs=100,
        planned_candidates=50,
    )

    empty = study.refresh()
    assert empty["planned_candidates"] == 50
    assert empty["counts"]["candidates"] == 0

    study.schedule((first,))
    proposed = study.refresh()
    assert proposed["counts"]["candidates"] == 1
    assert [candidate["trial"] for candidate in proposed["candidates"]] == [1]

    study.schedule((second,))
    assert [candidate["trial"] for candidate in study.refresh()["candidates"]] == [1, 2]


def test_pruned_only_candidate_is_terminal_censored_evidence_not_failure(
    tmp_path: Path,
) -> None:
    study = StudyTelemetry(tmp_path / "study")
    specification = _specification()
    study.initialize(
        name="training",
        execution_id="execution-1",
        strategy="adaptive",
        objective={"metric": "score", "mode": "max"},
        specifications=(),
        planned_candidates=2,
    )
    study.schedule((specification,))
    study._write_run(
        "trial-00001-seed-4",
        {
            "trial": 1,
            "seed": 4,
            "state": "pruned",
            "metrics": {"score": 0.1},
            "prune_reason": "p_competitive=0.01",
        },
    )
    study.candidates_observed((1,))

    snapshot = study.refresh()

    assert snapshot["candidates"][0]["state"] == "pruned"
    assert snapshot["hpo_analysis"]["candidate_observations"] == 0
    assert snapshot["hpo_analysis"]["censored_pruned_candidates"] == 1
    assert snapshot["counts"]["pruned_runs"] == 1
    signals = {
        value["parameter"]: value["pruning_signal"]
        for value in snapshot["hpo_analysis"]["parameters"]
    }
    assert signals["width"]["pruned"] == 1
    assert signals["width"]["pruned_rate"] == 1.0


def test_candidate_that_violates_an_objective_guardrail_is_not_selectable(
    tmp_path: Path,
) -> None:
    study = StudyTelemetry(tmp_path / "study")
    specification = _specification()
    study.initialize(
        name="training",
        execution_id="execution-1",
        strategy="adaptive",
        objective={
            "metric": "auprc",
            "mode": "max",
            "constraints": {"accuracy": {"min": 0.6}},
        },
        specifications=(),
    )
    study.schedule((specification,))
    study._write_run(
        "trial-00001-seed-4",
        {
            "trial": 1,
            "seed": 4,
            "state": "succeeded",
            "best_step": 3,
            "best_objective": 0.8,
            "metrics": {"auprc": 0.8, "accuracy": 0.5},
            "objective_observation": {
                "metric": "auprc",
                "mode": "max",
                "current": 0.8,
                "current_step": 3,
                "best": 0.8,
                "best_step": 3,
                "constraints": {
                    "accuracy": {
                        "value": 0.5,
                        "step": 3,
                        "min": 0.6,
                        "max": None,
                        "satisfied": False,
                    }
                },
            },
        },
    )

    snapshot = study.refresh()
    candidate = snapshot["candidates"][0]

    assert candidate["state"] == "infeasible"
    assert candidate["feasibility"]["feasible"] is False
    assert "selection_objective" not in candidate
    assert snapshot["hpo_analysis"]["infeasible_candidates"] == 1


def test_hpo_insights_explain_numeric_direction_without_claiming_causality() -> None:
    candidates = [
        {
            "trial": index,
            "parameters": {"learning_rate": float(index)},
            "latest_metrics": {"score": float(index) + (0.05 if index % 2 else 0.0)},
            "selection_objective": float(index) + (0.05 if index % 2 else 0.0),
            "runs": [{"state": "succeeded"}],
        }
        for index in range(1, 13)
    ]

    analysis = StudyInsightAnalyzer.analyze(candidates, {"metric": "score", "mode": "max"})

    parameter = analysis["parameters"][0]
    assert analysis["status"] == "actionable"
    assert parameter["parameter"] == "learning_rate"
    assert parameter["rank_correlation"] > 0.9
    assert parameter["confidence_label"] == "high"
    assert "Higher values" in parameter["conclusion"]
    assert "causal" in analysis["caveat"]
    assert analysis["next_question"]["parameter"] == "learning_rate"


def test_hpo_insights_compare_categories_and_keep_conditional_inactivity_visible() -> None:
    candidates = [
        {
            "parameters": {
                "optimizer": optimizer,
                **({"momentum": 0.9} if optimizer == "sgd" else {}),
            },
            "latest_metrics": {"loss": loss},
            "selection_objective": loss,
            "runs": [{"state": "succeeded"}],
        }
        for optimizer, loss in (
            ("adamw", 0.20),
            ("adamw", 0.22),
            ("adamw", 0.21),
            ("sgd", 0.80),
            ("sgd", 0.78),
            ("sgd", 0.82),
        )
    ]

    analysis = StudyInsightAnalyzer.analyze(candidates, {"metric": "loss", "mode": "min"})
    by_name = {value["parameter"]: value for value in analysis["parameters"]}

    assert by_name["optimizer"]["best_observed_value"] == "adamw"
    assert by_name["optimizer"]["status"] == "actionable"
    assert by_name["momentum"]["inactive_observations"] == 3


def test_hpo_insights_expose_visual_response_and_pairwise_joint_gain() -> None:
    candidates = [
        {
            "parameters": {"left": left, "right": right},
            "selection_objective": float(left != right),
            "runs": [{"state": "succeeded"}],
        }
        for _repeat in range(3)
        for left in (False, True)
        for right in (False, True)
    ]

    analysis = StudyInsightAnalyzer.analyze(candidates, {"metric": "score", "mode": "max"})
    left = next(value for value in analysis["parameters"] if value["parameter"] == "left")

    assert left["response"]["kind"] == "categorical"
    assert left["joint_relationships"][0]["parameter"] == "right"
    assert left["joint_relationships"][0]["gain"] > 0
    assert analysis["interaction_matrix"]["parameters"] == ["left", "right"]

    payload = {
        "work": {
            "items": [
                {
                    "study": {
                        "strategy": "adaptive",
                        "hpo_analysis": analysis,
                    }
                }
            ]
        }
    }
    screen = StudyParameterInsightRenderer.render(
        payload,
        0,
        selected_parameter=0,
        message="",
        width=120,
        height=32,
    )
    assert "MARGINAL RESPONSE" in screen
    assert "PAIRWISE JOINT PREDICTIVE GAIN" in screen
    assert "right" in screen


def test_hpo_insights_show_descriptive_evidence_from_two_candidates() -> None:
    candidates = [
        {
            "parameters": {"embedding_dim": dimension, "dropout": dropout},
            "selection_objective": objective,
            "runs": [{"state": "succeeded"}],
        }
        for dimension, dropout, objective in ((32, 0.1, 0.55), (64, 0.4, 0.62))
    ]

    analysis = StudyInsightAnalyzer.analyze(candidates, {"metric": "auprc", "mode": "max"})
    embedding = next(
        value for value in analysis["parameters"] if value["parameter"] == "embedding_dim"
    )

    assert embedding["observations"] == 2
    assert embedding["status"] == "learning"
    assert len(embedding["response"]["points"]) == 2
    assert embedding["joint_relationships"][0]["status"] == "coverage-only"
    assert embedding["joint_relationships"][0]["observations"] == 2


def test_hpo_renderer_recomputes_an_old_remote_analysis_snapshot_locally() -> None:
    candidates = [
        {
            "parameters": {"embedding_dim": dimension},
            "selection_objective": objective,
            "runs": [{"state": "succeeded"}],
        }
        for dimension, objective in ((16, 0.51), (32, 0.57), (64, 0.61))
    ]
    payload = {
        "work": {
            "items": [
                {
                    "study": {
                        "strategy": "adaptive",
                        "objective": {"metric": "val_auprc", "mode": "max"},
                        "candidates": candidates,
                        "hpo_analysis": {
                            "analysis_version": 2,
                            "parameters": [
                                {
                                    "parameter": "embedding_dim",
                                    "observations": 3,
                                    "distinct_values": 3,
                                }
                            ],
                        },
                    }
                }
            ]
        }
    }

    analysis, parameters = StudyInsightRenderer.analysis(payload, 0)

    assert analysis is not None
    assert analysis["analysis_version"] == 3
    assert len(parameters[0]["response"]["points"]) == 3
    screen = StudyParameterInsightRenderer.render(
        payload,
        0,
        selected_parameter=0,
        message="",
        width=120,
        height=32,
    )
    assert "More distinct completed candidates" not in screen
    assert "16→0.51" in screen


def test_study_snapshot_and_renderer_expose_hpo_analysis_and_latest_action(
    tmp_path: Path,
) -> None:
    study = StudyTelemetry(tmp_path / "study")
    specifications = tuple(
        {
            "trial_index": index,
            "seed": 4,
            "trial_parameters": {"width": index * 32},
        }
        for index in range(1, 6)
    )
    study.initialize(
        name="training",
        execution_id="execution-1",
        strategy="adaptive",
        objective={"metric": "score", "mode": "max"},
        specifications=specifications,
    )
    for index, specification in enumerate(specifications, 1):
        study.schedule((specification,))
        key = f"trial-{index:05d}-seed-4"
        study._write_run(  # controller-owned fixture evidence
            key,
            {
                "trial": index,
                "seed": 4,
                "state": "succeeded",
                "metrics": {"score": float(index)},
                "objective_observation": {
                    "metric": "score",
                    "mode": "max",
                    "current": float(index),
                    "current_step": 5,
                    "best": float(index),
                    "best_step": 5,
                },
            },
        )
    study.controller_decision({"decision": 7, "action": "ADD_SEED", "trial": 5, "seed": 17})
    snapshot = study.refresh()
    payload = {"work": {"items": [{"study": snapshot}]}}

    assert snapshot["hpo_analysis"]["parameters"][0]["parameter"] == "width"
    assert snapshot["controller"]["last"]["action"] == "ADD_SEED"
    screen = StudyInsightRenderer.render(
        payload,
        0,
        selected_parameter=0,
        message="",
        width=140,
        height=32,
    )
    assert "LambdaForge HPO insights" in screen
    assert "controller: ADD_SEED trial=5 seed=17" in screen
    assert "SELECTED HYPERPARAMETER · width" in screen
    assert "associations are exploratory, not causal" in screen


def test_curve_downsampling_preserves_the_exact_best_epoch() -> None:
    values = tuple(
        {"step": step, "value": 10.0 if step == 37 else float(step) / 1000}
        for step in range(1, 102)
    )

    reduced = JobService._downsample_curve(values, 10, preserve_step=37)

    assert len(reduced) == 10
    assert {value["step"] for value in reduced} >= {1, 37, 101}


def test_job_service_returns_downsampled_curves_and_only_the_selected_run_log(
    tmp_path: Path,
) -> None:
    job_root = tmp_path / "runtime" / "job-1"
    work_root = job_root / "work"
    run_dir = work_root / "runs" / "run-1" / "attempts" / "attempt-1"
    run_dir.mkdir(parents=True)
    log = run_dir / "work.log"
    log.write_text("selected run only\n", encoding="utf-8")
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "result_version": 1,
                "status": "failed",
                "failure": {
                    "type": "RuntimeError",
                    "message": "training exploded",
                    "traceback": (
                        "Traceback (most recent call last):\nRuntimeError: training exploded"
                    ),
                    "diagnostic": {"operation": "Training.run"},
                },
            }
        ),
        encoding="utf-8",
    )
    metrics = run_dir / "training-metrics.jsonl"
    metrics.write_text(
        json.dumps(
            {
                "kind": "chart-filter",
                "include": ["val_*", "epoch_time_s"],
                "exclude": ["*_aux"],
            }
        )
        + "\n"
        + "".join(
            json.dumps({"name": "val_loss", "value": 1 / step, "step": step, "split": None}) + "\n"
            for step in range(1, 101)
        ),
        encoding="utf-8",
    )
    study_root = job_root / "study"
    study_root.mkdir()
    (study_root / "summary.json").write_text(
        json.dumps(
            {
                "study_telemetry_version": 1,
                "objective": {"metric": "val_loss", "mode": "min"},
                "candidates": [
                    {
                        "trial": 1,
                        "parameters": {"width": 128},
                        "runs": [
                            {
                                "key": "trial-00001-seed-4",
                                "trial": 1,
                                "seed": 4,
                                "state": "failed",
                                "failure": {
                                    "type": "RuntimeError",
                                    "message": "training exploded",
                                },
                                "run_dir": str(run_dir),
                                "log_path": str(log),
                                "training_metrics_path": str(metrics),
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    now = datetime.now(timezone.utc).isoformat()
    store = JobStore(tmp_path / "records")
    store.write(
        JobRecord(
            "job-1",
            "local",
            "local",
            "provider-1",
            JobState.RUNNING,
            ("python", "work.py"),
            str(work_root),
            {},
            now,
            now,
            metadata={"name": "training"},
            job_type="work",
        )
    )
    profile = ClusterProfile(
        "local",
        storage={
            "state_root": str(tmp_path / "state"),
            "cache_root": str(tmp_path / "cache"),
            "run_root": str(tmp_path / "runtime"),
        },
    )
    service = JobService(ClusterCatalog({"local": profile}), store)

    detail = service.study_run("job-1", "trial-00001-seed-4", curve_points=20)

    assert detail["log"] == "selected run only\n"
    assert detail["parameters"] == {"width": 128}
    assert detail["chart_filter"] == {
        "include": ["val_*", "epoch_time_s"],
        "exclude": ["*_aux"],
    }
    assert len(detail["curves"]["val_loss"]) == 20
    assert detail["curves"]["val_loss"][0]["step"] == 1
    assert detail["curves"]["val_loss"][-1]["step"] == 100
    assert detail["best_step"] == 100
    assert detail["best_objective"] == pytest.approx(0.01)
    assert detail["failure"]["type"] == "RuntimeError"
    assert detail["failure"]["phase"] == "Training.run"
    assert "Traceback" in detail["failure"]["traceback"]
    assert detail["paths"]["result"].endswith("/result.json")
    with pytest.raises(ValueError, match="Invalid study Run key"):
        service.study_run("job-1", "../../escape")


def test_failed_study_run_has_an_expandable_persisted_failure_view() -> None:
    detail = {
        "trial": 2,
        "seed": 7,
        "state": "failed",
        "parameters": {"width": 128},
        "curves": {},
        "latest_metrics": {},
        "failure": {
            "type": "RuntimeError",
            "message": "training exploded",
            "phase": "Training.run",
            "result_path": "/owned/run/result.json",
            "traceback": "Traceback line one\nRuntimeError: training exploded",
        },
    }

    compact = StudyRunRenderer.render(
        detail,
        scroll=0,
        show_failure=False,
        message="",
        width=120,
        height=30,
    )
    expanded = StudyRunRenderer.render(
        detail,
        scroll=0,
        show_failure=True,
        message="",
        width=120,
        height=30,
    )

    assert "failure: RuntimeError: training exploded" in compact
    assert "e failure details" in compact
    assert "SCIENTIFIC FAILURE" in expanded
    assert "Phase: Training.run" in expanded
    assert "Persisted result: /owned/run/result.json" in expanded
    assert "Traceback line one" in expanded


def test_lightning_bridge_records_scalar_curves_and_validation_timing(tmp_path: Path) -> None:
    path = tmp_path / "training.jsonl"
    callback = AdaptiveHpoCallback(
        None,
        None,
        None,
        path,
        chart_include=["val_*", "epoch_time_s"],
        chart_exclude=["*_aux"],
    )
    trainer = SimpleNamespace(
        sanity_checking=False,
        current_epoch=2,
        callback_metrics={"train_loss": 0.4, "val_auprc": 0.91, "epoch": 2},
        should_stop=False,
    )

    callback.on_validation_epoch_start(trainer)
    callback.on_validation_epoch_end(trainer)

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    chart_filter = records[0]
    values = [value for value in records if value.get("kind") != "chart-filter"]
    assert chart_filter == {
        "kind": "chart-filter",
        "include": ["val_*", "epoch_time_s"],
        "exclude": ["*_aux"],
    }
    assert {value["name"] for value in values} >= {
        "train_loss",
        "val_auprc",
        "validation_time_s",
    }
    assert {value["step"] for value in values} == {3}


def test_lightning_bridge_records_epoch_wall_time_once_per_step(tmp_path: Path) -> None:
    path = tmp_path / "training.jsonl"
    callback = AdaptiveHpoCallback(None, None, None, path)
    trainer = SimpleNamespace(
        sanity_checking=False,
        current_epoch=0,
        callback_metrics={"train_loss": 0.4},
        should_stop=False,
        is_global_zero=True,
    )

    callback.on_train_epoch_start(trainer)
    callback.on_train_epoch_end(trainer)
    callback.on_fit_end(trainer)

    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert sum(value["name"] == "epoch_time_s" for value in values) == 1
    assert sum(value["name"] == "train_loss" for value in values) == 1


def test_study_renderers_expose_candidates_seeds_curves_and_failure() -> None:
    run = {
        "key": "trial-00001-seed-4",
        "seed": 4,
        "state": "failed",
        "latest_step": 8,
        "best_step": 6,
        "best_objective": 0.84,
        "gpu_index": 1,
        "gpu_token": "3",
        "latest_metrics": {"val_auprc": 0.8},
    }
    study = {
        "study_telemetry_version": 1,
        "name": "wisdom-v1",
        "strategy": "adaptive",
        "objective": {"metric": "val_auprc", "mode": "max"},
        "counts": {"candidates": 1, "scheduled_runs": 1, "failed_runs": 1},
        "candidates": [
            {
                "trial": 1,
                "state": "running",
                "parameters": {"hidden_dim": 128},
                "latest_metrics": {"val_auprc": 0.8},
                "current_objective": 0.8,
                "best_objective": 0.84,
                "best_seed": 4,
                "best_step": 6,
                "runs": [run],
            }
        ],
    }
    payload = {"work": {"items": [{"name": "wisdom-v1", "study": study}]}}

    candidate_screen = StudyRenderer.render_candidates(
        payload, 0, selected_candidate=0, message="", width=120, height=24
    )
    run_screen = StudyRenderer.render_runs(
        payload, 0, 0, selected_run=0, message="", width=120, height=24
    )
    detail_screen = StudyRunRenderer.render(
        {
            "trial": 1,
            "seed": 4,
            "state": "failed",
            "gpu_index": 1,
            "gpu_token": "3",
            "parameters": {"hidden_dim": 128},
            "duration_seconds": 9,
            "latest_metrics": {"val_auprc": 0.8, "epoch_time_s": 1.2},
            "curves": {
                "val_auprc": [
                    {"step": 1, "value": 0.5},
                    {"step": 2, "value": 0.8},
                ],
                "train_loss": [
                    {"step": 1, "value": 0.9},
                    {"step": 2, "value": 0.4},
                ],
                "epoch_time_s": [
                    {"step": 1, "value": 1.5},
                    {"step": 2, "value": 1.2},
                ],
            },
            "objective": {"metric": "val_auprc", "mode": "max"},
            "chart_filter": {"include": ["val_*", "epoch_time_s"]},
            "failure": {"type": "RuntimeError", "message": "out of memory"},
            "log": "exclusive line\n",
        },
        scroll=0,
        message="",
        width=120,
        height=24,
    )
    raw_screen = StudyRunRenderer.render(
        {
            "trial": 1,
            "seed": 4,
            "state": "failed",
            "parameters": {"hidden_dim": 128},
            "curves": {},
            "log": "exclusive line\n",
        },
        scroll=0,
        show_raw_log=True,
        message="",
        width=120,
        height=24,
    )

    assert "Trial 1" in candidate_screen and "hidden_dim" in candidate_screen
    assert "Seed 4" in run_screen and "8" in run_screen and "6" in run_screen
    assert "GPU 3" in run_screen and "GPU 3" in detail_screen
    assert "BEST OBJECTIVE" in candidate_screen
    assert "CURRENT OBJECTIVE" in candidate_screen
    assert "seed 4 @ 6" in candidate_screen
    assert "LAST EPOCH" in run_screen and "BEST EPOCH" in run_screen
    assert "BEST OBJECTIVE" in run_screen and "0.84" in run_screen
    assert "PARAMETER SUMMARY" not in candidate_screen
    assert "LATEST METRICS" not in run_screen
    assert "SELECTED SEED/RUN METRICS" in run_screen
    assert "val_auprc = 0.8" in run_screen
    assert "val_auprc" in detail_screen and "RuntimeError: out of memory" in detail_screen
    assert "EPOCH METRICS" in detail_screen and "SELECTED EPOCH 2" in detail_screen
    assert "★" in detail_screen and "◆" in detail_screen
    assert "exclusive line" in raw_screen


def test_seed_table_keeps_multiple_rows_visible_in_a_common_terminal() -> None:
    runs = [
        {
            "key": f"trial-00001-seed-{seed}",
            "seed": seed,
            "state": "succeeded",
            "latest_step": 40,
            "best_step": 31,
            "best_objective": 0.8 + seed / 1000,
            "latest_metrics": {
                "score": 0.7 + seed / 1000,
                **{f"metric_{index}": float(index) for index in range(12)},
            },
        }
        for seed in range(1, 9)
    ]
    payload = {
        "work": {
            "items": [
                {
                    "study": {
                        "name": "training",
                        "strategy": "adaptive",
                        "objective": {"metric": "score", "mode": "max"},
                        "candidates": [
                            {
                                "trial": 1,
                                "state": "completed",
                                "parameters": {f"parameter_{index}": index for index in range(12)},
                                "runs": runs,
                            }
                        ],
                    }
                }
            ]
        }
    }

    screen = StudyRenderer.render_runs(
        payload,
        0,
        0,
        selected_run=0,
        message="",
        width=150,
        height=24,
    )

    assert "Seed 1" in screen
    assert "Seed 2" in screen
    assert "Seed 3" in screen
    assert "SELECTED SEED/RUN METRICS" in screen


def test_study_dashboard_pages_curves_lists_parameters_and_expands_epoch() -> None:
    curves = {
        name: [
            {"step": 1, "value": start},
            {"step": 2, "value": start + 0.1},
        ]
        for name, start in {
            "objective": 0.5,
            "epoch_time_s": 10.0,
            "validation_time_s": 2.0,
            "train_loss": 0.9,
            "val_accuracy": 0.6,
            "gpu_mem_mb": 512.0,
        }.items()
    }
    detail = {
        "trial": 3,
        "seed": 7,
        "state": "running",
        "parameters": {
            "dropout": 0.25,
            "hidden_dim": 256,
            "learning_rate": 0.0003,
        },
        "duration_seconds": 31,
        "latest_metrics": {name: values[-1]["value"] for name, values in curves.items()},
        "curves": curves,
        "objective": {"metric": "objective", "mode": "max"},
        "log": "raw",
    }

    first = StudyRunRenderer.render(
        detail,
        scroll=0,
        chart_page=0,
        selected_epoch=0,
        message="",
        width=140,
        height=40,
    )
    second = StudyRunRenderer.render(
        detail,
        scroll=0,
        chart_page=1,
        selected_epoch=0,
        message="",
        width=140,
        height=40,
    )
    expanded = StudyEpochRenderer.render(
        detail,
        selected_epoch=0,
        message="",
        width=140,
        height=30,
    )

    assert "PARAMETERS (3)" in first
    assert "dropout = 0.25" in first and "hidden_dim = 256" in first
    assert "page 1/2" in first and "page 2/2" in second
    assert "n/p curve page" in first
    assert "●" in first
    assert "ALL EPOCH METRICS (6)" in expanded
    assert all(name in expanded for name in curves)
