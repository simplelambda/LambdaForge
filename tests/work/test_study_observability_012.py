"""Compact, isolated observability for concurrent scientific study Runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from lambdaforge.cli.LiveJobMonitor import StudyRenderer, StudyRunRenderer
from lambdaforge.controlplane import ClusterCatalog, ClusterProfile, JobService, JobState, JobStore
from lambdaforge.controlplane.jobs import JobRecord
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
        "\n".join(
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
    assert snapshot["counts"]["active_runs"] == 1
    assert json.loads((tmp_path / "progress.json").read_text())["completed"] == 0
    assert training.read_text(encoding="utf-8").count("val_loss") == 2


def test_job_service_returns_downsampled_curves_and_only_the_selected_run_log(
    tmp_path: Path,
) -> None:
    job_root = tmp_path / "runtime" / "job-1"
    work_root = job_root / "work"
    run_dir = work_root / "runs" / "run-1" / "attempts" / "attempt-1"
    run_dir.mkdir(parents=True)
    log = run_dir / "work.log"
    log.write_text("selected run only\n", encoding="utf-8")
    metrics = run_dir / "training-metrics.jsonl"
    metrics.write_text(
        "".join(
            json.dumps(
                {"name": "val_loss", "value": 1 / step, "step": step, "split": None}
            )
            + "\n"
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
                "candidates": [
                    {
                        "trial": 1,
                        "parameters": {"width": 128},
                        "runs": [
                            {
                                "key": "trial-00001-seed-4",
                                "trial": 1,
                                "seed": 4,
                                "state": "running",
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
    assert len(detail["curves"]["val_loss"]) == 20
    assert detail["curves"]["val_loss"][0]["step"] == 1
    assert detail["curves"]["val_loss"][-1]["step"] == 100
    with pytest.raises(ValueError, match="Invalid study Run key"):
        service.study_run("job-1", "../../escape")


def test_lightning_bridge_records_scalar_curves_and_validation_timing(tmp_path: Path) -> None:
    path = tmp_path / "training.jsonl"
    callback = AdaptiveHpoCallback(None, None, None, path)
    trainer = SimpleNamespace(
        sanity_checking=False,
        current_epoch=2,
        callback_metrics={"train_loss": 0.4, "val_auprc": 0.91, "epoch": 2},
        should_stop=False,
    )

    callback.on_validation_epoch_start(trainer)
    callback.on_validation_epoch_end(trainer)

    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert {value["name"] for value in values} >= {
        "train_loss",
        "val_auprc",
        "validation_time_s",
    }
    assert {value["step"] for value in values} == {3}


def test_study_renderers_expose_candidates_seeds_curves_and_failure() -> None:
    run = {
        "key": "trial-00001-seed-4",
        "seed": 4,
        "state": "failed",
        "latest_step": 8,
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
            "parameters": {"hidden_dim": 128},
            "duration_seconds": 9,
            "latest_metrics": {"val_auprc": 0.8, "epoch_time_s": 1.2},
            "curves": {
                "val_auprc": [
                    {"step": 1, "value": 0.5},
                    {"step": 2, "value": 0.8},
                ]
            },
            "failure": {"type": "RuntimeError", "message": "out of memory"},
            "log": "exclusive line\n",
        },
        scroll=0,
        message="",
        width=120,
        height=24,
    )

    assert "Trial 1" in candidate_screen and "hidden_dim" in candidate_screen
    assert "Seed 4" in run_screen and "8" in run_screen
    assert "val_auprc" in detail_screen and "RuntimeError: out of memory" in detail_screen
    assert "exclusive line" in detail_screen
