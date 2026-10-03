"""Recover real CPU studies from durable controller state without any remote execution."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from lambdaforge.controlplane.StudyExportService import _ARCHIVE_SCRIPT
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.work import WorkConfig, WorkRunner
from lambdaforge.work.models import atomic_json


def config_at(root: Path) -> WorkConfig:
    return WorkConfig.from_mapping(
        {
            "name": "recoverable",
            "run": "tests.work_cases.RecoverableStudyWork",
            "seeds": [1],
            "resources": {"cpu": 1},
            "search": {
                "trials": 2,
                "startup_trials": 2,
                "max_parallel": 1,
                "confirmation_seeds": [],
                "early_stopping": False,
                "choice": {"values": [0, 1]},
            },
            "objective": {"metric": "score", "mode": "max"},
        },
        source=root / "study.yaml",
    )


def test_retry_keeps_valid_evidence_and_resumes_only_failed_attempt(tmp_path: Path) -> None:
    config = config_at(tmp_path)
    first = WorkRunner().run(config)
    assert first.status == "failed"
    good = next(run for run in first.runs if run.ok)
    bad = next(run for run in first.runs if not run.ok)
    decisions = first.execution_dir / "hpo-control" / "decisions.jsonl"
    original_history = decisions.read_text()
    original_map = json.loads((first.execution_dir / "hpo-control" / "state.json").read_text())[
        "public_trial_map"
    ]
    second = WorkRunner().run(config, resume_execution=first.execution_dir)
    assert second.status == "succeeded"
    assert len(second.runs) == 2
    assert next(run for run in second.runs if run.parameters["choice"] == 1).run_dir == good.run_dir
    recovered = next(run for run in second.runs if run.parameters["choice"] == 0)
    assert recovered.attempt_number == bad.attempt_number + 1
    assert recovered.resumed_from_checkpoint
    assert recovered.run_id == bad.run_id
    assert decisions.read_text().startswith(original_history)
    state = json.loads((first.execution_dir / "hpo-control" / "state.json").read_text())
    assert len(state["runs"]) == 3  # Physical failure remains auditable and budgeted.
    assert state["public_trial_map"] == original_map


def test_code_fix_requires_acknowledgement_and_preserves_real_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = config_at(tmp_path)
    first = WorkRunner().run(config)
    origin = (first.execution_dir / "execution.json").read_bytes()
    runner = importlib.import_module("lambdaforge.work.runner")
    monkeypatch.setattr(runner, "_code_identity", lambda root: {"source": "corrected-code"})
    with pytest.raises(ValueError, match="Consumer code changed"):
        WorkRunner().run(config, resume_execution=first.execution_dir)
    second = WorkRunner().run(config, resume_execution=first.execution_dir, accept_code_change=True)
    assert second.status == "succeeded"
    assert second.scientific_fingerprint != first.scientific_fingerprint
    assert (first.execution_dir / "execution.json").read_bytes() == origin
    assert next(run for run in second.runs if run.parameters["choice"] == 0).resumed_from_checkpoint
    assert json.loads((first.execution_dir / "recovery-history.jsonl").read_text())[
        "code_identity"
    ] == {"source": "corrected-code"}


def test_changed_configuration_or_missing_evidence_fails_closed(tmp_path: Path) -> None:
    config = config_at(tmp_path)
    first = WorkRunner().run(config)
    changed = WorkConfig.from_mapping({**config.raw, "seeds": [2]}, source=config.source)
    with pytest.raises(ValueError, match="unchanged configuration"):
        WorkRunner().run(changed, resume_execution=first.execution_dir, accept_code_change=True)
    first.runs[0].run_dir.joinpath("result.json").unlink()
    with pytest.raises(ValueError, match="regular persisted file"):
        WorkRunner().run(config, resume_execution=first.execution_dir)


def test_recovery_dry_run_does_not_mutate_controller_and_lock_prevents_duplicates(
    tmp_path: Path,
) -> None:
    config = config_at(tmp_path)
    first = WorkRunner().run(config)
    state = first.execution_dir / "hpo-control" / "state.json"
    before = state.read_bytes()
    plan = WorkRunner().run(config, resume_execution=first.execution_dir, dry_run=True)
    assert plan.execution_id == first.execution_id
    assert state.read_bytes() == before
    assert not (first.execution_dir / "recovery-history.jsonl").exists()
    with CrossProcessFileLock(
        first.execution_dir / ".controller.lock",
        shared=False,
        timeout_seconds=0.1,
        poll_interval_seconds=0.01,
    ):
        with pytest.raises(TimeoutError):
            WorkRunner().run(config, resume_execution=first.execution_dir)


def test_interrupted_pending_seed_restores_exact_identity(tmp_path: Path) -> None:
    config = config_at(tmp_path)
    first = WorkRunner().run(config)
    state_path = first.execution_dir / "hpo-control" / "state.json"
    state = json.loads(state_path.read_text())
    removed = next(value for value in state["runs"] if value["status"] == "failed")
    state["runs"] = [value for value in state["runs"] if value is not removed]
    state["pending_actions"] = [
        {"candidate_pool_index": removed["pool_trial"], "seed": removed["seed"], "phase": "search"}
    ]
    atomic_json(state_path, state)
    second = WorkRunner().run(config, resume_execution=first.execution_dir)
    assert second.status == "succeeded"
    assert {run.seed for run in second.runs} == {1}


def test_missing_checkpoint_restarts_failed_run_without_repeating_valid_evidence(
    tmp_path: Path,
) -> None:
    config = config_at(tmp_path)
    first = WorkRunner().run(config)
    bad = next(run for run in first.runs if not run.ok)
    checkpoint_root = bad.run_dir.parent.parent / "checkpoints"
    for path in checkpoint_root.rglob("*"):
        if path.is_file():
            path.unlink()
    second = WorkRunner().run(config, resume_execution=first.execution_dir)
    retried = next(run for run in second.runs if not run.ok)
    assert retried.attempt_number == bad.attempt_number + 1
    assert retried.resumed_from_checkpoint is False
    assert next(run.run_dir for run in second.runs if run.ok) == next(
        run.run_dir for run in first.runs if run.ok
    )


def test_retry_does_not_reset_spent_run_budget(tmp_path: Path) -> None:
    original = config_at(tmp_path)
    config = WorkConfig.from_mapping(
        {**original.raw, "search": {**original.raw["search"], "max_runs": 2}},
        source=original.source,
    )
    first = WorkRunner().run(config)
    second = WorkRunner().run(config, resume_execution=first.execution_dir)
    assert second.status == "failed"
    assert [run.run_dir for run in second.runs] == [run.run_dir for run in first.runs]
    state = json.loads((first.execution_dir / "hpo-control" / "state.json").read_text())
    assert len(state["runs"]) == 2


def test_valid_confirmation_seeds_are_not_repeated_on_recovery(tmp_path: Path) -> None:
    original = config_at(tmp_path)
    config = WorkConfig.from_mapping(
        {
            **original.raw,
            "run": "tests.work_cases.FlatAdaptiveScoreWork",
            "search": {
                "trials": 2,
                "startup_trials": 2,
                "max_parallel": 1,
                "confirmation_seeds": [99],
                "early_stopping": False,
                "choice": {"values": [0, 1]},
            },
        },
        source=original.source,
    )
    first = WorkRunner().run(config)
    manifest = first.execution_dir / "result.json"
    terminal = json.loads(manifest.read_text())
    terminal["status"] = "failed"  # Failure after publishing all valid Run evidence.
    atomic_json(manifest, terminal)
    second = WorkRunner().run(config, resume_execution=first.execution_dir)
    assert second.status == "succeeded"
    assert [run.run_dir for run in second.runs] == [run.run_dir for run in first.runs]
    assert any(run.study_phase == "confirmation" for run in second.runs)


def test_another_interruption_keeps_deferred_recovery_seed_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = config_at(tmp_path)
    first = WorkRunner().run(config)
    state_path = first.execution_dir / "hpo-control" / "state.json"
    state = json.loads(state_path.read_text())
    good = next(value for value in state["runs"] if value["status"] == "succeeded")
    state["pending_actions"] = [
        {"candidate_pool_index": good["pool_trial"], "seed": 2, "phase": "search"}
    ]
    atomic_json(state_path, state)
    runner = importlib.import_module("lambdaforge.work.runner")

    def interrupted(*args: object, **kwargs: object) -> None:
        raise RuntimeError("controller interrupted again")

    monkeypatch.setattr(runner, "_execute_adaptive_dispatch", interrupted)
    with pytest.raises(RuntimeError, match="interrupted again"):
        WorkRunner().run(config, resume_execution=first.execution_dir)
    after = json.loads(state_path.read_text())
    assert {value["seed"] for value in after["pending_actions"]} == {1, 2}


def test_recovered_execution_can_be_exported_from_new_job(tmp_path: Path) -> None:
    owner = tmp_path / "job-owner" / "work"
    owner.mkdir(parents=True)
    first = WorkRunner().run(config_at(owner))
    second = WorkRunner().run(config_at(owner), resume_execution=first.execution_dir)
    job = tmp_path / "job-new"
    (job / "work").mkdir(parents=True)
    archive = job / "study.zip"
    process = subprocess.run(
        (
            sys.executable,
            "-c",
            _ARCHIVE_SCRIPT,
            str(job),
            str(archive),
            second.execution_id,
            job.name,
            "default",
            str(second.execution_dir),
            str(owner),
        ),
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    with zipfile.ZipFile(archive) as package:
        assert any(path.endswith("recovery-history.jsonl") for path in package.namelist())
        assert any(path.endswith("work.log") for path in package.namelist())


def test_new_job_rebuilds_live_study_view_and_keeps_completed_run_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = config_at(tmp_path)
    first = WorkRunner().run(config)  # Older job has no console telemetry at all.
    telemetry_root = tmp_path / "new-job" / "study"
    monkeypatch.setenv("LAMBDAFORGE_STUDY_PATH", str(telemetry_root))
    runner = importlib.import_module("lambdaforge.work.runner")
    dispatch = runner._execute_adaptive_dispatch

    def inspect(*args: object, **kwargs: object) -> object:
        index = json.loads((telemetry_root / "index.json").read_text())
        assert index["finished"] is False
        assert "status" not in index
        return dispatch(*args, **kwargs)

    monkeypatch.setattr(runner, "_execute_adaptive_dispatch", inspect)
    second = WorkRunner().run(config, resume_execution=first.execution_dir)
    assert second.status == "succeeded"
    index = json.loads((telemetry_root / "interactive.json").read_text())
    assert len(index["candidates"]) == 2
    assert index["counts"]["failed_runs"] == 0
    for result in second.runs:
        assert any(
            json.loads(path.read_text()).get("run_dir") == str(result.run_dir)
            for path in (telemetry_root / "runs").glob("*.json")
        )
