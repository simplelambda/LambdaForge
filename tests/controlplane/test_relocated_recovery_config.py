"""Public recovery configuration selection over exact native persisted executions."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lambdaforge.cli.parser import build_parser
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.work import WorkConfig, WorkRunner
from lambdaforge.work.models import atomic_json
from lambdaforge.work.runner import _execute_run
from tests.controlplane.test_study_retry import recovery_jobs as recovery_fixture


@pytest.fixture
def recovery_jobs(tmp_path: Path) -> Any:
    return recovery_fixture.__wrapped__(tmp_path)


def persisted(jobs: JobService, previous: Any, destination: Path) -> tuple[Path, Path]:
    source = Path(previous.config_path)
    source.write_text(
        "name: recovery\nrun: tests.work_cases.FixedRecoveryWork\n"
        "seeds: [4, 7, 32, 54, 65, 94, 109, 124, 142, 167]\n"
        "objective: {metric: score, mode: max}\nresources: {cpu: 1}\n"
        "execution: {failure_retries: 0}\n"
    )
    result = WorkRunner(
        dispatcher=lambda values, **kwargs: tuple(_execute_run(value) for value in values),
        execution_root=Path(previous.work_dir) / ".lambdaforge" / "runs" / "recovery",
    ).run(WorkConfig.from_yaml(source))
    atomic_json(Path(previous.work_dir).parent / "result.json", result.to_dict())
    destination.write_text(
        "# Relocated YAML; comments and key order do not define identity\n" + source.read_text()
    )
    source.unlink()
    return destination, result.execution_dir


def test_public_preview_uses_original_snapshot_when_yaml_was_moved(
    recovery_jobs: Any, tmp_path: Path
) -> None:
    jobs, previous, _ = recovery_jobs
    selected, execution = persisted(jobs, previous, tmp_path / "relocated.yaml")
    original = {p: p.read_bytes() for p in execution.rglob("*") if p.is_file()}
    registry = {p: p.read_bytes() for p in jobs.store.root.rglob("*") if p.is_file()}
    preview = jobs.retry_preview(previous.job_id, config=selected)
    assert preview["configuration"]["scientific_compatibility"] == "verified"
    assert preview["configuration"]["sha256"] == hashlib.sha256(selected.read_bytes()).hexdigest()
    assert (preview["reuse_runs"], preview["retry_runs"], preview["pending_runs"]) == (9, 1, 0)
    planned = jobs.retry(previous.job_id, config=selected, dry_run=True)
    assert planned.recovery_plan["configuration"]["selected_path"] == str(selected)
    assert original == {p: p.read_bytes() for p in execution.rglob("*") if p.is_file()}
    assert registry == {p: p.read_bytes() for p in jobs.store.root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="Historical recovery YAML is missing"):
        jobs.retry(previous.job_id)


def test_semantic_change_and_post_preview_edit_cannot_override_identity(
    recovery_jobs: Any, tmp_path: Path
) -> None:
    jobs, previous, _ = recovery_jobs
    selected, _execution = persisted(jobs, previous, tmp_path / "relocated.yaml")
    preview = jobs.retry_preview(previous.job_id, config=selected)
    text = selected.read_text()
    selected.write_text(text + "# Edited after preview\n")
    with pytest.raises(ValueError, match="changed after recovery preview"):
        jobs.retry(
            previous.job_id,
            config=selected,
            config_sha256=preview["configuration"]["sha256"],
            accept_code_change=True,
        )
    selected.write_text(text.replace("mode: max", "mode: min"))
    with pytest.raises(ValueError, match="unchanged configuration"):
        jobs.retry(previous.job_id, config=selected, accept_code_change=True)
    assert len(jobs.store.records()) == 1


def test_missing_or_corrupt_original_identity_fails_actionably(
    recovery_jobs: Any, tmp_path: Path
) -> None:
    jobs, previous, _ = recovery_jobs
    selected, execution = persisted(jobs, previous, tmp_path / "relocated.yaml")
    manifest = json.loads((execution / "execution.json").read_text())
    manifest.pop("scientific_fingerprint")
    atomic_json(execution / "execution.json", manifest)
    with pytest.raises(ValueError, match="Original scientific configuration identity is missing"):
        jobs.retry_preview(previous.job_id, config=selected)


def test_retry_cli_exposes_same_configuration_selection() -> None:
    parser = build_parser()
    semantic = parser.parse_args(["retry", "work-a", "--config", "moved.yaml", "--dry-run"])
    native = parser.parse_args(["jobs", "retry", "job-a", "--config", "moved.yaml", "--dry-run"])
    assert semantic.config == native.config == Path("moved.yaml")


def test_accepted_recovery_request_is_audited_and_native_runner_reuses_successes(
    recovery_jobs: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs, previous, _ = recovery_jobs
    selected, execution = persisted(jobs, previous, tmp_path / "relocated.yaml")
    preview = jobs.retry_preview(previous.job_id, config=selected)
    origin = (execution / "execution.json").read_bytes()
    successful = {
        path: path.read_bytes()
        for path in execution.glob("runs/*/attempts/*/result.json")
        if json.loads(path.read_text()).get("status") == "succeeded"
    }
    actual_popen = subprocess.Popen

    def launch(command: Any, *args: Any, **kwargs: Any) -> Any:
        if "lambdaforge.controlplane.SubmissionWorker" in command:
            return SimpleNamespace(pid=1234)
        return actual_popen(command, *args, **kwargs)

    monkeypatch.setattr("subprocess.Popen", launch)
    handle = jobs.retry(
        previous.job_id,
        config=selected,
        config_sha256=preview["configuration"]["sha256"],
        accept_code_change=True,
    )
    assert handle.recovery_request_id is not None
    accepted = jobs.get(handle.job_id, refresh=False)
    assert accepted.to_dict()["recovery"]["phase"] == "accepted"
    assert accepted.metadata["recovery_request"]["plan"] == {
        "reuse_runs": 9,
        "retry_runs": 1,
        "pending_runs": 0,
        "preserved_completed_runs": 9,
    }
    origin_events = jobs.store.events(previous.job_id)
    assert origin_events[-1]["recovery_request_id"] == handle.recovery_request_id
    assert origin_events[-1]["phase"] == "RECOVERY_ACCEPTED"
    assert jobs.store.events(handle.job_id)[0]["recovery_request_id"] == handle.recovery_request_id
    jobs.update_preparation(handle.job_id, "validation")
    assert jobs.get(handle.job_id, refresh=False).to_dict()["recovery"]["phase"] == "preparing"
    restored = WorkRunner(
        dispatcher=lambda values, **kwargs: tuple(_execute_run(value) for value in values),
    ).run(WorkConfig.from_yaml(selected), resume_execution=execution, accept_code_change=True)
    assert restored.status == "succeeded"
    assert restored.execution_id == execution.name
    assert next(run for run in restored.runs if run.seed == 54).attempt_number == 2
    assert origin == (execution / "execution.json").read_bytes()
    assert all(path.read_bytes() == data for path, data in successful.items())
