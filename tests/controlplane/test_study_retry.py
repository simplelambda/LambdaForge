"""Recovery handoff and storage protection using local files and transport only."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.jobs import JobRecord, JobState
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.controlplane.LocalTransport import LocalTransport
from lambdaforge.controlplane.ResearchWork import aggregate_research_work
from lambdaforge.controlplane.StorageService import StorageService
from lambdaforge.work.models import atomic_json


@pytest.fixture
def recovery_jobs(tmp_path: Path) -> tuple[JobService, JobRecord, Path]:
    work = tmp_path / "job-owner" / "work"
    work.mkdir(parents=True)
    config = tmp_path / "study.yaml"
    config.write_text(
        "name: recovery\nrun: tests.work_cases.RecoverableStudyWork\n"
        "seeds: [1]\nsearch:\n  trials: 2\n  choice: {values: [0, 1]}\n"
        "objective: {metric: score, mode: max}\nresources: {cpu: 1}\n"
    )
    execution = work / ".lambdaforge" / "runs" / "recovery" / "execution-original"
    atomic_json(execution / "execution.json", {"execution_id": execution.name})
    atomic_json(
        execution / "hpo-control" / "state.json",
        {
            "execution_id": execution.name,
            "state_version": 5,
            "runs": [],
            "pending_actions": [{"seed": 1}],
            "proposed_pool_trials": [1],
        },
    )
    atomic_json(
        work.parent / "study" / "controller.json",
        {"initialization": {"control_state_path": str(execution / "hpo-control" / "state.json")}},
    )
    profile = ClusterProfile(
        "fake-remote",
        transport="ssh",
        host="never-connect.invalid",
        workspace=str(tmp_path / "storage"),
    )
    catalog = ClusterCatalog({"fake-remote": profile})
    factory = SimpleNamespace(
        transport=lambda profile: LocalTransport(),
        scheduler=lambda profile, transport: SimpleNamespace(),
    )
    jobs = JobService(catalog, JobStore(tmp_path / "registry"), factory)
    now = datetime.now(timezone.utc).isoformat()
    record = JobRecord(
        "job-owner",
        profile.name,
        "local",
        None,
        JobState.FAILED,
        (),
        str(work),
        {"cpu_cores": 1},
        now,
        now,
        config_path=str(config),
        metadata={
            "name": "recovery",
            "study_expected": True,
            "scientific_identity": "sha256:original",
            "submission_mode": "asynchronous",
            "run_arguments": [],
            "source_config_path": str(config),
        },
        job_type="work",
    )
    jobs.store.write(record)
    return jobs, record, execution


def test_retry_passes_exact_state_to_fresh_bundle_and_keeps_work_identity(
    recovery_jobs: tuple[JobService, JobRecord, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs, previous, execution = recovery_jobs
    launches: list[Any] = []
    actual_popen = subprocess.Popen

    def launch(command: Any, *args: Any, **kwargs: Any) -> Any:
        if "lambdaforge.controlplane.SubmissionWorker" in command:
            launches.append(command)
            return SimpleNamespace(pid=1234)
        return actual_popen(command, *args, **kwargs)

    monkeypatch.setattr("subprocess.Popen", launch)
    preview = jobs.retry_preview(previous.job_id)
    assert preview["resumable"] is True
    assert preview["execution_dir"] == str(execution)
    assert preview["pending_actions"] == 1
    handle = jobs.retry(previous.job_id, accept_code_change=True)
    record = jobs.get(handle.job_id, refresh=False)
    arguments = record.metadata["run_arguments"]
    assert arguments[arguments.index("--resume-execution") + 1] == str(execution)
    assert "--accept-code-change" in arguments
    assert record.retry_of == previous.job_id
    assert record.metadata["recovery_dependencies"] == [previous.job_id]
    staged = record.with_updates(work_dir=str(execution.parents[5] / "job-new" / "work"))
    old_log = execution / "runs" / "run-old" / "attempts" / "attempt-0001" / "work.log"
    assert str(jobs._owned_study_path(staged, str(old_log))) == str(old_log)
    with pytest.raises(RuntimeError, match="escaped"):
        jobs._owned_study_path(staged, "/other/secret.log")
    assert len(aggregate_research_work(jobs.store.records())) == 1
    assert launches
    with pytest.raises(Exception, match="active|already|duplicate"):
        jobs.retry(previous.job_id)
    assert len(launches) == 1


def test_recovery_owner_survives_gc_and_single_job_deletion(
    recovery_jobs: tuple[JobService, JobRecord, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs, previous, _execution = recovery_jobs
    actual_popen = subprocess.Popen

    def launch(command: Any, *args: Any, **kwargs: Any) -> Any:
        if "lambdaforge.controlplane.SubmissionWorker" in command:
            return SimpleNamespace(pid=1234)
        return actual_popen(command, *args, **kwargs)

    monkeypatch.setattr("subprocess.Popen", launch)
    handle = jobs.retry(previous.job_id)
    with pytest.raises(ValueError, match="retained"):
        jobs.delete(previous.job_id)
    storage = StorageService(jobs.catalog, jobs.factory, jobs=jobs)
    with pytest.raises(ValueError, match="retained"):
        storage.delete_job(previous.cluster, previous.job_id)
    captured: dict[str, Any] = {}

    def invoke(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs["references"])
        return {"candidates": [], "reclaimable_bytes": 0, "preserved": []}

    monkeypatch.setattr(storage, "_invoke", invoke)
    storage.gc(previous.cluster)
    assert previous.job_id not in captured["terminal_jobs"]
    assert jobs.recovery_dependents(previous.job_id) == (handle.job_id,)


def test_missing_state_is_not_silently_restarted(
    recovery_jobs: tuple[JobService, JobRecord, Path],
) -> None:
    jobs, previous, execution = recovery_jobs
    (execution / "hpo-control" / "state.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        jobs.retry_preview(previous.job_id)


def test_preview_fails_closed_for_foreign_execution(
    recovery_jobs: tuple[JobService, JobRecord, Path],
    tmp_path: Path,
) -> None:
    jobs, previous, _execution = recovery_jobs
    foreign = tmp_path / "other" / "execution-foreign"
    atomic_json(foreign / "execution.json", {"execution_id": foreign.name})
    atomic_json(foreign / "hpo-control" / "state.json", {"execution_id": foreign.name})
    atomic_json(Path(previous.work_dir).parent / "result.json", {"execution_dir": str(foreign)})
    with pytest.raises(ValueError, match="outside"):
        jobs.retry_preview(previous.job_id)
