"""Recovery handoff and storage protection using local files and transport only."""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
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
from lambdaforge.work import WorkConfig, WorkRunner
from lambdaforge.work.models import atomic_json
from lambdaforge.work.runner import _execute_run


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


def test_automatic_sweep_preview_works_without_an_installed_remote_runtime(
    recovery_jobs: tuple[JobService, JobRecord, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs, previous, _execution = recovery_jobs
    config = WorkConfig.from_mapping(
        {
            "name": "recovery",
            "run": "tests.work_cases.FixedRecoveryWork",
            "with": {"failures": 0},
            "sweep": {"space": {"choice": [0, 1]}},
            "resources": {"cpu": 1},
            "execution": {"max_runs": 4},
            "objective": {"metric": "score", "mode": "max", "range": [0, 1]},
        },
        source=Path(str(previous.config_path)),
    )

    def dispatch(values: Any, **kwargs: Any) -> tuple[Any, ...]:
        queue = list(values)
        results = []
        while queue:
            result = _execute_run(queue.pop(0))
            results.append(result)
            queue = list(kwargs["on_result"](result, queue, ()))
        return tuple(results)

    result = WorkRunner(
        dispatcher=dispatch,
        execution_root=Path(previous.work_dir) / ".lambdaforge" / "runs" / "recovery",
    ).run(config)
    atomic_json(Path(previous.work_dir).parent / "result.json", result.to_dict())

    class IsolatedTransport:
        def run(self, command: Any, **kwargs: Any) -> Any:
            # No site packages, PYTHONPATH, installed LambdaForge or real SSH provider.
            return subprocess.run(
                [sys.executable, "-I", "-S", *command[1:]],
                capture_output=True,
                text=True,
                timeout=kwargs["timeout"],
                check=False,
            )

    monkeypatch.setattr(jobs.factory, "transport", lambda profile: IsolatedTransport())
    before = {path: path.read_bytes() for path in result.execution_dir.rglob("*") if path.is_file()}
    preview = jobs.retry_preview(previous.job_id)
    assert preview["reuse_runs"] == 4
    assert preview["retry_runs"] == preview["pending_runs"] == 0
    assert before == {
        path: path.read_bytes() for path in result.execution_dir.rglob("*") if path.is_file()
    }


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


def test_fixed_preview_and_retry_handoff_preserve_nine_runs_without_adaptive_state(
    recovery_jobs: tuple[JobService, JobRecord, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs, previous, _execution = recovery_jobs
    source = Path(str(previous.config_path))
    source.write_text(
        "name: recovery\nrun: tests.work_cases.FixedRecoveryWork\n"
        "seeds: [4, 7, 32, 54, 65, 94, 109, 124, 142, 167]\n"
        "objective: {metric: score, mode: max}\nresources: {cpu: 1}\n"
        "execution: {failure_retries: 0}\n"
    )
    config = WorkConfig.from_yaml(source)

    def dispatch(values: Any, **kwargs: Any) -> tuple[Any, ...]:
        return tuple(_execute_run(value) for value in values)

    result = WorkRunner(
        dispatcher=dispatch,
        execution_root=Path(previous.work_dir) / ".lambdaforge" / "runs" / "recovery",
    ).run(config)
    atomic_json(Path(previous.work_dir).parent / "result.json", result.to_dict())
    before = {path: path.read_bytes() for path in result.execution_dir.rglob("*") if path.is_file()}
    preview = jobs.retry_preview(previous.job_id)
    assert preview["strategy"] == "repeated"
    assert (preview["reuse_runs"], preview["retry_runs"], preview["pending_runs"]) == (9, 1, 0)
    assert not (result.execution_dir / "hpo-control" / "state.json").exists()
    registry = {path: path.read_bytes() for path in jobs.store.root.rglob("*") if path.is_file()}
    planned = jobs.retry(previous.job_id, dry_run=True)
    assert planned.state == JobState.PLANNED
    assert planned.recovery_plan["reuse_runs"] == 9
    assert registry == {
        path: path.read_bytes() for path in jobs.store.root.rglob("*") if path.is_file()
    }
    launches: list[Any] = []
    actual_popen = subprocess.Popen

    def launch(command: Any, *args: Any, **kwargs: Any) -> Any:
        if "lambdaforge.controlplane.SubmissionWorker" in command:
            launches.append(command)
            return SimpleNamespace(pid=1234)
        return actual_popen(command, *args, **kwargs)

    monkeypatch.setattr("subprocess.Popen", launch)
    handles, errors = [], []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(jobs.retry, previous.job_id) for _ in range(2)]
        for future in futures:
            try:
                handles.append(future.result())
            except Exception as error:
                errors.append(str(error))
    assert len(handles) == 1 and len(errors) == 1
    assert any(word in errors[0] for word in ("active", "already", "duplicate"))
    handle = handles[0]
    record = jobs.get(handle.job_id, refresh=False)
    assert record.metadata["recovery_execution_dir"] == str(result.execution_dir)
    assert record.metadata["recovery_owner_job"] == previous.job_id
    assert "--resume-execution" in record.metadata["run_arguments"]
    with pytest.raises(Exception, match="active|already|duplicate"):
        jobs.retry(previous.job_id)
    assert len(launches) == 1
    assert before == {
        path: path.read_bytes() for path in result.execution_dir.rglob("*") if path.is_file()
    }
