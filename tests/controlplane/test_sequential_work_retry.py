"""Public recovery route for sequential Works shares ownership and duplicate guards."""

import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.jobs import JobRecord, JobState
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.controlplane.LocalTransport import LocalTransport
from lambdaforge.work import WorkRunner
from tests.work.test_sequential_recovery import config
from tests.work.test_single_work_recovery import snapshot


def test_public_preview_apply_duplicate_and_gc(tmp_path, monkeypatch):
    monkeypatch.setenv("LAMBDAFORGE_JOB_ID", "job-owner")
    work = tmp_path / "job-owner/work"
    work.mkdir(parents=True)
    current = config(work, fail_first=False)
    result = WorkRunner().run(current)
    profile = ClusterProfile(
        "fixture",
        transport="ssh",
        host="never-connect.invalid",
        workspace=str(tmp_path / "storage"),
    )
    factory = SimpleNamespace(
        transport=lambda _: LocalTransport(), scheduler=lambda *_: SimpleNamespace()
    )
    jobs = JobService(
        ClusterCatalog({profile.name: profile}), JobStore(tmp_path / "index"), factory
    )
    now = datetime.now(timezone.utc).isoformat()
    record = JobRecord(
        "job-owner",
        profile.name,
        "local",
        None,
        JobState.FAILED,
        (),
        str(work),
        current.resources.to_dict(),
        now,
        now,
        config_path=str(current.source),
        job_type="work",
        metadata={
            "source_config_path": str(current.source),
            "study_expected": False,
            "submission_mode": "asynchronous",
            "run_arguments": [],
            "name": "pipeline",
        },
    )
    jobs.store.write(record)
    original_scheduler = factory.scheduler
    factory.scheduler = lambda *_: SimpleNamespace(
        state=lambda _: JobState.RUNNING, details=lambda _: {"provider_observation": "running"}
    )
    jobs.store.write(record.with_updates(scheduler_id="live"))
    with pytest.raises(ValueError, match="may still own live"):
        jobs.retry_preview(record.job_id)
    factory.scheduler = original_scheduler
    jobs.store.write(record)
    before = snapshot(tmp_path)
    plan = jobs.retry(record.job_id, dry_run=True, config=current.source).recovery_plan
    assert plan["strategy"] == "sequential"
    assert (plan["reuse_runs"], plan["retry_runs"], plan["pending_runs"]) == (1, 1, 1)
    assert snapshot(tmp_path) == before
    actual = subprocess.Popen
    launches = []

    def popen(command, *args, **kwargs):
        if "lambdaforge.controlplane.SubmissionWorker" in command:
            launches.append(command)
            return SimpleNamespace(pid=1234)
        return actual(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", popen)

    def apply():
        try:
            return jobs.retry(record.job_id, accept_code_change=True).job_id
        except (ValueError, RuntimeError):
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: apply(), range(2)))
    assert sum(value is not None for value in outcomes) == 1
    assert len(launches) == 1
    recovered = jobs.store.get(next(value for value in outcomes if value))
    assert recovered.metadata["recovery_execution_dir"] == str(result.execution_dir)
    assert recovered.metadata["recovery_owner_job"] == record.job_id
    with pytest.raises(ValueError, match="retained"):
        jobs.delete(record.job_id)
