"""Public recovery preview/apply routes preserve single Work ownership and resource review."""

from __future__ import annotations

import json
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
from lambdaforge.controlplane.StorageService import StorageService
from lambdaforge.work import WorkRunner
from tests.work.test_single_work_recovery import config, snapshot


@pytest.fixture
def single_job(tmp_path, monkeypatch):
    monkeypatch.setenv("LAMBDAFORGE_JOB_ID", "job-single-owner")
    work = tmp_path / "job-single-owner/work"
    first_config = config(work)
    first = WorkRunner().run(first_config)
    # Source recipe is renamed, with more operational memory but identical physics/inputs.
    selected = config(work, filename="enrich.yaml", ram="96GiB")
    profile = ClusterProfile(
        "fixture-remote",
        transport="ssh",
        host="never-connect.invalid",
        workspace=str(tmp_path / "storage"),
    )
    catalog = ClusterCatalog({profile.name: profile})
    factory = SimpleNamespace(
        transport=lambda _: LocalTransport(), scheduler=lambda *_: SimpleNamespace()
    )
    jobs = JobService(catalog, JobStore(tmp_path / "index"), factory)
    now = datetime.now(timezone.utc).isoformat()
    record = JobRecord(
        "job-single-owner",
        profile.name,
        "local",
        None,
        JobState.FAILED,
        (),
        str(work),
        first_config.resources.to_dict(),
        now,
        now,
        config_path=str(first_config.source),
        job_type="work",
        metadata={
            "source_config_path": str(selected.source),
            "study_expected": False,
            "submission_mode": "asynchronous",
            "run_arguments": [],
            "name": "enrich",
            "scientific_identity": first.scientific_fingerprint,
        },
    )
    jobs.store.write(record)
    return jobs, record, first, selected


def test_single_metadata_only_retry_dry_run_does_not_solve_environment_or_mutate(
    single_job, monkeypatch
):
    jobs, record, first, selected = single_job

    def forbidden(*args, **kwargs):
        raise AssertionError("preview must not prepare/solve an exact native environment")

    from lambdaforge.controlplane.ControlPlane import ControlPlane

    monkeypatch.setattr(ControlPlane, "submit", forbidden)
    before = snapshot(first.execution_dir)
    index_before = snapshot(jobs.store.root)
    handle = jobs.retry(record.job_id, dry_run=True, config=selected.source)
    plan = handle.recovery_plan
    assert plan["strategy"] == "single"
    assert plan["owner_job_id"] == record.job_id
    assert plan["runs"][0]["next_attempt"] == 2
    assert plan["effective_resources"]["ram_bytes"] == 96 * 1024**3
    assert plan["resource_changes"]["ram_bytes"]["previous"] == 16 * 1024**3
    assert "--resume-study-path" not in handle.preview["run_arguments"]
    assert before == snapshot(first.execution_dir)
    assert index_before == snapshot(jobs.store.root)


def test_single_retry_apply_pins_owner_config_and_effective_resources_and_gc(
    single_job, monkeypatch
):
    jobs, record, first, selected = single_job
    actual_popen = subprocess.Popen
    launches = []

    def popen(command, *args, **kwargs):
        if "lambdaforge.controlplane.SubmissionWorker" in command:
            launches.append(command)
            return SimpleNamespace(pid=1234)
        return actual_popen(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", popen)
    preview = jobs.retry_preview(record.job_id, config=selected.source)
    handle = jobs.retry(
        record.job_id,
        config=selected.source,
        accept_code_change=True,
        config_sha256=preview["configuration"]["sha256"],
    )
    new = jobs.get(handle.job_id, refresh=False)
    assert new.resources["ram_bytes"] == 96 * 1024**3
    assert new.metadata["study_expected"] is False
    assert new.metadata["recovery_owner_job"] == record.job_id
    assert new.metadata["recovery_execution_dir"] == str(first.execution_dir)
    assert new.metadata["recovery_dependencies"] == [record.job_id]
    assert "--resume-execution" in new.metadata["run_arguments"]
    assert "--resume-study-path" not in new.metadata["run_arguments"]
    assert launches
    with pytest.raises(ValueError, match="retained"):
        jobs.delete(record.job_id)
    with pytest.raises(ValueError, match="retained"):
        StorageService(jobs.catalog, jobs.factory, jobs=jobs).delete_job(
            record.cluster, record.job_id
        )
    with pytest.raises((ValueError, RuntimeError), match="active|already|duplicate"):
        jobs.retry(record.job_id, accept_code_change=True)
    assert len(launches) == 1


def test_single_retry_code_ack_and_reviewed_config_hash(single_job, monkeypatch):
    jobs, record, first, selected = single_job
    from lambdaforge.reproducibility.CodeIdentity import CodeIdentity

    monkeypatch.setattr(
        CodeIdentity,
        "capture",
        lambda *_: SimpleNamespace(to_dict=lambda: {"revision": "changed"}),
    )
    with pytest.raises(ValueError, match="code changed"):
        jobs.retry(record.job_id)
    preview = jobs.retry_preview(record.job_id)
    selected.source.write_text(selected.source.read_text() + "\n# changed after review\n")
    with pytest.raises(ValueError, match="changed after recovery preview"):
        jobs.retry(
            record.job_id, accept_code_change=True, config_sha256=preview["configuration"]["sha256"]
        )


def test_single_retry_refuses_live_provider_even_when_record_says_failed(single_job, monkeypatch):
    jobs, record, *_ = single_job
    jobs.store.write(record.with_updates(scheduler_id="live"))
    scheduler = SimpleNamespace(
        state=lambda _: JobState.RUNNING, details=lambda _: {"provider_observation": "running"}
    )
    monkeypatch.setattr(jobs.factory, "scheduler", lambda *_: scheduler)
    with pytest.raises(ValueError, match="may still own live"):
        jobs.retry_preview(record.job_id)


def test_single_concurrent_recovery_accepts_exactly_one_request(single_job, monkeypatch):
    jobs, record, *_ = single_job
    actual = subprocess.Popen

    def popen(command, *args, **kwargs):
        if "lambdaforge.controlplane.SubmissionWorker" in command:
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
    assert len(jobs.recovery_dependents(record.job_id)) == 1


def test_unknown_missing_single_evidence_cannot_fall_back_to_fresh_execution(single_job):
    jobs, record, first, _ = single_job
    manifest = first.execution_dir / "execution.json"
    value = json.loads(manifest.read_text())
    value.pop("resolved_configuration")
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="recovery state is missing"):
        jobs.retry(record.job_id, dry_run=True)


def test_synchronous_native_work_uses_same_checkpoint_recovery_path(single_job):
    jobs, record, *_ = single_job
    jobs.store.write(
        record.with_updates(metadata={**record.metadata, "submission_mode": "synchronous"})
    )
    handle = jobs.retry(record.job_id, dry_run=True)
    assert handle.recovery_plan["strategy"] == "single"
    assert handle.preview["operation"] == "work-recovery"
