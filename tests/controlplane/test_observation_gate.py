"""Concurrent monitor/Console probes retain native clocks and avoid duplicate host scans."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.jobs import JobState
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.controlplane.ObservationGate import ObservationGate
from tests.controlplane.test_operational_health import record
from tests.controlplane.test_operational_providers import job_service


def test_concurrent_inventory_reuses_last_good_without_advancing_clocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs, original = job_service(tmp_path)
    stored = jobs.store.get(original.job_id)
    before = stored.to_dict()["operational"]
    started, release = Event(), Event()
    calls = []

    def inventory(name: str):
        calls.append(name)
        started.set()
        assert release.wait(5)
        return (
            {
                "request": {"job_id": original.job_id},
                "state": {
                    "job_id": original.job_id,
                    "state": "running",
                    "provider_observation": "alive",
                    "heartbeat_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            },
        )

    monkeypatch.setattr(jobs, "_inventory", inventory)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            first = pool.submit(jobs.reconcile, cluster="fake")
            assert started.wait(5)
            others = [pool.submit(jobs.reconcile, cluster="fake") for _ in range(3)]
            for future in others:
                snapshot = future.result(timeout=2)[0]
                current = snapshot.to_dict()["operational"]
                for key in before:
                    if key.endswith("_age_seconds") and before[key] is not None:
                        assert current[key] >= before[key]
                    else:
                        assert current[key] == before[key]
                assert snapshot.metadata == stored.metadata
            release.set()
            assert first.result(timeout=5)[0].to_dict()["operational"]["health"] == "healthy"
    finally:
        release.set()
    assert calls == ["fake"]


def test_independent_service_instances_do_not_duplicate_a_cluster_probe(tmp_path: Path) -> None:
    jobs, _ = job_service(tmp_path)
    other = JobService(jobs.catalog, JobStore(jobs.store.root))
    before = other.store.records()
    with ObservationGate(jobs.store.root, "fake"):
        assert other.reconcile(cluster="fake") == before
    assert other.store.records() == before


def test_gate_receipt_does_not_claim_provider_freshness_or_suppress_distinct_jobs(
    tmp_path: Path,
) -> None:
    first = ObservationGate(tmp_path, "cluster", scope="job-a")
    overlapping = ObservationGate(tmp_path, "cluster", scope="job-a")
    distinct = ObservationGate(tmp_path, "cluster", scope="job-b")
    with first as execute:
        assert execute
    with overlapping as execute:
        assert not execute
    with distinct as execute:
        assert execute
    assert "last_confirmed_at_utc" not in json.loads(first.path.read_text())


def test_failed_gate_never_publishes_receipt_and_can_be_retried(tmp_path: Path) -> None:
    gate = ObservationGate(tmp_path, "cluster")
    with pytest.raises(RuntimeError):
        with gate as execute:
            assert execute
            raise RuntimeError("interrupted before observations commit")
    assert not gate.path.exists()
    with ObservationGate(tmp_path, "cluster") as execute:
        assert execute


def test_corrupt_gate_receipt_is_not_empty_success_and_releases_lock(tmp_path: Path) -> None:
    gate = ObservationGate(tmp_path, "host")
    gate.path.parent.mkdir(parents=True)
    gate.path.write_text("{incomplete")
    with pytest.raises(ValueError):
        with gate:
            pytest.fail("Corrupt receipt cannot become successful coalescing evidence")
    assert not gate.lock.acquired
    assert gate.path.read_text() == "{incomplete"


def test_symlinked_gate_ancestor_is_refused_before_creating_external_files(tmp_path: Path) -> None:
    owned = tmp_path / "owned"
    outside = tmp_path / "outside"
    owned.mkdir()
    outside.mkdir()
    (owned / ".operational").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Unsafe provider observation gate"):
        with ObservationGate(owned, "host"):
            pytest.fail("Linked ancestor cannot acquire a provider gate")
    assert list(outside.iterdir()) == []


def test_slurm_jobs_keep_independent_observation_lanes(tmp_path: Path) -> None:
    started = {"job-a": Event(), "job-b": Event()}
    release = Event()

    def state(key: str):
        started[key].set()
        assert release.wait(5)
        return JobState.QUEUED

    scheduler = SimpleNamespace(state=state, details=lambda key: {"source": "squeue"})
    profile = ClusterProfile("slurm-host", scheduler="slurm", workspace=str(tmp_path))
    factory = SimpleNamespace(transport=lambda value: None, scheduler=lambda *args: scheduler)
    jobs = JobService(
        ClusterCatalog({profile.name: profile}), JobStore(tmp_path / "registry"), factory
    )
    for key in started:
        jobs.store.write(
            record().with_updates(
                job_id=key,
                scheduler_id=key,
                scheduler="slurm",
                cluster=profile.name,
                work_dir=str(tmp_path / key / "work"),
            )
        )
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(jobs.get, key, include_study=False) for key in started]
            assert all(event.wait(5) for event in started.values())
            release.set()
            assert all(future.result().state is JobState.QUEUED for future in futures)
    finally:
        release.set()


@pytest.mark.skipif(os.name != "posix", reason="Abrupt local owner death integration")
def test_abrupt_observer_death_releases_gate_without_poisoning_next_probe(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-c",
        "import sys,time; from pathlib import Path; "
        "from lambdaforge.controlplane.ObservationGate import ObservationGate; "
        "gate=ObservationGate(Path(sys.argv[1]), 'host'); gate.__enter__(); "
        "print('locked',flush=True); time.sleep(30)",
        str(tmp_path),
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "locked"
        with ObservationGate(tmp_path, "host") as execute:
            assert not execute
        process.kill()
        process.wait(timeout=5)
        with ObservationGate(tmp_path, "host") as execute:
            assert execute
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
