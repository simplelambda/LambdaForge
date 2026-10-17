"""An unreadable historical Job must not poison its execution host's inventory."""

from __future__ import annotations

import inspect
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.jobs import JobRecord, JobState
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.controlplane.LocalTransport import LocalTransport
from lambdaforge.controlplane.OverviewService import OverviewService
from lambdaforge.controlplane.ProcessIdentity import ProcessIdentity
from lambdaforge.controlplane.ProcessScheduler import ProcessScheduler
from lambdaforge.controlplane.ProcessSupervisor import ProcessSupervisor


def host_job(root: Path, job_id: str, state: JobState = JobState.RUNNING) -> Path:
    """Write disposable native host evidence, never launch a scientific process."""
    directory = root / job_id
    directory.mkdir(parents=True)
    now = datetime.now(timezone.utc).isoformat()
    request = {"job_id": job_id, "command": [sys.executable], "resources": {}}
    payload: dict[str, Any] = {
        "job_id": job_id,
        "state": state.value,
        "heartbeat_at_utc": now,
    }
    if not state.terminal:
        identity = ProcessIdentity.create(
            os.getpid(),
            os.getpgrp() if os.name == "posix" else os.getpid(),
            [sys.executable],
            job_id,
        )
        payload.update(supervisor_identity=identity.to_dict(), process_identity=identity.to_dict())
    (directory / "request.json").write_text(json.dumps(request), encoding="utf-8")
    (directory / "state.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory


def service(tmp_path: Path) -> tuple[JobService, Path]:
    root = tmp_path / "host" / "jobs"
    profile = ClusterProfile(
        "fake",
        transport="local",
        python=sys.executable,
        workspace=str(tmp_path / "host"),
        storage={"run_root": str(root)},
    )
    return JobService(ClusterCatalog({"fake": profile}), JobStore(tmp_path / "registry")), root


def remember(jobs: JobService, directory: Path, state: JobState = JobState.UNKNOWN) -> JobRecord:
    now = datetime.now(timezone.utc).isoformat()
    record = JobRecord(
        directory.name,
        "fake",
        "local",
        directory.name,
        state,
        (),
        str(directory / "work"),
        {},
        now,
        now,
    )
    jobs.store.write(record)
    return record


@pytest.mark.parametrize(
    "fault", ["malformed_state", "empty_state", "missing_request", "foreign_request", "symlink"]
)
def test_inventory_isolates_corruption_and_never_adopts_error_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    if fault == "symlink" and os.name != "posix":
        pytest.skip("Symlink creation requires additional Windows privileges.")
    jobs, root = service(tmp_path)
    good = host_job(root, "job-good")
    finished = host_job(root, "job-finished", JobState.SUCCEEDED)
    bad = host_job(root, "job-bad")
    untracked = host_job(root, "job-untracked")
    (untracked / "state.json").write_text("not JSON", encoding="utf-8")
    if fault == "malformed_state":
        (bad / "state.json").write_text("{unfinished", encoding="utf-8")
    elif fault == "empty_state":
        (bad / "state.json").write_text("{}", encoding="utf-8")
    elif fault == "missing_request":
        (bad / "request.json").unlink()
    elif fault == "foreign_request":
        (bad / "request.json").write_text('{"job_id":"job-other"}', encoding="utf-8")
    else:
        (bad / "request.json").unlink()
        (bad / "request.json").symlink_to(good / "request.json")
    originals = {path: path.read_bytes() for path in root.rglob("*.json")}
    for directory in (good, finished, bad):
        remember(jobs, directory)
    monkeypatch.setattr(ProcessSupervisor, "_process_markers", lambda: ({}, []))
    rows = ProcessSupervisor.inventory(root)
    assert len(rows) == 4
    errors = {row["request"]["job_id"] for row in rows if "observation_error" in row}
    assert errors == {bad.name, untracked.name}
    monkeypatch.setattr(jobs, "_inventory", lambda name: rows)
    values = {record.job_id: record for record in OverviewService._jobs(jobs)}
    assert values[good.name].state is JobState.RUNNING
    assert values[good.name].to_dict()["operational"]["health"] == "healthy"
    assert values[finished.name].state is JobState.SUCCEEDED
    assert values[finished.name].to_dict()["operational"]["health"] == "healthy"
    assert values[bad.name].state is JobState.UNKNOWN
    health = values[bad.name].to_dict()["operational"]
    assert health["health"] == "degraded"
    assert health["reason"] == "INVALID_PROVIDER_OBSERVATION"
    assert bad.name in health["last_error"]["message"]
    assert untracked.name not in values  # Failed evidence cannot create a native Job.
    assert all(path.read_bytes() == content for path, content in originals.items())


def test_shipped_observer_and_native_service_clear_prior_cluster_error(tmp_path: Path) -> None:
    jobs, root = service(tmp_path)
    good = host_job(root, "job-current")
    bad = host_job(root, "job-old")
    (bad / "state.json").write_text("broken historical record", encoding="utf-8")
    for directory in (good, bad):
        record = remember(jobs, directory, JobState.RUNNING)
        jobs.observation_failed(record.job_id, ValueError("old cluster-wide inventory failure"))
    profile = jobs.catalog.get("fake")
    scheduler = ProcessScheduler(LocalTransport(), profile)
    rows = scheduler.inventory()  # Executes the shipped observer as a separate interpreter.
    assert len(rows) == 2
    assert any(row.get("observation_error") for row in rows)
    jobs.reconcile(all_clusters=True)
    current = jobs.store.get(good.name)
    assert current.state is JobState.RUNNING
    assert current.to_dict()["operational"]["health"] == "healthy"
    raw = current.metadata["operational_observation"]
    assert raw["last_error"] is None
    assert raw["consecutive_failures"] == 0
    old = jobs.store.get(bad.name)
    assert old.state is JobState.UNKNOWN
    assert old.to_dict()["operational"]["health"] == "degraded"
    assert "job-old" in old.to_dict()["operational"]["last_error"]["message"]
    with pytest.raises(RuntimeError, match="Could not read durable state"):
        scheduler.state(bad.name)  # Exact corrupted Job reads must still fail closed.


def test_bad_terminal_record_retains_outcome_and_healthy_jobs_keep_refreshing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs, root = service(tmp_path)
    good = host_job(root, "job-running")
    bad = host_job(root, "job-failed", JobState.FAILED)
    remember(jobs, good)
    remember(jobs, bad, JobState.FAILED)
    valid = (bad / "state.json").read_bytes()
    (bad / "state.json").write_text("corrupt", encoding="utf-8")
    monkeypatch.setattr(ProcessSupervisor, "_process_markers", lambda: ({}, []))
    monkeypatch.setattr(jobs, "_inventory", lambda name: ProcessSupervisor.inventory(root))
    jobs.reconcile(all_clusters=True)
    terminal = jobs.store.get(bad.name)
    assert terminal.state is JobState.FAILED
    assert terminal.to_dict()["operational"]["health"] == "degraded"
    first = jobs.store.get(good.name).metadata["operational_observation"]
    jobs.reconcile(all_clusters=True)
    second = jobs.store.get(good.name).metadata["operational_observation"]
    assert second["last_confirmed_at_utc"] > first["last_confirmed_at_utc"]
    # Repair occurs only in the disposable fixture. Observation must never repair evidence.
    (bad / "state.json").write_bytes(valid)
    jobs.reconcile(all_clusters=True)
    repaired = jobs.store.get(bad.name)
    assert repaired.state is JobState.FAILED
    assert repaired.to_dict()["operational"]["health"] == "healthy"
    assert repaired.metadata["operational_observation"]["last_error"] is None


@pytest.mark.parametrize("error", [None, {}, {"type": "ValueError", "message": []}])
def test_malformed_error_envelopes_are_not_a_successful_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Any
) -> None:
    jobs, _ = service(tmp_path)
    payload = (
        {
            "request": {"job_id": "job-bad"},
            "state": {"job_id": "job-bad", "state": "unknown"},
            "observation_error": error,
        },
    )
    scheduler = type("Scheduler", (), {"inventory": lambda self: payload})()
    monkeypatch.setattr(jobs.factory, "scheduler", lambda *args: scheduler)
    with pytest.raises(ValueError, match="malformed per-Job error"):
        jobs._inventory("fake")
    assert jobs.store.records() == ()


def test_non_job_directory_names_do_not_poison_native_inventory(tmp_path: Path) -> None:
    jobs, root = service(tmp_path)
    good = host_job(root, "job-native")
    for name in ("job-UPPER", "job-odd name", "job-record.tmp"):
        host_job(root, name)
    remember(jobs, good)
    rows = jobs._inventory("fake")
    assert [row["request"]["job_id"] for row in rows] == [good.name]
    jobs.reconcile(all_clusters=True)
    assert jobs.store.get(good.name).state is JobState.RUNNING
    assert len(jobs.store.records()) == 1


def test_observer_does_not_use_loaded_method_line_offsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate stale inspect offsets without editing production files or loaded code."""
    jobs, root = service(tmp_path)
    good = host_job(root, "job-hot-update")
    (good / "study").mkdir()
    (good / "study" / "overview.json").write_text(
        json.dumps({"counts": {"candidates": 2}}), encoding="utf-8"
    )
    remember(jobs, good)
    real_source = inspect.getsource

    def stale_source(value: Any) -> str:
        if getattr(value, "__qualname__", "").startswith("ProcessSupervisor."):
            # Old co_firstlineno can point at a different method after a source update.
            return "    @staticmethod\n    def wrong_method():\n        return None\n"
        if getattr(value, "__name__", "") == "study_overview":
            return "def wrong_overview():\n    return None\n"
        return real_source(value)

    monkeypatch.setattr(inspect, "getsource", stale_source)
    scheduler = ProcessScheduler(LocalTransport(), jobs.catalog.get("fake"))
    assert scheduler.state(good.name) is JobState.RUNNING
    rows = scheduler.inventory()
    assert len(rows) == 1
    assert rows[0]["state"]["provider_observation"] == "alive"
    assert rows[0]["state"]["study"]["counts"]["candidates"] == 2
