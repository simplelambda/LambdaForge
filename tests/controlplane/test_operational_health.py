"""Fault injection for operational observations, distinct from scientific evidence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lambdaforge.controlplane.jobs import JobRecord, JobState
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.controlplane.OperationalHealth import ObservationPolicy, OperationalHealth
from lambdaforge.controlplane.ProcessIdentity import ProcessIdentity
from lambdaforge.controlplane.ProcessSupervisor import ProcessSupervisor

NOW = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)


def record() -> JobRecord:
    return JobRecord(
        "job-health",
        "test",
        "local",
        "job-health",
        JobState.RUNNING,
        ("python",),
        "/owned/work",
        {},
        (NOW - timedelta(hours=1)).isoformat(),
        NOW.isoformat(),
        metadata={"scientific_identity": "original", "retained_evidence": {"runs": 9}},
    )


def observe(store: JobStore, seconds: int, **options: object) -> JobRecord:
    at = (NOW + timedelta(seconds=seconds)).isoformat()
    return store.record_observation(
        "job-health",
        started=at,
        finished=at,
        source="process-host",
        **options,
    )


def test_local_rewrite_and_legacy_state_never_manufacture_freshness() -> None:
    health = OperationalHealth.describe(record(), now=NOW)
    assert health["health"] == "unknown"
    assert "last_success_at_utc" not in health
    assert health["observation_age_seconds"] is None


def test_failed_probe_retains_confirmation_and_science_then_recovers(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    store.write(record())
    successful = observe(
        store, 0, state=JobState.RUNNING, details={"heartbeat_at_utc": NOW.isoformat()}
    )
    failed = observe(store, 1, error=TimeoutError("password=private"))
    health = OperationalHealth.describe(failed, now=NOW + timedelta(seconds=1))
    assert health["health"] == "degraded"
    assert health["execution_state"] == "running"
    assert health["last_success_at_utc"] == NOW.isoformat()
    assert "private" not in json.dumps(failed.to_dict())
    failed = observe(store, 2, error=ConnectionError("offline"))
    assert (
        OperationalHealth.describe(failed, now=NOW + timedelta(seconds=2))["health"]
        == "unreachable"
    )
    assert failed.metadata["retained_evidence"] == successful.metadata["retained_evidence"]
    recovered = observe(
        store, 3, state=JobState.RUNNING, details={"heartbeat_at_utc": NOW.isoformat()}
    )
    assert (
        OperationalHealth.describe(recovered, now=NOW + timedelta(seconds=3))["health"] == "healthy"
    )


def test_late_observation_cannot_erase_new_fact_or_cancel(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    store.write(record())
    newer = observe(store, 2, error=TimeoutError("unreachable"))
    late = observe(store, 1, state=JobState.RUNNING)
    assert late == newer
    store.write(newer.with_updates(state=JobState.CANCELLED))
    observed = observe(store, 3, state=JobState.RUNNING)
    assert observed.state is JobState.CANCELLED
    assert observed.metadata["retained_evidence"] == {"runs": 9}


def test_repeated_unclassified_query_error_does_not_invent_a_network_outage(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    store.write(record())
    observe(store, 0, state=JobState.RUNNING)
    for seconds in (1, 2, 3):
        current = observe(store, seconds, error=RuntimeError("Query parser failed"))
        health = OperationalHealth.describe(current, now=NOW + timedelta(seconds=seconds))
        assert health["health"] == "degraded"
        assert health["connectivity"] == "unknown"
        assert health["last_confirmed_state"] == "running"
        assert health["diagnostic"]["category"] == "warning"


def test_authentication_failure_is_distinct_from_science_and_unknown_query_errors(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path)
    store.write(record())
    error = type("AuthenticationException", (Exception,), {})("Authentication rejected")
    observe(store, 0, state=JobState.RUNNING)
    current = observe(store, 1, error=error)
    health = OperationalHealth.describe(current, now=NOW + timedelta(seconds=1))
    assert health["diagnostic"]["category"] == "authentication"
    assert health["connectivity"] == "unreachable"
    assert current.state == JobState.UNKNOWN  # uncertainty, not a scientific failure
    assert health["last_confirmed_state"] == "running"
    assert current.metadata["retained_evidence"] == {"runs": 9}


def test_provider_clock_and_heartbeat_are_distinct_from_progress_expectations(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path)
    store.write(record())
    current = observe(
        store, 0, state=JobState.RUNNING, details={"heartbeat_at_utc": NOW.isoformat()}
    )
    later = NOW + timedelta(seconds=60)
    assert OperationalHealth.describe(current, now=later)["reason"] == "HEARTBEAT_STALE"
    assert (
        OperationalHealth.describe(current, now=NOW + timedelta(seconds=181))["health"] == "stale"
    )
    assert (
        OperationalHealth.describe(current, now=NOW - timedelta(seconds=30))["freshness"]
        == "clock_skew"
    )
    current = observe(
        store, 1, state=JobState.RUNNING, details={"heartbeat_at_utc": later.isoformat()}
    )
    assert (
        OperationalHealth.describe(
            current, now=later, policy=ObservationPolicy(progress_seconds=30)
        )["reason"]
        == "SCIENTIFIC_PROGRESS_STALLED"
    )
    assert OperationalHealth.describe(current, now=later)["health"] == "healthy"


def test_policy_rejects_fractional_failure_counts() -> None:
    with pytest.raises(ValueError):
        ObservationPolicy(unreachable_failures=1.5)
    for field in ("freshness_seconds", "heartbeat_seconds", "grace_seconds"):
        with pytest.raises(ValueError, match="finite and positive"):
            ObservationPolicy.from_mapping({field: None})


def test_legacy_terminal_file_is_visible_but_never_grants_owned_recovery(tmp_path: Path) -> None:
    directory = tmp_path / "job-legacy"
    directory.mkdir()
    (directory / "state.json").write_text(json.dumps({"job_id": directory.name, "state": "failed"}))
    observed = ProcessSupervisor.observe(directory, markers=({}, True))
    assert observed["state"] == "failed"
    assert observed["provider_observation"] == "unknown"
    assert observed["observation_reason"] == "LEGACY_TERMINAL_OWNERSHIP_UNVERIFIED"
    before = (directory / "state.json").read_bytes()
    assert (directory / "state.json").read_bytes() == before
    (directory / "state.json").write_text(
        json.dumps({"job_id": directory.name, "state": "running"})
    )
    with pytest.raises(ValueError, match="ownership request is missing"):
        ProcessSupervisor.observe(directory, markers=({}, True))
    assert not (directory / "request.json").exists()


def test_degraded_liveness_does_not_create_scientific_confirmation(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    store.write(record())
    current = observe(
        store,
        0,
        state=JobState.RUNNING,
        details={
            "provider_observation": "degraded",
            "observation_reason": "SCIENTIFIC_CHILD_UNVERIFIED",
        },
    )
    health = OperationalHealth.describe(current, now=NOW)
    assert health["health"] == "degraded"
    assert health["reason"] == "SCIENTIFIC_CHILD_UNVERIFIED"
    assert health.get("last_confirmed_at_utc") is None


def host_record(directory: Path, identity: ProcessIdentity | None) -> None:
    directory.mkdir()
    (directory / "request.json").write_text(json.dumps({"job_id": directory.name}))
    (directory / "state.json").write_text(
        json.dumps(
            {
                "job_id": directory.name,
                "state": "running",
                "heartbeat_at_utc": NOW.isoformat(),
                "supervisor_identity": identity.to_dict() if identity else None,
                "process_identity": identity.to_dict() if identity else None,
            }
        )
    )


@pytest.mark.skipif(os.name != "posix", reason="POSIX process identity integration")
def test_real_host_lifetime_lost_state_file_never_means_running(tmp_path: Path) -> None:
    child = subprocess.Popen(
        (sys.executable, "-c", "import time; time.sleep(30)"), start_new_session=True
    )
    try:
        directory = tmp_path / "job-health"
        identity = ProcessIdentity.create(child.pid, child.pid, (sys.executable,), directory.name)
        host_record(directory, identity)
        assert ProcessSupervisor.observe(directory)["provider_observation"] == "alive"
        child.terminate()
        child.wait(timeout=5)
        observed = ProcessSupervisor.observe(directory, markers=({}, True))
        assert observed["provider_observation"] == "lost_confirmed"
        assert observed["state"] == "unknown"
        # Observation is read-only; original process failure evidence stays byte-for-byte.
        assert json.loads((directory / "state.json").read_text())["state"] == "running"
        assert (
            ProcessSupervisor.observe(directory, markers=({}, False))["provider_observation"]
            == "unknown"
        )
        assert (
            ProcessSupervisor.observe(directory, markers=({directory.name: [123]}, True))[
                "provider_observation"
            ]
            == "unknown"
        )
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_legacy_and_corrupt_and_unsafe_process_records_are_not_healthy(tmp_path: Path) -> None:
    directory = tmp_path / "job-health"
    host_record(directory, None)
    assert (
        ProcessSupervisor.observe(directory, markers=({}, True))["provider_observation"]
        == "unknown"
    )
    (directory / "state.json").write_text("invalid")
    with pytest.raises(TypeError):
        ProcessSupervisor.observe(directory)
    (directory / "state.json").unlink()
    (directory / "state.json").symlink_to(directory / "request.json")
    with pytest.raises(ValueError):
        ProcessSupervisor.observe(directory)
