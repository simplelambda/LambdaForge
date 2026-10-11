"""Typed provider facts never become guessed scientific failures or retry grants."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lambdaforge.controlplane.jobs import JobState
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.controlplane.OperationalHealth import OperationalHealth
from lambdaforge.controlplane.OperationalIncidents import OperationalIncidents
from lambdaforge.diagnostics.failure import operational_diagnostic
from tests.controlplane.test_operational_health import NOW, observe, record


@pytest.mark.parametrize(
    ("raw", "category", "reason"),
    [
        ("NODE_FAIL", "execution", "scheduler_node_failure"),
        ("OUT_OF_MEMORY", "resource", "scheduler_memory_limit"),
        ("PREEMPTED", "execution", "scheduler_preemption"),
        ("BOOT_FAIL", "environment", "scheduler_boot_failure"),
        ("TIMEOUT", "resource", "scheduler_time_limit"),
        ("CANCELLED by 10", "cancelled", "scheduler_cancellation"),
    ],
)
def test_scheduler_diagnosis_retains_native_category_and_evidence(
    tmp_path: Path,
    raw: str,
    category: str,
    reason: str,
) -> None:
    store = JobStore(tmp_path)
    store.write(record())
    current = observe(
        store,
        0,
        state=JobState.FAILED,
        details={"source": "sacct", "raw_state": raw, "exit_code": "0:9"},
    )
    diagnostic = OperationalHealth.describe(current, now=NOW)["diagnostic"]
    assert diagnostic["category"] == category
    assert diagnostic["failure_disposition"]["reason"] == reason
    assert diagnostic["context"]["source"] == "sacct"
    assert diagnostic["context"]["confirmation"] == "verified"
    assert diagnostic["context"]["event_time_utc"] == NOW.isoformat()
    assert not diagnostic["failure_disposition"]["automatic_recovery_eligible"]
    assert diagnostic["fixes"] and diagnostic["commands"]


def test_signal_or_generic_cuda_error_is_not_inferred_oom() -> None:
    health = {
        "health": "healthy",
        "freshness": "fresh",
        "job_id": "job-exact",
        "provider_evidence": {"source": "process-host", "exit_code": 137},
    }
    unexplained = operational_diagnostic(health, state="failed")
    assert unexplained["failure_disposition"]["reason"] == "terminal_cause_unavailable"
    generic = operational_diagnostic(
        health,
        state="failed",
        failure={"type": "RuntimeError", "message": "CUDA error: illegal memory access"},
    )
    assert generic["failure_disposition"]["reason"] == "consumer_exception"
    assert not generic["failure_disposition"]["automatic_recovery_eligible"]
    oom = operational_diagnostic(
        health,
        state="failed",
        failure={"type": "torch.OutOfMemoryError", "message": "allocation failed"},
    )
    assert oom["failure_disposition"]["reason"] == "gpu_memory_allocation"


def test_malformed_response_is_not_proof_of_unreachable_host(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    store.write(record())
    observe(store, 0, state=JobState.RUNNING, details={"heartbeat_at_utc": NOW.isoformat()})
    observe(store, 1, error=ValueError("invalid response"))
    failed = observe(store, 2, error=ValueError("invalid response"))
    health = OperationalHealth.describe(failed, now=NOW + timedelta(seconds=2))
    assert health["health"] == "degraded"
    assert health["connectivity"] == "unknown"
    assert health["diagnostic"]["category"] == "warning"
    assert health["last_confirmed_state"] == "running"
    assert health["last_confirmed_at_utc"] == NOW.isoformat()


def test_stale_node_failure_fact_is_not_presented_as_current_confirmation() -> None:
    diagnosis = operational_diagnostic(
        {
            "health": "stale",
            "freshness": "stale",
            "reason": "PROVIDER_OBSERVATION_STALE",
            "provider_evidence": {"source": "sacct", "raw_state": "NODE_FAIL"},
        },
        state="unknown",
    )
    assert diagnosis["context"]["confirmation"] == "unconfirmed"
    assert diagnosis["failure_disposition"]["termination_type"] == "unknown"


@pytest.mark.parametrize("raw", ["PREEMPTED", "CANCELLED by 10"])
def test_scheduler_preemption_is_an_incident_but_user_cancellation_is_not(
    tmp_path: Path, raw: str
) -> None:
    store = JobStore(tmp_path)
    store.write(record())
    at = datetime.now(timezone.utc).isoformat()
    store.record_observation(
        "job-health",
        started=at,
        finished=at,
        source="sacct",
        state=JobState.CANCELLED,
        details={"source": "sacct", "raw_state": raw, "exit_code": "0:15"},
    )
    incidents = OperationalIncidents(store).list()
    if raw == "PREEMPTED":
        assert len(incidents) == 1
        assert incidents[0]["code"] == "SCHEDULER_PREEMPTION"
        assert incidents[0]["diagnostic"]["failure_disposition"]["reason"] == "scheduler_preemption"
    else:
        assert not incidents


def test_first_unknown_observation_is_not_scientific_failure_or_gpu_loss() -> None:
    diagnosis = OperationalHealth.describe(record(), now=NOW)["diagnostic"]
    assert diagnosis["category"] == "warning"
    assert diagnosis["failure_disposition"]["termination_type"] == "unknown"
    assert diagnosis["context"]["event_time_utc"] is None
