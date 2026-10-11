"""Recovery admission, process observation and scientific progress are different facts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from lambdaforge.controlplane.jobs import JobRecord, JobState
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.controlplane.RecoveryObservation import RecoveryObservation


def test_recovery_phases_are_native_lifecycle_not_a_successful_button(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc).isoformat()
    record = JobRecord(
        "job-recovery",
        "fake",
        "local",
        None,
        JobState.PREPARING,
        (),
        str(tmp_path),
        {},
        now,
        now,
        metadata={
            "recovery_request": {
                "request_id": "recovery-a",
                "origin_job_id": "job-original",
                "requested_at_utc": now,
            }
        },
    )
    store = JobStore(tmp_path / "registry")
    store.write(record)
    assert record.to_dict()["recovery"]["phase"] == "accepted"
    record = store.merge_metadata(record.job_id, {"submission_phase": "validation"})
    assert record.to_dict()["recovery"]["phase"] == "preparing"
    record = record.with_updates(scheduler_id="job-recovery", state=JobState.RUNNING)
    assert record.to_dict()["recovery"]["phase"] == "scheduler_acknowledged"
    record = record.with_updates(
        metadata={
            **record.metadata,
            "operational_observation": {
                "heartbeat_at_utc": now,
                "progress_signature": {"epoch": 0},
            },
        }
    )
    assert record.to_dict()["recovery"]["phase"] == "runtime_observed"
    record = record.with_updates(
        metadata={
            **record.metadata,
            "operational_observation": {
                "heartbeat_at_utc": now,
                "progress_signature": {"epoch": 2},
                "recovery_progress_at_utc": now,
                "recovery_progress_request_id": "recovery-a",
            },
        }
    )
    assert record.to_dict()["recovery"]["phase"] == "scientific_progress_observed"
    record = record.with_updates(state=JobState.FAILED)
    assert record.to_dict()["recovery"]["phase"] == "failed"
    store.write(record)
    store.append_event(
        record.job_id, state="failed", phase="validation", message="Failed explicitly"
    )
    assert store.events(record.job_id)[0]["recovery_request_id"] == "recovery-a"


def test_unverified_unknown_and_imported_evidence_are_not_recovery_permission() -> None:
    assert not RecoveryObservation.eligible("unknown", {})
    assert not RecoveryObservation.eligible(
        "unknown", {"health": "lost_confirmed", "freshness": "stale"}
    )
    assert RecoveryObservation.eligible(
        "unknown", {"health": "lost_confirmed", "freshness": "fresh"}
    )
