"""Recovery acknowledgement is not incident resolution or scientific progress."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lambdaforge.controlplane.jobs import JobState
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.controlplane.OperationalIncidents import OperationalIncidents
from tests.controlplane.test_operational_health import record

NOW = datetime.now(timezone.utc) - timedelta(seconds=10)


@pytest.fixture(autouse=True)
def fresh_observer_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    # Collection may precede execution by minutes in the full suite. These tests exercise
    # current receipt freshness, not the age of pytest's module import.
    # Pytest may import this file as test_recovery_incidents while other fixtures import
    # tests.controlplane.test_recovery_incidents. Patch this exact module, not its alias.
    monkeypatch.setitem(globals(), "NOW", datetime.now(timezone.utc) - timedelta(seconds=10))


def setup_recovery(tmp_path: Path) -> tuple[JobStore, OperationalIncidents, str]:
    assert abs((datetime.now(timezone.utc) - NOW).total_seconds()) < 30
    store = JobStore(tmp_path)
    origin = record().with_updates(state=JobState.FAILED)
    store.write(origin)
    incidents = OperationalIncidents(store)
    incident = incidents.record(origin, "EXECUTION_FAILED", severity="error", summary="lost")
    store.append_event(
        origin.job_id,
        state="failed",
        phase="RECOVERY_ACCEPTED",
        message="Accepted, not running yet",
        recovery_request_id="recovery-a",
    )
    store.write(
        origin.with_updates(
            job_id="job-recovery",
            state=JobState.PREPARING,
            retry_of=origin.job_id,
            metadata={
                "scientific_identity": "original",
                "recovery_work_identity": "original",
                "recovery_dependencies": [origin.job_id],
                "recovery_request": {
                    "origin_job_id": origin.job_id,
                    "request_id": "recovery-a",
                    "requested_at_utc": (NOW - timedelta(seconds=1)).isoformat(),
                },
            },
        )
    )
    return store, incidents, incident["incident_id"]


def observe_recovery(store: JobStore, seconds: int, details: dict) -> None:
    at = (NOW + timedelta(seconds=seconds)).isoformat()
    store.record_observation(
        "job-recovery",
        started=at,
        finished=at,
        source="process-host",
        state=JobState.RUNNING,
        details={"heartbeat_at_utc": at, **details},
    )


def test_only_fresh_scientific_progress_resolves_audited_origin(tmp_path: Path) -> None:
    store, incidents, identifier = setup_recovery(tmp_path)
    origin_bytes = (store.root / "job-health.json").read_bytes()
    assert incidents.list()[0]["status"] == "open"
    observe_recovery(store, 0, {"heartbeat_at_utc": NOW.isoformat()})
    assert incidents.list()[0]["status"] == "open"
    observe_recovery(store, 1, {"progress": {"completed": 0, "total": 10}})
    assert incidents.list()[0]["status"] == "open"
    observe_recovery(store, 2, {"progress": {"completed": 1, "total": 10}})
    incident = next(value for value in incidents.list() if value["incident_id"] == identifier)
    assert incident["status"] == "resolved"
    assert incident["recovered_by"]["job_id"] == "job-recovery"
    assert incident["recovered_by"]["request_id"] == "recovery-a"
    assert (store.root / "job-health.json").read_bytes() == origin_bytes
    events = store.events("job-health")
    assert events[-1]["recovery_request_id"] == "recovery-a"
    assert events[-1]["phase"] == "INCIDENT_RESOLVED"
    observe_recovery(store, 3, {"progress": {"completed": 2, "total": 10}})
    assert store.events("job-health") == events
    assert OperationalIncidents(JobStore(tmp_path)).list()[0]["status"] == "resolved"


def test_reused_runs_total_changes_and_counter_resets_are_not_recovery_progress(
    tmp_path: Path,
) -> None:
    store, incidents, _ = setup_recovery(tmp_path)
    for seconds, progress in enumerate(
        (
            {"completed": 9, "total": 10},  # reused evidence loaded
            {"completed": 9, "total": 100},  # planner adjusted obligations, no computation
            {"completed": 8, "total": 10},  # failed Run requeued, not progress
            {"completed": 8, "total": 10},
        )
    ):
        observe_recovery(store, seconds, {"progress": progress})
        assert incidents.list()[0]["status"] == "open"
        assert not store.get("job-recovery").to_dict()["recovery"]["scientific_progress_observed"]
    observe_recovery(store, 4, {"progress": {"completed": 9, "total": 10}})
    assert incidents.list()[0]["status"] == "resolved"


@pytest.mark.parametrize("state", [JobState.UNKNOWN, JobState.FAILED, JobState.RUNNING])
def test_loss_or_terminal_failure_cannot_stamp_recovery_progress(
    tmp_path: Path, state: JobState
) -> None:
    store, incidents, _ = setup_recovery(tmp_path)
    observe_recovery(store, 0, {"progress": {"completed": 9, "total": 10}})
    at = (NOW + timedelta(seconds=1)).isoformat()
    store.record_observation(
        "job-recovery",
        started=at,
        finished=at,
        source="process-host",
        state=state,
        details={
            "progress": {"completed": 10, "total": 10},
            "provider_observation": "lost_confirmed" if state == JobState.RUNNING else "confirmed",
        },
    )
    assert not store.get("job-recovery").to_dict()["recovery"]["scientific_progress_observed"]
    assert incidents.list()[0]["status"] == "open"
    # A subsequent heartbeat alone cannot legitimize that earlier failure/loss snapshot.
    observe_recovery(store, 2, {})
    assert not store.get("job-recovery").to_dict()["recovery"]["scientific_progress_observed"]
    assert incidents.list()[0]["status"] == "open"


@pytest.mark.parametrize("reused", [0, 9])
def test_preparation_zero_and_unknown_snapshot_are_not_a_runtime_progress_baseline(
    tmp_path: Path,
    reused: int,
) -> None:
    store, incidents, identifier = setup_recovery(tmp_path)
    for seconds, state, completed in (
        (0, JobState.PREPARING, 0),
        (1, JobState.UNKNOWN, 0),
    ):
        at = (NOW + timedelta(seconds=seconds)).isoformat()
        store.record_observation(
            "job-recovery",
            started=at,
            finished=at,
            source="process-host",
            state=state,
            details={"progress": {"completed": completed, "total": 10}},
        )
    observe_recovery(store, 2, {"progress": {"completed": reused, "total": 10}})
    assert not store.get("job-recovery").to_dict()["recovery"]["scientific_progress_observed"]
    origin = next(value for value in incidents.list() if value["incident_id"] == identifier)
    assert origin["status"] == "open"
    observe_recovery(store, 3, {"progress": {"completed": reused + 1, "total": 10}})
    origin = next(value for value in incidents.list() if value["incident_id"] == identifier)
    assert origin["status"] == "resolved"


@pytest.mark.parametrize("observations", [(0, 9, 10), (10,)])
def test_native_preview_baseline_distinguishes_reuse_from_first_fresh_completion(
    tmp_path: Path,
    observations: tuple[int, ...],
) -> None:
    store, incidents, identifier = setup_recovery(tmp_path)
    current = store.get("job-recovery")
    store.merge_metadata(
        current.job_id,
        {
            "recovery_request": {
                **current.metadata["recovery_request"],
                "plan": {
                    "preserved_completed_runs": 9,
                },
            }
        },
    )
    for seconds, completed in enumerate(observations):
        observe_recovery(store, seconds, {"progress": {"completed": completed, "total": 10}})
        origin = next(value for value in incidents.list() if value["incident_id"] == identifier)
        assert origin["status"] == ("open" if completed <= 9 else "resolved")
    progressed = store.get(current.job_id).metadata["operational_observation"][
        "recovery_progress_at_utc"
    ]
    observe_recovery(store, 5, {"progress": {"completed": 10, "total": 10}})
    assert (
        store.get(current.job_id).metadata["operational_observation"]["recovery_progress_at_utc"]
        == progressed
    )


@pytest.mark.parametrize(
    "change", ["request", "identity", "cluster", "dependency", "provider", "failed", "old_progress"]
)
def test_unverified_recovery_cannot_resolve_original_incident(tmp_path: Path, change: str) -> None:
    store, incidents, identifier = setup_recovery(tmp_path)
    current = store.get("job-recovery")
    metadata = dict(current.metadata)
    if change == "request":
        metadata["recovery_request"] = dict(metadata["recovery_request"])
        metadata["recovery_request"]["request_id"] = "not-accepted"
    if change == "identity":
        metadata["recovery_work_identity"] = "other"
    if change == "dependency":
        metadata["recovery_dependencies"] = []
    if change == "old_progress":
        metadata["recovery_request"] = {
            **metadata["recovery_request"],
            "requested_at_utc": (NOW + timedelta(seconds=100)).isoformat(),
        }
    store.write(
        current.with_updates(
            metadata=metadata,
            cluster="other" if change == "cluster" else current.cluster,
            state=JobState.FAILED if change == "failed" else current.state,
        )
    )
    details = {"progress": {"completed": 1, "total": 10}}
    if change == "provider":
        details["provider_observation"] = "unknown"
    observe_recovery(store, 0, details)
    incident = next(value for value in incidents.list() if value["incident_id"] == identifier)
    assert incident["status"] == "open"
