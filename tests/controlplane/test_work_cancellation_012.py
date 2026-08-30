"""Semantic Work cancellation must converge every active provider Job."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, cast

import pytest

from lambdaforge.controlplane import ClusterCatalog, ClusterProfile, JobState
from lambdaforge.controlplane.jobs import JobRecord
from lambdaforge.controlplane.WorkService import WorkService


def _record(job_id: str, state: JobState) -> JobRecord:
    now = datetime.now(timezone.utc).isoformat()
    return JobRecord(
        job_id,
        "local",
        "local",
        job_id,
        state,
        ("python", "work.py"),
        f"/tmp/{job_id}/work",
        {},
        now,
        now,
        metadata={
            "name": "training",
            "scientific_identity": "sha256:same-work",
        },
        job_type="work",
    )


class RecordingJobs:
    def __init__(self, *, fail: str | None = None) -> None:
        self.records = {
            "job-1": _record("job-1", JobState.RUNNING),
            "job-2": _record("job-2", JobState.RUNNING),
            "job-3": _record("job-3", JobState.FAILED),
            "job-4": _record("job-4", JobState.CANCELLED),
        }
        self.cancelled: list[str] = []
        self.fail = fail

    def list(self, *, refresh: bool = True) -> tuple[JobRecord, ...]:
        del refresh
        return tuple(self.records.values())

    def get(self, job_id: str, *, refresh: bool = True) -> JobRecord:
        del refresh
        return self.records[job_id]

    def cancel(self, job_id: str) -> JobRecord:
        self.cancelled.append(job_id)
        if job_id == self.fail:
            raise RuntimeError("provider unavailable")
        stopped = self.records[job_id].with_updates(state=JobState.CANCELLED)
        self.records[job_id] = stopped
        return stopped


def _service(jobs: RecordingJobs) -> WorkService:
    profile = ClusterProfile("local")
    return WorkService(
        ClusterCatalog({"local": profile}),
        jobs=cast(Any, jobs),
        storage=cast(Any, object()),
    )


def test_cancel_work_stops_every_active_job_and_skips_terminal_history() -> None:
    jobs = RecordingJobs()

    result = _service(jobs).cancel("training")

    assert jobs.cancelled == ["job-1", "job-2", "job-4"]
    assert result["status"] == "cancelled"
    assert [value["job_id"] for value in result["cancelled_jobs"]] == ["job-1", "job-2"]
    assert result["reconciled_cancelled_jobs"] == [
        {"job_id": "job-4", "state": "cancelled"}
    ]
    assert result["already_terminal"] == [{"job_id": "job-3", "state": "failed"}]


def test_cancel_work_attempts_remaining_jobs_before_reporting_a_provider_failure() -> None:
    jobs = RecordingJobs(fail="job-1")

    with pytest.raises(RuntimeError, match="cancellation was incomplete"):
        _service(jobs).cancel("training")

    assert jobs.cancelled == ["job-1", "job-2", "job-4"]
    assert jobs.records["job-2"].state is JobState.CANCELLED
