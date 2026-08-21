"""Focused job timing and usage presentation tests."""

from __future__ import annotations

from datetime import datetime, timezone

from lambdaforge.controlplane.JobObservation import JobObservation
from lambdaforge.controlplane.jobs import JobRecord, JobState
from lambdaforge.controlplane.ResourceService import ResourceService


def test_job_observation_separates_age_queue_runtime_and_actual_usage() -> None:
    record = JobRecord(
        "job-1",
        "gpu",
        "local",
        "job-1",
        JobState.SUCCEEDED,
        ("python", "run.py"),
        "/work",
        {"cpu_cores": 2, "gpu_count": 1},
        "2026-08-21T10:00:00+00:00",
        "2026-08-21T10:03:10+00:00",
        metadata={
            "remote_state": {
                "started_at_utc": "2026-08-21T10:01:00+00:00",
                "finished_at_utc": "2026-08-21T10:03:00+00:00",
                "observed_usage": {"cpu_percent": 150, "rss_bytes": 1024},
            }
        },
    )

    value = JobObservation.describe(record, now=datetime(2026, 8, 21, 10, 5, tzinfo=timezone.utc))

    assert value["timing"]["age_seconds"] == 300
    assert value["timing"]["queue_seconds"] == 60
    assert value["timing"]["runtime_seconds"] == 120
    assert value["timing"]["elapsed_seconds"] == 180
    assert value["usage"]["observed"]["cpu_percent"] == 150
    assert value["usage"]["requested"]["gpu_count"] == 1


def test_personal_cluster_usage_aggregates_active_job_requests_and_observations() -> None:
    first = JobRecord(
        "job-1",
        "gpu",
        "local",
        "job-1",
        JobState.RUNNING,
        ("python",),
        "/work",
        {"cpu_cores": 2, "ram_bytes": 4096, "gpu_count": 1},
        "2026-08-21T10:00:00+00:00",
        "2026-08-21T10:01:00+00:00",
        metadata={
            "remote_state": {
                "observed_usage": {
                    "cpu_percent": 150,
                    "rss_bytes": 2048,
                    "gpu_memory_bytes": 1024,
                }
            }
        },
    )
    second = first.with_updates(
        job_id="job-2",
        resources={"cpu_cores": 1, "ram_bytes": 1024, "gpu_count": 0},
        metadata={},
    )

    value = ResourceService._personal((first, second), "local")

    assert value["requested"] == {
        "cpu_cores": 3,
        "ram_bytes": 5120,
        "gpu_count": 1,
        "gpu_memory_bytes": 0,
    }
    assert value["observed"]["cpu_percent"] == 150
    assert value["observed"]["rss_bytes"] == 2048
    assert value["observed"]["job_count"] == 1
