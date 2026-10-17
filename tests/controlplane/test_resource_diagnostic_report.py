"""Bounded, project-owned, sanitized report from saved metadata; no remote computation."""

import json
import os

import pytest

from lambdaforge.cli.parser import build_parser
from lambdaforge.controlplane.jobs import JobRecord, JobState
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.controlplane.ResourceDiagnosticReport import ResourceDiagnosticReport


def record(job="job-01"):
    return JobRecord(
        job,
        "cluster-secret-label",
        "local",
        job,
        JobState.RUNNING,
        ("password=secret-argv",),
        "/private/checkpoints",
        {},
        "2026-10-13T10:00:00+00:00",
        "2026-10-13T11:00:00+00:00",
        metadata={
            "token": "secret-token",
            "dataset": "/private/source",
            "remote_state": {
                "error": "password=secret-error",
                "observed_usage": {
                    "timestamp_utc": "2026-10-13T10:30:00+00:00",
                    "rss_bytes": 100,
                    "process_count": 3,
                    "threads": 4,
                    "gpu_memory_bytes": 50,
                    "memory_attribution": {"rss_bytes": 100, "pss_bytes": 80, "uss_bytes": 60},
                    "gpu_observation": {
                        "quality": "measured",
                        "error": "private exception",
                        "statistics": {"requests": 50, "real_queries": 3, "secrets": "hidden"},
                    },
                },
            },
        },
    )


def test_report_has_separate_saved_observation_and_never_includes_sensitive_bytes(tmp_path):
    store = JobStore(tmp_path)
    store.write(record())
    before = (tmp_path / "job-01.json").read_bytes()
    report = ResourceDiagnosticReport.build(
        store, since="2026-10-13T10:00:00Z", until="2026-10-13T12:00:00Z"
    )
    encoded = json.dumps(report)
    assert report["job_count"] == 1
    row = report["jobs"][0]
    assert row["observed"]["process_count"] == 3
    assert row["memory_attribution"]["pss_bytes"] == 80
    assert row["gpu_observation"]["statistics"]["real_queries"] == 3
    assert not report["coverage"]["complete_timeline"]
    assert all(
        secret not in encoded
        for secret in (
            "secret-argv",
            "secret-token",
            "secret-error",
            "private",
            "hidden",
            "cluster-secret-label",
        )
    )
    assert (tmp_path / "job-01.json").read_bytes() == before
    assert "not establish" in report["causality"]


def test_report_filter_bounds_unknown_metrics_and_never_claims_empty_measurement(tmp_path):
    store = JobStore(tmp_path)
    store.write(record().with_updates(metadata={}, updated_at_utc="2026-10-13T10:30:00Z"))
    report = ResourceDiagnosticReport.build(
        store, since="2026-10-13T10:00:00Z", until="2026-10-13T12:00:00Z"
    )
    assert report["jobs"][0]["observed"]["rss_bytes"] is None
    assert report["jobs"][0]["gpu_observation"]["quality"] == "unknown"
    assert (
        ResourceDiagnosticReport.build(
            store, cluster="another", since="2026-10-13T10:00:00Z", until="2026-10-13T12:00:00Z"
        )["job_count"]
        == 0
    )


def test_bounded_metadata_refuses_links_corruption_and_oversized_records(tmp_path):
    store = JobStore(tmp_path)
    store.write(record())
    (tmp_path / "job-02.json").write_text("broken")
    (tmp_path / "job-03.json").symlink_to(tmp_path / "job-01.json")
    (tmp_path / "job-04.json").write_bytes(b"x" * (2 * 1024**2 + 1))
    value = store.bounded_metadata(limit=10)
    assert len(value["records"]) == 1 and value["skipped"] == 3
    assert store.bounded_metadata(limit=1)["truncated"] == 3


@pytest.mark.skipif(os.name != "posix", reason="FIFO read safety")
def test_diagnostic_metadata_never_blocks_on_non_regular_file(tmp_path):
    os.mkfifo(tmp_path / "job-fifo.json")
    assert JobStore(tmp_path).bounded_metadata()["skipped"] == 1


@pytest.mark.parametrize(
    "since,until",
    [("invalid", "2026-10-13T12:00:00Z"), ("2026-10-14T10:00:00Z", "2026-10-13T12:00:00Z")],
)
def test_invalid_interval_refused(tmp_path, since, until):
    with pytest.raises(ValueError, match="timestamps"):
        ResourceDiagnosticReport.build(JobStore(tmp_path), since=since, until=until)


def test_cli_native_report_flags_are_explicit():
    args = build_parser().parse_args(
        [
            "resources",
            "--report",
            "--on",
            "local",
            "--since",
            "2026-10-13T10:00:00Z",
            "--until",
            "2026-10-13T12:00:00Z",
            "--limit",
            "100",
            "--json",
        ]
    )
    assert args.report and args.limit == 100 and not args.processes
