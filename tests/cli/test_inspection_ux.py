"""Read-only native Job diagnostics/ambiguous names and observer cancellation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pytest

from lambdaforge.cli.inspection import render_inspection, run_inspection_command
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.InspectionService import InspectionService
from lambdaforge.controlplane.jobs import JobRecord, JobState
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.ProjectContext import ProjectContext


@pytest.fixture
def service(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="inspection-fixture"\nversion="1"\n')
    project = ProjectContext.discover(tmp_path)
    catalog = ClusterCatalog({"local": ClusterProfile("local")}, project=project)
    store = JobStore(tmp_path / "job-metadata", project=project)
    at = datetime.now(timezone.utc).isoformat()
    store.write(
        JobRecord(
            "job-inspection",
            "local",
            "local",
            None,
            JobState.UNKNOWN,
            ("python",),
            str(tmp_path),
            {},
            at,
            at,
            metadata={"name": "same-name", "scientific_identity": "sha256:original"},
        )
    )
    jobs = JobService(catalog, store=store)
    return InspectionService(catalog, jobs=jobs)


def test_diagnose_keeps_unknown_distinct_and_cannot_retry_or_probe(service, monkeypatch):
    monkeypatch.setattr(
        service.jobs, "retry", lambda *a, **kw: pytest.fail("diagnose cannot retry")
    )
    monkeypatch.setattr(service.jobs, "_reconcile", lambda *a, **kw: pytest.fail("hidden probe"))
    service.jobs.store.append_event(
        "job-inspection", state="unknown", phase="observation", message="test"
    )
    value = service.diagnose("job-inspection")
    assert value["entity"]["state"] == "unknown" and not value["science_mutated"]
    assert not value["recovery"]["automatic_retry"]
    assert value["possible_causes"] == []
    assert value["recent_history"][0]["message"] == "test"
    assert "not inspected" in render_inspection(value)


def test_name_collision_is_explicit_instead_of_arbitrary_catalogue_order(service):
    with pytest.raises(ValueError, match="Ambiguous selector"):
        service.inspect("same-name")
    assert service.inspect("same-name", kind="job")["job_id"] == "job-inspection"
    assert service.inspect("same-name", kind="work")["work_id"].startswith("work-")


def test_inspection_uses_compact_catalog_without_reading_result_or_artifact_bytes(
    service, monkeypatch
):
    from lambdaforge.work.ResultStore import ResultStore

    monkeypatch.setattr(
        ResultStore,
        "catalog",
        lambda self: (
            {
                "name": "execution-test",
                "execution_id": "execution-fixture",
                "status": "succeeded",
                "run_count": 20,
                "attempt_count": 20,
            },
        ),
    )
    monkeypatch.setattr(ResultStore, "select", lambda *a, **kw: pytest.fail("deep evidence read"))
    assert service.inspect("execution-fixture")["run_count"] == 20


def test_watch_ndjson_is_bounded_backoff_and_leaving_never_cancels(service, monkeypatch, capsys):
    import lambdaforge.cli.inspection as commands

    monkeypatch.setattr(commands, "InspectionService", lambda catalog: service)
    monkeypatch.setattr(ClusterCatalog, "load", classmethod(lambda cls, *a: service.catalog))
    monkeypatch.setattr(service.jobs, "cancel", lambda *a: pytest.fail("watch cannot cancel"))
    waits = []
    monkeypatch.setattr(commands.time, "sleep", waits.append)
    arguments = argparse.Namespace(
        command="watch",
        selector="job-inspection",
        kind="job",
        json=True,
        clusters=None,
        refresh=False,
        interval=2,
        count=3,
    )
    assert run_inspection_command(arguments) == 0
    assert len(capsys.readouterr().out.splitlines()) == 1
    assert waits == [2, 3]

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(commands.time, "sleep", interrupt)
    arguments.count = None
    assert run_inspection_command(arguments) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "unknown"


def test_invalid_watch_arguments_fail_before_loading_or_refreshing_catalog(monkeypatch):
    monkeypatch.setattr(
        ClusterCatalog, "load", lambda *a: pytest.fail("invalid query cannot probe")
    )
    args = argparse.Namespace(command="watch", interval=0, count=3)
    with pytest.raises(ValueError, match="Watch interval"):
        run_inspection_command(args)


def test_one_shot_inspection_has_the_same_response_bound(service, monkeypatch):
    import lambdaforge.cli.inspection as commands

    monkeypatch.setattr(commands, "InspectionService", lambda *a: service)
    monkeypatch.setattr(ClusterCatalog, "load", classmethod(lambda cls, *a: service.catalog))
    monkeypatch.setattr(service, "inspect", lambda *a, **kw: {"oversized": "x" * 512 * 1024})
    args = argparse.Namespace(
        command="inspect",
        selector="job-inspection",
        kind="job",
        clusters=None,
        refresh=False,
        json=True,
    )
    with pytest.raises(ValueError, match="512 KiB"):
        run_inspection_command(args)
