"""Rejected/accepted recovery is durable operational evidence, not scientific mutation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from lambdaforge.cli.common import diagnostic_context, report_error
from lambdaforge.cli.jobs import run_job_command
from lambdaforge.cli.parser import build_parser
from lambdaforge.controlplane.jobs import JobState
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.diagnostics import DiagnosticContext
from tests.controlplane.test_study_retry import recovery_jobs as recovery_fixture


@pytest.fixture
def recovery_jobs(tmp_path: Path) -> Any:
    return recovery_fixture.__wrapped__(tmp_path)


def test_rejected_apply_survives_reopen_without_touching_science(
    recovery_jobs: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs, previous, execution = recovery_jobs
    (execution / "hpo-control" / "state.json").unlink()
    original = {path: path.read_bytes() for path in execution.rglob("*") if path.is_file()}
    job_bytes = (jobs.store.root / f"{previous.job_id}.json").read_bytes()
    before = {path: path.read_bytes() for path in jobs.store.root.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="missing"):
        jobs.retry(previous.job_id, dry_run=True)
    assert before == {
        path: path.read_bytes() for path in jobs.store.root.rglob("*") if path.is_file()
    }
    with pytest.raises(ValueError, match="missing") as caught:
        jobs.retry(previous.job_id)
    request_id = caught.value.recovery_request_id
    assert request_id.startswith("recovery-")
    reopened = JobService(jobs.catalog, jobs.store, jobs.factory)
    monkeypatch.setattr(reopened, "_provider", lambda *_: pytest.fail("Audit must not query host"))
    audit = reopened.recovery_history(previous.job_id)
    assert [event["phase"] for event in audit["events"]["items"]] == [
        "RECOVERY_BLOCKED",
        "RECOVERY_INSPECTING",
    ]
    assert all(event["recovery_request_id"] == request_id for event in audit["events"]["items"])
    assert audit["provider_refreshed"] is False and audit["science_mutated"] is False
    assert len(jobs.store.records()) == 1
    assert job_bytes == (jobs.store.root / f"{previous.job_id}.json").read_bytes()
    assert original == {path: path.read_bytes() for path in execution.rglob("*") if path.is_file()}


def test_explicit_inspection_is_audited_but_native_dry_run_is_not(recovery_jobs: Any) -> None:
    jobs, previous, _ = recovery_jobs
    assert not jobs.store.events(previous.job_id)
    jobs.retry_preview(previous.job_id)
    assert not jobs.store.events(previous.job_id)
    preview = jobs.inspect_recovery(previous.job_id)
    assert preview["preview_request_id"].startswith("recovery-preview-")
    assert jobs.store.events(previous.job_id)[-1]["phase"] == "RECOVERY_PREVIEW_READY"
    assert len(jobs.store.records()) == 1


def test_inspection_and_audit_io_errors_never_hide_original_error(
    recovery_jobs: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs, previous, _ = recovery_jobs

    def rejected(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("Not compatible; password=private-value")

    monkeypatch.setattr(jobs, "retry_preview", rejected)
    with pytest.raises(ValueError) as inspected:
        jobs.inspect_recovery(previous.job_id)
    audit = jobs.recovery_history(previous.job_id)
    assert "private-value" not in json.dumps(audit)
    assert audit["events"]["items"][0]["recovery_request_id"] == inspected.value.recovery_request_id
    append = jobs.store.append_event

    def broken_audit(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("phase") == "RECOVERY_BLOCKED":
            raise OSError("Audit filesystem unavailable")
        return append(*args, **kwargs)

    monkeypatch.setattr(jobs.store, "append_event", broken_audit)
    with pytest.raises(ValueError, match="Not compatible") as applied:
        jobs.retry(previous.job_id)
    assert type(applied.value) is ValueError
    assert "Audit filesystem unavailable" in applied.value.recovery_audit_error
    assert applied.value.recovery_request_id


def test_audit_links_exact_identity_not_shared_name_and_keeps_native_phases(
    recovery_jobs: Any,
) -> None:
    jobs, previous, _ = recovery_jobs
    request = {"request_id": "recovery-exact", "origin_job_id": previous.job_id}
    new = replace(
        previous,
        job_id="job-recovered",
        retry_of=previous.job_id,
        state=JobState.PREPARING,
        metadata={
            **previous.metadata,
            "recovery_dependencies": [previous.job_id],
            "recovery_request": request,
        },
    )
    jobs.store.write(new)
    for identifier, changed in (
        ("job-foreign-cluster", {"cluster": "other-host"}),
        ("job-foreign-science", {"metadata": {**new.metadata, "scientific_identity": "other"}}),
        (
            "job-foreign-request",
            {
                "metadata": {
                    **new.metadata,
                    "recovery_request": {**request, "origin_job_id": "other"},
                }
            },
        ),
        (
            "job-foreign-dependency",
            {"metadata": {**new.metadata, "recovery_dependencies": previous.job_id}},
        ),
    ):
        jobs.store.write(replace(new, job_id=identifier, **changed))
    audit = jobs.recovery_history(new.job_id)
    assert {row["job_id"] for row in audit["jobs"]} == {previous.job_id, new.job_id}
    assert (
        next(row for row in audit["jobs"] if row["job_id"] == new.job_id)["recovery"]["phase"]
        == "accepted"
    )


def test_audit_page_is_complete_bounded_and_rejects_links(
    recovery_jobs: Any, tmp_path: Path
) -> None:
    jobs, previous, _ = recovery_jobs
    for index in range(207):
        jobs.store.append_event(previous.job_id, state="failed", message=str(index))
    stream = jobs.store.root / f"{previous.job_id}.events.jsonl"
    with stream.open("a") as handle:
        handle.write('{"job_id":"other", "message":"foreign"}\n')
        handle.write("x" * 70000 + '\n{"broken":\n')
    before = stream.read_bytes()
    page = jobs.recovery_history(previous.job_id, limit=100)["events"]
    assert page["total"] == 207 and page["next_offset"] == 100
    assert [int(row["message"]) for row in page["items"]] == list(range(206, 106, -1))
    final = jobs.recovery_history(previous.job_id, offset=200)["events"]
    assert len(final["items"]) == 7 and final["next_offset"] is None
    assert before == stream.read_bytes()
    for bad in ({"limit": True}, {"offset": -1}, {"limit": 201}):
        with pytest.raises(ValueError, match="pagination"):
            jobs.recovery_history(previous.job_id, **bad)
    outside = tmp_path / "untrusted-events.jsonl"
    outside.write_bytes(before)
    stream.unlink()
    stream.symlink_to(outside)
    with pytest.raises(ValueError, match="Unsafe Job audit"):
        jobs.recovery_history(previous.job_id)


def test_cli_saved_audit_and_diagnostic_expose_same_request(
    recovery_jobs: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    jobs, previous, _ = recovery_jobs
    (Path(str(previous.config_path))).unlink()
    with pytest.raises(ValueError) as caught:
        jobs.retry(previous.job_id)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "diagnostics"))
    with diagnostic_context(DiagnosticContext.from_argv(("retry", previous.job_id, "--json"))):
        assert report_error(caught.value) == 2
    diagnostic = json.loads(capsys.readouterr().out)
    assert diagnostic["context"]["recovery_request_id"] == caught.value.recovery_request_id
    monkeypatch.setattr("lambdaforge.cli.jobs.JobService", lambda catalog: jobs)
    monkeypatch.setattr("lambdaforge.cli.jobs.ClusterCatalog.load", lambda *_: jobs.catalog)
    arguments = build_parser().parse_args(["jobs", "recovery", previous.job_id, "--json"])
    assert run_job_command(arguments) == 0
    audit = json.loads(capsys.readouterr().out)
    assert audit["events"]["items"][0]["recovery_request_id"] == caught.value.recovery_request_id
