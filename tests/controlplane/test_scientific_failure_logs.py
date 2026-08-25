"""Structured scientific failures remain visible across provider log boundaries."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lambdaforge.cli.jobs import run_job_command
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.CommandResult import CommandResult
from lambdaforge.controlplane.jobs import JobRecord, JobState
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.controlplane.Scheduler import Scheduler
from lambdaforge.controlplane.Transport import Transport


def _execution_result(*, failed: bool = True) -> dict[str, Any]:
    failure = (
        {
            "type": "IncompleteRead",
            "message": "IncompleteRead(2183 bytes read)",
            "traceback": (
                "Traceback (most recent call last):\n"
                "  File \"work.py\", line 20, in run\n"
                "http.client.IncompleteRead: IncompleteRead(2183 bytes read)\n"
            ),
            "diagnostic": {"operation": "resume_map item download"},
        }
        if failed
        else None
    )
    return {
        "execution_result_version": 1,
        "name": "wisdom-dna-design",
        "execution_id": "execution-abc",
        "status": "failed" if failed else "succeeded",
        "runs": [
            {
                "name": "wisdom-dna-design",
                "run_id": "run-1",
                "attempt_id": "attempt-1",
                "attempt_number": 1,
                "job_id": "job-1",
                "failure": failure,
            }
        ],
        "outputs": {},
        "summary": {},
    }


class _ResultTransport(Transport):
    LEGACY_PATH = "/remote/jobs/job-1/work/.lambdaforge/runs/study/execution-abc/result.json"

    def __init__(self, result: dict[str, Any] | None, *, legacy: bool = False) -> None:
        self.result, self.legacy = result, legacy

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        del cwd, timeout
        if tuple(command[:2]) == ("test", "-L"):
            return CommandResult(1)
        if command[0] == "cat" and command[1] == "/remote/jobs/job-1/result.json":
            return CommandResult(
                0 if self.result is not None and not self.legacy else 1,
                json.dumps(self.result) if self.result is not None and not self.legacy else "",
            )
        if command[0] == "find":
            return CommandResult(0, f"{self.LEGACY_PATH}\n") if self.legacy else CommandResult(1)
        if command[0] == "cat" and command[1] == self.LEGACY_PATH:
            return CommandResult(
                0 if self.result is not None else 1,
                json.dumps(self.result) if self.result is not None else "",
            )
        return CommandResult(1)

    def put(self, source: str | Path, destination: str | Path) -> None:
        raise AssertionError(f"Log inspection must not transfer {source} to {destination}.")


class _LogScheduler(Scheduler):
    def __init__(self, output: str) -> None:
        self.output = output

    def submit(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("The log test never submits work.")

    def state(self, scheduler_id: str) -> JobState:
        del scheduler_id
        return JobState.FAILED

    def logs(self, scheduler_id: str, *, tail: int | None = None) -> str:
        del scheduler_id
        lines = self.output.splitlines(keepends=True)
        return "".join(lines[-tail:]) if tail is not None else self.output

    def cancel(self, scheduler_id: str) -> None:
        del scheduler_id


class _Factory:
    def __init__(self, result: dict[str, Any] | None, output: str, *, legacy: bool = False) -> None:
        self.remote = _ResultTransport(result, legacy=legacy)
        self.provider = _LogScheduler(output)

    def transport(self, profile: ClusterProfile) -> Transport:
        del profile
        return self.remote

    def scheduler(self, profile: ClusterProfile, transport: Transport) -> Scheduler:
        del profile, transport
        return self.provider


def _service(
    tmp_path: Path,
    *,
    result: dict[str, Any] | None,
    output: str = "last scientific line\n",
    legacy: bool = False,
) -> JobService:
    profile = ClusterProfile.from_mapping(
        "remote",
        {
            "transport": "ssh",
            "host": "gpu.invalid",
            "workspace": "/remote",
            "scheduler": "local",
        },
    )
    store = JobStore(tmp_path / "records")
    now = datetime.now(timezone.utc).isoformat()
    store.write(
        JobRecord(
            "job-1",
            "remote",
            "local",
            "job-1",
            JobState.FAILED if result and result.get("status") == "failed" else JobState.SUCCEEDED,
            ("python", "-m", "lambdaforge", "run", "config.yaml"),
            "/remote/jobs/job-1/work",
            {},
            now,
            now,
            metadata={"name": "wisdom-dna-design"},
            job_type="work",
        )
    )
    return JobService(
        ClusterCatalog({"remote": profile}),
        store,
        _Factory(result, output, legacy=legacy),  # type: ignore[arg-type]
    )


def test_remote_logs_append_structured_failure_after_tail(tmp_path: Path) -> None:
    report = _service(tmp_path, result=_execution_result()).log_report("job-1", tail=1)

    assert "last scientific line" in report["text"]
    assert "== Scientific failure ==" in report["text"]
    assert "IncompleteRead: IncompleteRead(2183 bytes read)" in report["text"]
    assert "Phase: resume_map item download" in report["text"]
    assert "Persisted result: /remote/jobs/job-1/result.json" in report["text"]
    assert "Traceback (most recent call last)" not in report["text"]
    assert report["failure"]["type"] == "IncompleteRead"
    assert report["failure"]["location"] == "wisdom-dna-design, Attempt 1"


def test_verbose_remote_logs_include_persisted_traceback(tmp_path: Path) -> None:
    report = _service(tmp_path, result=_execution_result()).log_report(
        "job-1", include_traceback=True
    )

    assert "Traceback (most recent call last)" in report["text"]
    assert "work.py" in report["text"]


def test_existing_traceback_is_not_duplicated(tmp_path: Path) -> None:
    failure = _execution_result()["runs"][0]["failure"]
    output = f"{failure['traceback']}"
    report = _service(tmp_path, result=_execution_result(), output=output).log_report(
        "job-1", include_traceback=True
    )

    assert report["text"].count("Traceback (most recent call last)") == 1
    assert "== Scientific failure ==" not in report["text"]


def test_successful_work_has_no_scientific_failure_section(tmp_path: Path) -> None:
    report = _service(tmp_path, result=_execution_result(failed=False)).log_report("job-1")

    assert report["failure"] is None
    assert report["failures"] == []
    assert "Scientific failure" not in report["text"]


def test_pre_change_workspace_result_is_discovered_and_matched_to_job(tmp_path: Path) -> None:
    report = _service(tmp_path, result=_execution_result(), legacy=True).log_report("job-1")

    assert report["failure"]["type"] == "IncompleteRead"
    assert report["result_path"] == _ResultTransport.LEGACY_PATH


def test_jobs_logs_json_exposes_structured_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = _service(tmp_path, result=_execution_result())
    monkeypatch.setattr("lambdaforge.cli.jobs.JobService", lambda _catalog: service)
    code = run_job_command(
        SimpleNamespace(
            clusters=None,
            job_command="logs",
            job_id="job-1",
            follow=False,
            tail=1,
            json=True,
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["failure"]["type"] == "IncompleteRead"
    assert payload["result_path"] == "/remote/jobs/job-1/result.json"


def test_jobs_show_human_exposes_failure_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = _service(tmp_path, result=_execution_result())
    monkeypatch.setattr("lambdaforge.cli.jobs.JobService", lambda _catalog: service)
    code = run_job_command(
        SimpleNamespace(
            clusters=None,
            job_command="show",
            job_id="job-1",
            json=False,
        )
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "Scientific failure: IncompleteRead" in output
    assert "Location: wisdom-dna-design, Attempt 1" in output
