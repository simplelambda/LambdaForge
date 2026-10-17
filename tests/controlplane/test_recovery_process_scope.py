"""Unrelated access sessions must not block recovery; uncertain workers must."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psutil
import pytest

from lambdaforge.controlplane.jobs import JobState
from lambdaforge.controlplane.ProcessIdentity import ProcessIdentity
from lambdaforge.controlplane.ProcessScheduler import ProcessScheduler
from lambdaforge.controlplane.ProcessSupervisor import ProcessSupervisor
from tests.controlplane.test_operational_providers import job_service


def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str = "failed") -> Path:
    directory = tmp_path / "job-fault"
    directory.mkdir()
    supervisor = ProcessIdentity(101, 101, 100.0, "digest", directory.name)
    child = ProcessIdentity(102, 102, 101.0, "digest", directory.name)
    (directory / "request.json").write_text(json.dumps({"job_id": directory.name}))
    (directory / "state.json").write_text(
        json.dumps(
            {
                "job_id": directory.name,
                "state": state,
                "supervisor_identity": supervisor.to_dict(),
                "process_identity": child.to_dict(),
            }
        )
    )
    monkeypatch.setattr(ProcessIdentity, "storage_status", lambda self: "dead")
    monkeypatch.setattr(ProcessSupervisor, "_host_boot_time", lambda: None)
    return directory


UNRELATED = [
    {"pid": 11, "create_time": 10.0},  # PAM/system service predates the fresh native Job.
    {"pid": 12, "create_time": 150.0, "foreign_session_origin": 20.0},
]


@pytest.mark.parametrize("state", ["failed", "cancelled", "timeout", "running"])
def test_job_scoped_denials_preserve_evidence_and_allow_owned_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    directory = host(tmp_path, monkeypatch, state)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    observed = ProcessSupervisor.observe(directory, markers=({}, UNRELATED))
    assert observed["unresolved_process_count"] == 0
    assert observed["excluded_unrelated_process_count"] == 2
    assert observed["owned_process_count"] == 0
    assert observed["provider_observation"] == (
        "lost_confirmed" if state == "running" else "terminal"
    )
    jobs, record = job_service(tmp_path / "controller")
    scheduler = SimpleNamespace(
        state=lambda key: JobState(observed["state"]), details=lambda key: observed
    )
    monkeypatch.setattr(jobs, "_provider", lambda value: (None, None, scheduler))
    evidence = jobs._verify_recovery_provider(record.with_updates(state=JobState.FAILED))
    assert evidence["ownership"] == (
        "lost_confirmed" if state == "running" else "terminal_confirmed"
    )
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    assert len(jobs.store.records()) == 1


@pytest.mark.parametrize(
    "unresolved",
    [
        [{"pid": 13}],
        [{"pid": 13, "create_time": 150.0}],
        [{"pid": 13, "create_time": 100.0}],
        [{"pid": 13, "create_time": 150.0, "foreign_session_origin": 140.0}],
    ],
)
def test_new_unknown_or_reparented_workers_still_block_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unresolved: list[dict[str, Any]]
) -> None:
    directory = host(tmp_path, monkeypatch)
    observed = ProcessSupervisor.observe(directory, markers=({}, UNRELATED + unresolved))
    assert observed["provider_observation"] == "unknown"
    assert observed["unresolved_processes"] == unresolved
    jobs, record = job_service(tmp_path / "controller")
    scheduler = SimpleNamespace(state=lambda key: JobState.FAILED, details=lambda key: observed)
    monkeypatch.setattr(jobs, "_provider", lambda value: (None, None, scheduler))
    with pytest.raises(ValueError, match="PROCESS_OWNERSHIP_UNRESOLVED"):
        jobs._verify_recovery_provider(record.with_updates(state=JobState.FAILED))


@pytest.mark.parametrize("alive", ["supervisor", "child", "orphan", "inaccessible"])
def test_terminal_outcome_is_not_permission_to_duplicate_live_or_inaccessible_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alive: str
) -> None:
    directory = host(tmp_path, monkeypatch)
    monkeypatch.setattr(
        ProcessIdentity,
        "storage_status",
        lambda self: (
            "unresolved"
            if alive == "inaccessible"
            else "alive"
            if self.pid == {"supervisor": 101, "child": 102}.get(alive)
            else "dead"
        ),
    )
    markers = {directory.name: [103]} if alive == "orphan" else {}
    observed = ProcessSupervisor.observe(directory, markers=(markers, []))
    assert observed["provider_observation"] == (
        "unknown" if alive == "inaccessible" else "degraded"
    )


@pytest.mark.skipif(os.name != "posix", reason="POSIX foreign session metadata")
@pytest.mark.parametrize(
    "fault", [None, "init", "same_user", "new_service", "access", "no_session", "pid_reused"]
)
def test_independent_access_session_requires_positive_lifetime_and_foreign_session_evidence(
    monkeypatch: pytest.MonkeyPatch, fault: str | None
) -> None:
    uid = os.getuid() + 1
    parent = SimpleNamespace(
        pid=1 if fault == "init" else 30,
        uids=lambda: (uid,) * 3,
        create_time=lambda: 160.0 if fault == "new_service" else 20.0,
        is_running=lambda: True,
    )
    leader = SimpleNamespace(
        pid=40,
        parent=lambda: parent,
        uids=lambda: ((os.getuid() if fault == "same_user" else uid),) * 3,
        create_time=lambda: 140.0,
        is_running=lambda: True,
    )
    process = SimpleNamespace(
        pid=50, create_time=lambda: 150.0, is_running=lambda: fault != "pid_reused"
    )

    def lookup(pid: int) -> Any:
        if fault == "access":
            raise psutil.AccessDenied(pid)
        return leader

    monkeypatch.setattr(psutil, "Process", lookup)
    monkeypatch.setattr(os, "getsid", lambda pid: 41 if fault == "no_session" else 40)
    value = ProcessSupervisor._unreadable_process(process)
    assert ("foreign_session_origin" in value) == (fault is None)


def test_same_inventory_snapshot_is_scoped_independently_for_each_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = host(tmp_path, monkeypatch)
    second = tmp_path / "job-second"
    second.mkdir()
    (second / "request.json").write_text(json.dumps({"job_id": second.name}))
    (second / "state.json").write_text(
        json.dumps(
            {
                "job_id": second.name,
                "state": "failed",
                "supervisor_identity": ProcessIdentity(201, 201, 200.0, "d", second.name).to_dict(),
                "process_identity": ProcessIdentity(202, 202, 201.0, "d", second.name).to_dict(),
            }
        )
    )
    calls = []

    def scan() -> Any:
        calls.append(1)
        return {}, [{"pid": 13, "create_time": 150.0}]

    monkeypatch.setattr(ProcessSupervisor, "_process_markers", scan)
    rows = {r["request"]["job_id"]: r["state"] for r in ProcessSupervisor.inventory(tmp_path)}
    assert calls == [1]
    assert rows[first.name]["provider_observation"] == "unknown"
    assert rows[second.name]["provider_observation"] == "terminal"


def test_scan_retains_exact_markers_and_denial_metadata_not_a_global_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = SimpleNamespace(username=lambda: "researcher")

    def denied() -> Any:
        raise psutil.AccessDenied(12)

    processes = [
        SimpleNamespace(
            pid=11,
            username=lambda: "researcher",
            status=lambda: psutil.STATUS_SLEEPING,
            environ=lambda: {"LAMBDAFORGE_JOB_ID": "job-fault"},
        ),
        SimpleNamespace(
            pid=12,
            username=lambda: "researcher",
            status=lambda: psutil.STATUS_SLEEPING,
            environ=denied,
        ),
        SimpleNamespace(pid=13, username=lambda: "foreign"),
        SimpleNamespace(pid=14, username=lambda: "researcher", status=lambda: psutil.STATUS_ZOMBIE),
    ]
    monkeypatch.setattr(psutil, "Process", lambda: current)
    monkeypatch.setattr(psutil, "process_iter", lambda *args: processes)
    monkeypatch.setattr(
        ProcessSupervisor, "_unreadable_process", lambda process: {"pid": process.pid}
    )
    monkeypatch.setattr(os, "getpid", lambda: 10)
    markers, unresolved = ProcessSupervisor._process_markers()
    assert markers == {"job-fault": [11]}
    assert unresolved == [{"pid": 12}]


@pytest.mark.parametrize("state", ["failed", "running"])
def test_no_owner_birth_cannot_exclude_uncertain_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    directory = host(tmp_path, monkeypatch, state)
    payload = json.loads((directory / "state.json").read_text())
    payload.update(supervisor_identity=None, process_identity=None)
    (directory / "state.json").write_text(json.dumps(payload))
    observed = ProcessSupervisor.observe(directory, markers=({}, UNRELATED))
    assert observed["provider_observation"] == "unknown"
    assert observed["unresolved_process_count"] == 2


def test_shipped_observer_includes_scope_helpers_without_importing_remote_lambdaforge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = host(tmp_path, monkeypatch)
    code = ProcessScheduler._observation_code("observe")
    assert "def _marker_scope" in code and "def _unreadable_process" in code
    monkeypatch.setattr(sys, "argv", ["observer", str(directory)])
    monkeypatch.setattr(psutil, "process_iter", lambda *args: ())
    exec(code, {})
    assert json.loads(capsys.readouterr().out)["provider_observation"] == "terminal"


@pytest.mark.parametrize("state", ["cancelled", "failed", "running"])
def test_verified_reboot_excludes_new_access_sessions_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    directory = host(tmp_path, monkeypatch, state)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    monkeypatch.setattr(ProcessSupervisor, "_host_boot_time", lambda: 200.0)
    # Even if permissions prevent inspecting a reused PID, the original lifetime
    # predates this boot. Unmarked access sessions were born on the new boot.
    monkeypatch.setattr(ProcessIdentity, "storage_status", lambda self: "unresolved")
    observed = ProcessSupervisor.observe(
        directory,
        markers=({}, [{"pid": 13, "create_time": 250.0}]),
    )
    assert observed["observed_boot_time"] == 200.0
    assert observed["process_lifetimes"] == {
        "supervisor_identity": "dead",
        "process_identity": "dead",
    }
    assert observed["unresolved_process_count"] == 0
    assert observed["excluded_unrelated_process_count"] == 1
    assert observed["provider_observation"] == (
        "lost_confirmed" if state == "running" else "terminal"
    )
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


@pytest.mark.parametrize("fault", ["marker", "unknown_birth", "new_owner", "unknown_boot"])
def test_reboot_does_not_waive_live_markers_or_uncertain_new_owners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    directory = host(tmp_path, monkeypatch)
    monkeypatch.setattr(
        ProcessSupervisor, "_host_boot_time", lambda: None if fault == "unknown_boot" else 200.0
    )
    if fault == "new_owner":
        payload = json.loads((directory / "state.json").read_text())
        payload["process_identity"]["create_time"] = 210.0
        (directory / "state.json").write_text(json.dumps(payload))
        monkeypatch.setattr(ProcessIdentity, "storage_status", lambda self: "unresolved")
    markers = {directory.name: [13]} if fault == "marker" else {}
    uncertain = [{"pid": 14}] if fault == "unknown_birth" else [{"pid": 14, "create_time": 250.0}]
    observed = ProcessSupervisor.observe(directory, markers=(markers, uncertain))
    assert observed["provider_observation"] == ("degraded" if fault == "marker" else "unknown")


@pytest.mark.parametrize("boot", [float("nan"), float("inf"), -1.0, 0.0])
def test_invalid_boot_evidence_cannot_certify_recovery(
    monkeypatch: pytest.MonkeyPatch, boot: float
) -> None:
    monkeypatch.setattr(psutil, "boot_time", lambda: boot)
    assert ProcessSupervisor._host_boot_time() is None


def test_boot_query_failure_stays_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def denied() -> float:
        raise OSError("unavailable")

    monkeypatch.setattr(psutil, "boot_time", denied)
    assert ProcessSupervisor._host_boot_time() is None
