"""Queue/accounting failures and real process ownership must not look like empty success."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.CommandResult import CommandResult
from lambdaforge.controlplane.jobs import JobRecord, JobState
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.controlplane.SlurmScheduler import SlurmScheduler
from tests.controlplane.test_secure_cluster_customization import FakeTransport


class SlurmTransport(FakeTransport):
    def run(self, command: Any, *, cwd: Any = None, timeout: float | None = None) -> Any:
        assert timeout == 15.0
        return super().run(command, cwd=cwd)


@pytest.mark.parametrize(
    ("row", "state"),
    [
        ("NODE_FAIL|NodeDown|0:9|gpu16|start|end", JobState.FAILED),
        ("OUT_OF_MEMORY|None|0:9|gpu12|start|end", JobState.FAILED),
        ("TIMEOUT|TimeLimit|0:15|gpu16|start|end", JobState.TIMEOUT),
        ("PREEMPTED|Preempted|0:15|gpu16|start|end", JobState.CANCELLED),
        ("CANCELLED+|None|0:15|gpu16|start|end", JobState.CANCELLED),
    ],
)
def test_accounting_preserves_specific_terminal_evidence(row: str, state: JobState) -> None:
    transport = SlurmTransport([CommandResult(0, "", ""), CommandResult(0, row, "")])
    scheduler = SlurmScheduler(transport)
    assert scheduler.state("123") is state
    evidence = scheduler.details("123")
    assert evidence["raw_state"] == row.split("|")[0]
    assert evidence["exit_code"] == row.split("|")[2]
    assert evidence["source"] == "sacct"
    assert len(transport.commands) == 2  # details is the same observation, never another RPC


def test_accounting_delay_is_unknown_not_healthy_or_failed() -> None:
    scheduler = SlurmScheduler(SlurmTransport([CommandResult(0, "", "")] * 2))
    assert scheduler.state("123") is JobState.UNKNOWN
    assert scheduler.details("123")["observation_reason"] == "ACCOUNTING_PENDING_OR_JOB_UNKNOWN"


def test_queue_error_requires_positive_accounting_evidence() -> None:
    scheduler = SlurmScheduler(
        SlurmTransport([CommandResult(1, "", "controller unavailable"), CommandResult(0, "", "")])
    )
    with pytest.raises(RuntimeError, match="SLURM observation failed"):
        scheduler.state("123")
    scheduler = SlurmScheduler(
        SlurmTransport(
            [CommandResult(1, "", "queue down"), CommandResult(0, "NODE_FAIL|NodeDown|0:9", "")]
        )
    )
    assert scheduler.state("123") is JobState.FAILED
    assert scheduler.details("123")["queue_error"] == "queue down"


@pytest.mark.parametrize("accounting", ["NODE_FAIL|NodeDown|0:9", ""])
def test_queue_transport_timeout_still_checks_accounting_but_never_fabricates_loss(
    accounting: str,
) -> None:
    calls = []

    class Transport:
        def run(self, command: Any, *, timeout: float) -> Any:
            calls.append(command)
            assert timeout == 15.0
            if command[0] == "squeue":
                raise TimeoutError("queue timeout; password=hidden")
            return CommandResult(0, accounting, "")

    scheduler = SlurmScheduler(Transport())
    if accounting:
        assert scheduler.state("123") is JobState.FAILED
        assert "hidden" not in scheduler.details("123")["queue_error"]
    else:
        with pytest.raises(TimeoutError, match="queue timeout"):
            scheduler.state("123")
        assert "123" not in scheduler._observations
    assert len(calls) == 2


def test_failed_slurm_refresh_discards_previous_details() -> None:
    transport = SlurmTransport(
        [
            CommandResult(0, "RUNNING|None|gpu16", ""),
            CommandResult(1, "", "queue failed"),
            CommandResult(1, "", "accounting failed"),
        ]
    )
    scheduler = SlurmScheduler(transport)
    assert scheduler.state("123") is JobState.RUNNING
    with pytest.raises(RuntimeError, match="SLURM observation failed"):
        scheduler.state("123")
    assert "123" not in scheduler._observations


def job_service(tmp_path: Path) -> tuple[JobService, JobRecord]:
    now = datetime.now(timezone.utc) - timedelta(seconds=1)
    profile = ClusterProfile("fake", transport="local", workspace=str(tmp_path / "remote"))
    jobs = JobService(ClusterCatalog({"fake": profile}), JobStore(tmp_path / "registry"))
    record = JobRecord(
        "job-fault",
        "fake",
        "local",
        "job-fault",
        JobState.RUNNING,
        (),
        str(tmp_path / "remote" / "jobs" / "job-fault" / "work"),
        {},
        now.isoformat(),
        now.isoformat(),
    )
    jobs.store.write(record)
    jobs.store.record_observation(
        record.job_id,
        started=now.isoformat(),
        finished=now.isoformat(),
        source="process-host",
        state=JobState.RUNNING,
        details={"provider_observation": "alive"},
    )
    return jobs, record


def test_inventory_failure_preserves_confirmed_state_and_is_not_empty_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs, record = job_service(tmp_path)

    def disconnected(name: str) -> Any:
        raise TimeoutError("network unavailable")

    monkeypatch.setattr(jobs, "_inventory", disconnected)
    jobs.reconcile(all_clusters=True)
    failed = jobs.store.get(record.job_id)
    assert failed.state is JobState.UNKNOWN
    assert failed.to_dict()["operational"]["connectivity"] == "unreachable"
    assert failed.to_dict()["operational"]["last_confirmed_state"] == "running"
    monkeypatch.setattr(jobs, "_inventory", lambda name: ())
    jobs.reconcile(all_clusters=True)
    unknown = jobs.store.get(record.job_id)
    assert unknown.state is JobState.UNKNOWN
    assert unknown.to_dict()["operational"]["health"] == "unknown"
    assert unknown.to_dict()["operational"]["connectivity"] == "reachable"


@pytest.mark.parametrize("observation", ["unknown", "degraded", "alive"])
def test_recovery_rechecks_live_or_unresolved_ownership_before_duplication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    observation: str,
) -> None:
    jobs, record = job_service(tmp_path)
    terminal = record.with_updates(state=JobState.FAILED)
    scheduler = SimpleNamespace(
        state=lambda key: JobState.RUNNING if observation == "alive" else JobState.FAILED,
        details=lambda key: {"provider_observation": observation},
    )
    monkeypatch.setattr(jobs, "_provider", lambda value: (None, None, scheduler))
    with pytest.raises(ValueError, match="Recovery blocked"):
        jobs._verify_recovery_provider(terminal)
    assert len(jobs.store.records()) == 1
