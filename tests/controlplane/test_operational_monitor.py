"""Native monitor lifecycle and durable optional notification fault injection."""

from __future__ import annotations

import json
import os
import time
from datetime import timedelta
from pathlib import Path

import pytest

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.controlplane.MonitorService import MonitorService
from lambdaforge.controlplane.OperationalIncidents import OperationalIncidents
from tests.controlplane.test_operational_health import NOW, record


def test_incidents_deduplicate_escalate_and_resolve_across_restarts(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    store.write(record())
    incidents = OperationalIncidents(store)
    for index in range(10):
        incidents.record(
            record(),
            "CONNECTIVITY_DEGRADED",
            severity="warning",
            summary="offline",
            now=NOW + timedelta(seconds=index),
        )
    assert len(incidents.list()) == 1
    assert incidents.list()[0]["occurrences"] == 1
    assert len(store.events("job-health")) == 1
    restarted = OperationalIncidents(store)
    escalated = restarted.record(
        record(),
        "CONNECTIVITY_DEGRADED",
        severity="error",
        summary="offline",
        now=NOW + timedelta(seconds=10),
    )
    assert escalated["incident_id"] == incidents.list()[0]["incident_id"]
    assert escalated["occurrences"] == 2
    assert len(store.events("job-health")) == 2
    restarted.resolve(escalated["incident_id"], now=NOW + timedelta(seconds=11))
    assert incidents.list()[0]["status"] == "resolved"


def test_notification_failure_is_durable_bounded_and_never_changes_science(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JobStore(tmp_path)
    store.write(record())
    incidents = OperationalIncidents(store)
    incidents.record(
        record(), "CONNECTIVITY_DEGRADED", severity="error", summary="offline", now=NOW
    )
    calls = []

    def unavailable(*args: object) -> None:
        calls.append(args)
        raise RuntimeError("server password=not-safe-to-persist")

    monkeypatch.setattr(OperationalIncidents, "send", unavailable)
    original = (store.root / "job-health.json").read_bytes()
    for seconds in (0, 1, 2, 900, 1800, 2700):
        OperationalIncidents(store).deliver({"enabled": True}, now=NOW + timedelta(seconds=seconds))
    assert len(calls) == 3
    delivered = incidents.list()[0]["notification"]
    assert delivered["status"] == "failed"
    assert delivered["attempts"] == 3
    assert "not-safe-to-persist" not in json.dumps(delivered)
    assert (store.root / "job-health.json").read_bytes() == original


def test_only_provider_acceptance_marks_notification_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JobStore(tmp_path)
    store.write(record())
    incidents = OperationalIncidents(store)
    incidents.record(
        record(), "CONNECTIVITY_DEGRADED", severity="error", summary="offline", now=NOW
    )
    monkeypatch.setattr(OperationalIncidents, "send", lambda *args: None)
    incidents.deliver({"enabled": True}, now=NOW)
    assert incidents.list()[0]["notification"]["status"] == "sent"
    assert incidents.list()[0]["notification"]["accepted_at_utc"] == NOW.isoformat()


def test_smtp_delivery_does_not_lock_provider_commits_or_acknowledge_a_new_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JobStore(tmp_path)
    store.write(record())
    incidents = OperationalIncidents(store)
    first = incidents.record(
        record(),
        "CONNECTIVITY_DEGRADED",
        severity="warning",
        summary="offline",
        now=NOW,
    )

    def accepted_old_transition(*args: object) -> None:
        # An operational transition during SMTP must not wait on the SMTP-held lock.
        incidents.record(
            record(),
            "CONNECTIVITY_DEGRADED",
            severity="error",
            summary="offline",
            now=NOW + timedelta(seconds=1),
        )

    monkeypatch.setattr(OperationalIncidents, "send", accepted_old_transition)
    incidents.deliver({"enabled": True}, now=NOW)
    current = incidents.list()[0]
    assert current["incident_id"] == first["incident_id"]
    assert current["severity"] == "error" and current["occurrences"] == 2
    assert current["notification"]["status"] == "pending"
    assert current["notification"]["attempts"] == 0


def test_monitor_config_is_validated_and_project_scoped(tmp_path: Path) -> None:
    monitor = MonitorService(JobService(store=JobStore(tmp_path)))
    for invalid in (
        {"interval_seconds": 0},
        {"smtp": {"password": "never-store"}},
        {"observation": {"unreachable_failures": 1.2}},
        {"other": 2},
        {
            "smtp": {
                "enabled": True,
                "host": "mail",
                "sender": "x",
                "recipients": ["y"],
                "tls": "none",
            }
        },
    ):
        with pytest.raises((ValueError, TypeError)):
            monitor.config(invalid)
    assert not (monitor.root / "config.json").exists()
    monitor.config({"interval_seconds": 60, "observation": {"progress_seconds": 1000}})
    restored = MonitorService(monitor.jobs)
    assert restored.config()["observation"]["progress_seconds"] == 1000
    assert (
        MonitorService(JobService(store=JobStore(tmp_path / "other"))).config()["observation"] == {}
    )


@pytest.mark.skipif(os.name != "posix", reason="Detached POSIX monitor integration")
def test_detached_monitor_duplicate_start_stop_and_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lambdaforge.ProjectContext import ProjectContext

    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text('[tool.lambdaforge]\nproject_id="observer-test"\n')
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[2] / "src"))
    project = ProjectContext.discover(project_root)
    monitor = MonitorService(JobService(ClusterCatalog.load(project=project)))
    assert monitor.status()["health"] == "stopped"
    try:
        launched = monitor.start()
        assert launched["started"]
        assert monitor.start()["started"] is False
        first = launched["identity"]["pid"]
        monitor.stop()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and monitor.status()["process_lifetime"] == "alive":
            time.sleep(0.1)
        assert monitor.status()["health"] == "stopped"
        restarted = monitor.start()
        assert restarted["identity"]["pid"] != first
        assert monitor.jobs.store.records() == ()
    finally:
        monitor.stop()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and monitor.status()["process_lifetime"] == "alive":
            time.sleep(0.1)


def test_systemd_install_is_preview_only_without_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lambdaforge.ProjectContext import ProjectContext

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "configuration"))
    project = ProjectContext(tmp_path, "native-observer")
    monitor = MonitorService(
        JobService(
            ClusterCatalog.load(project=project), store=JobStore(tmp_path / "jobs", project=project)
        )
    )
    result = monitor.install()
    assert result["applied"] is False
    assert not Path(result["unit"]).exists()
    assert "Restart=on-failure" in result["content"]
    assert "No linger change" in result["logout_persistence"]
