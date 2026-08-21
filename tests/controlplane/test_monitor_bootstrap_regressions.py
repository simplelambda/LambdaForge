"""Regress responsive monitoring and safe managed-environment maintenance."""

from __future__ import annotations

import json
import os
import time
import zipfile
from collections.abc import Sequence
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from lambdaforge.cli.LiveJobMonitor import (
    ClusterDetailRenderer,
    LiveJobMonitor,
    LogViewerRenderer,
    MonitorRenderer,
    ResourceHistory,
    SnapshotProcess,
    _move_overview_selection,
)
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.CommandResult import CommandResult
from lambdaforge.controlplane.ExecutionBundle import ExecutionBundle
from lambdaforge.controlplane.ManagedEnvironmentProvider import ManagedEnvironmentProvider
from lambdaforge.controlplane.ProjectWheelBuilder import ProjectWheelBuilder
from lambdaforge.controlplane.StorageOperations import StorageOperations
from lambdaforge.controlplane.Transport import Transport


class BlockingOverview:
    """Represent a provider call that cannot complete within an interactive key cycle."""

    @staticmethod
    def snapshot() -> dict[str, Any]:
        time.sleep(30)
        return {}


class ImmediateOverview:
    """Return one serializable machine snapshot immediately."""

    @staticmethod
    def snapshot() -> dict[str, Any]:
        return {"snapshot_version": 1, "jobs": {"items": []}}


def test_snapshot_process_can_be_cancelled_without_waiting_for_provider_timeout() -> None:
    poller = SnapshotProcess(cast(Any, BlockingOverview()))
    poller.start()

    started = time.monotonic()
    poller.close()

    assert time.monotonic() - started < 1.0
    assert not poller.running


def test_snapshot_process_returns_completed_machine_data() -> None:
    poller = SnapshotProcess(cast(Any, ImmediateOverview()))
    poller.start()
    deadline = time.monotonic() + 2.0
    result = None
    while result is None and time.monotonic() < deadline:
        result = poller.take()
        time.sleep(0.01)

    assert result is not None
    payload, error = result
    assert error is None
    assert payload is not None and payload["snapshot_version"] == 1


def test_renderer_scrolls_to_keep_keyboard_selection_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lambdaforge.cli.LiveJobMonitor.shutil.get_terminal_size",
        lambda fallback: os.terminal_size((120, 18)),
    )
    items = [
        {
            "job_id": f"job-{index}",
            "metadata": {"name": f"trial-{index}"},
            "job_type": "task",
            "state": "running",
            "cluster": "gpu",
            "created_at_utc": "",
            "resources": {},
        }
        for index in range(5)
    ]

    rendered = MonitorRenderer.render(
        {"jobs": {"items": items, "by_state": {"running": 5}, "total": 5}},
        selected=4,
    )

    assert "▶ job-4" in rendered
    assert "Selected: job-4" in rendered


def test_monitor_scrolls_clusters_without_always_showing_personal_usage() -> None:
    history = ResourceHistory(30)
    clusters = [
        {
            "cluster": f"gpu-{index}",
            "online": True,
            "observed": {
                "cpu_total": 8,
                "cpu_load": index * 10,
                "ram_total_bytes": 100,
                "ram_available_bytes": 50,
                "gpus": [
                    {
                        "utilization_percent": 25,
                        "memory_total_bytes": 100,
                    }
                ],
            },
            "personal": {
                "observed": {
                    "cpu_percent": 100,
                    "rss_bytes": 10,
                    "gpu_memory_bytes": 20,
                    "job_count": 1,
                },
                "requested": {"cpu_cores": 2, "ram_bytes": 20, "gpu_count": 1},
            },
        }
        for index in range(6)
    ]
    payload = {"clusters": clusters, "jobs": {"items": [], "by_state": {}, "total": 0}}
    history.record(payload)

    rendered = MonitorRenderer.render(
        payload,
        selected_cluster=5,
        focus="clusters",
        history=history,
        width=180,
        height=18,
    )

    assert "▶ gpu-5" in rendered
    assert "↳ me" not in rendered
    assert "mine requested" not in rendered
    assert "clusters 3-6 of 6" in rendered


def test_vertical_navigation_crosses_between_jobs_and_clusters() -> None:
    focus, job, cluster = _move_overview_selection(
        "jobs", 0, 0, direction=-1, job_count=3, cluster_count=2
    )
    assert (focus, job, cluster) == ("clusters", 0, 1)

    focus, job, cluster = _move_overview_selection(
        focus, job, cluster, direction=1, job_count=3, cluster_count=2
    )
    assert (focus, job, cluster) == ("jobs", 0, 1)

    focus, job, cluster = _move_overview_selection(
        focus, job, cluster, direction=1, job_count=3, cluster_count=2
    )
    assert (focus, job, cluster) == ("jobs", 1, 1)


def test_monitor_renders_confirmation_before_cancelling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "clusters": [],
        "jobs": {
            "items": [
                {
                    "job_id": "job-1",
                    "cluster": "local",
                    "state": "running",
                    "created_at_utc": "",
                    "metadata": {},
                }
            ],
            "by_state": {"running": 1},
            "total": 1,
        },
    }

    class ImmediateSnapshots:
        def __init__(self, overview: Any) -> None:
            del overview
            self.pending = False

        @property
        def running(self) -> bool:
            return False

        def start(self) -> None:
            self.pending = True

        def take(self) -> tuple[dict[str, Any], None] | None:
            if not self.pending:
                return None
            self.pending = False
            return payload, None

        def close(self) -> None:
            pass

    class ScriptedTerminal:
        def __init__(self, stream: Any) -> None:
            del stream
            self.keys = iter(("x", "y", "q"))

        def __enter__(self) -> ScriptedTerminal:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def key(self, timeout: float) -> str:
            del timeout
            return next(self.keys)

    class RecordingJobs:
        cancelled: list[str] = []

        def cancel(self, job_id: str) -> Any:
            self.cancelled.append(job_id)
            return SimpleNamespace(job_id=job_id, state=SimpleNamespace(value="cancelled"))

    monkeypatch.setattr("lambdaforge.cli.LiveJobMonitor.SnapshotProcess", ImmediateSnapshots)
    monkeypatch.setattr("lambdaforge.cli.LiveJobMonitor._TerminalSession", ScriptedTerminal)
    output = StringIO()
    jobs = RecordingJobs()

    result = LiveJobMonitor(cast(Any, object()), cast(Any, jobs), interval=1, stream=output).run()

    assert result == 0
    assert jobs.cancelled == ["job-1"]
    assert "Cancel job-1? Press x again, y or Enter to confirm" in output.getvalue()


def test_cluster_detail_renders_history_personal_usage_and_only_cluster_jobs() -> None:
    history = ResourceHistory(30)
    clusters = [
        {
            "cluster": "gpu-a",
            "online": True,
            "observed": {
                "cpu_total": 8,
                "cpu_load": 75,
                "ram_total_bytes": 100,
                "ram_available_bytes": 25,
                "gpus": [{"utilization_percent": 50, "memory_total_bytes": 100}],
            },
            "personal": {
                "observed": {
                    "cpu_percent": 200,
                    "rss_bytes": 20,
                    "gpu_memory_bytes": 25,
                    "job_count": 1,
                },
                "requested": {"cpu_cores": 2, "ram_bytes": 20, "gpu_count": 1},
            },
        },
        {"cluster": "gpu-b", "online": True, "observed": {}, "personal": {}},
    ]
    jobs = [
        {
            "job_id": "job-on-a",
            "cluster": "gpu-a",
            "state": "running",
            "created_at_utc": "",
            "timing": {"runtime_seconds": 10},
        },
        {
            "job_id": "job-on-b",
            "cluster": "gpu-b",
            "state": "running",
            "created_at_utc": "",
        },
    ]
    payload = {"clusters": clusters, "jobs": {"items": jobs}}
    history.record(payload)

    rendered = ClusterDetailRenderer.render(
        payload,
        0,
        selected_job=0,
        history=history,
        message="",
        width=160,
        height=36,
    )

    assert "mine requested: C2" in rendered
    assert "mine observed: C2.0 cores" in rendered
    assert "█ cluster  ▓ mine" in rendered
    assert "job-on-a" in rendered
    assert "job-on-b" not in rendered


def test_complete_log_viewer_pages_over_the_same_document() -> None:
    text = "\n".join(f"line-{index}" for index in range(20))

    rendered = LogViewerRenderer.render("job-1", text, scroll=10, message="", width=80, height=8)

    assert "lines 11-14 of 20" in rendered
    assert "line-10" in rendered
    assert "line-14" not in rendered


def _consumer_wheel(path: Path, requirement: str) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "consumer-1.0.dist-info/METADATA",
            "Metadata-Version: 2.4\n"
            "Name: wisdom-protein\n"
            "Version: 0.7.0\n"
            f"Requires-Dist: {requirement}\n",
        )
    return path


def test_consumer_lambdaforge_bound_is_checked_before_remote_install(tmp_path: Path) -> None:
    wheel = _consumer_wheel(
        tmp_path / "consumer.whl",
        "lambdaforge[adaptive-hpo,parquet]>=0.8.0,<0.9",
    )

    with pytest.raises(ValueError, match=r"excludes LambdaForge 0\.9\.1"):
        ProjectWheelBuilder.validate_framework_dependency(wheel, "0.9.1", project_root=tmp_path)

    compatible = _consumer_wheel(
        tmp_path / "compatible.whl",
        "lambdaforge[parquet]>=0.9,<0.10",
    )
    ProjectWheelBuilder.validate_framework_dependency(compatible, "0.9.1")


def test_online_pip_install_does_not_reference_an_absent_wheelhouse(tmp_path: Path) -> None:
    class RecordingTransport(Transport):
        def __init__(self) -> None:
            self.commands: list[tuple[str, ...]] = []

        def run(
            self,
            command: Sequence[str],
            *,
            cwd: str | Path | None = None,
            timeout: float | None = None,
        ) -> CommandResult:
            del cwd, timeout
            self.commands.append(tuple(command))
            return CommandResult(1) if command[:2] == ("test", "-f") else CommandResult(0)

        def put(self, source: str | Path, destination: str | Path) -> None:
            del source, destination

        def get(self, source: str | Path, destination: str | Path) -> None:
            del source, destination

    transport = RecordingTransport()
    profile = ClusterProfile(
        "gpu", transport="ssh", host="gpu.invalid", workspace="/work", environment="managed"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    bundle = ExecutionBundle(
        "bundle",
        tmp_path,
        manifest,
        manifest,
        2,
        environment_id="env-online",
        package_names=("lambdaforge.whl", "consumer.whl"),
        offline=False,
    )

    ManagedEnvironmentProvider().prepare(
        profile, transport, bundle, remote_bundle_dir="/remote/bundle"
    )

    pip = next(
        command for command in transport.commands if command[1:4] == ("-m", "pip", "install")
    )
    assert "--find-links" not in pip


def test_environment_pruning_retains_active_and_running_job_references(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    environments = cache / "environments"
    state = tmp_path / "state"
    jobs = tmp_path / "jobs"
    for name in ("env-current", "env-old", "env-running"):
        directory = environments / name
        (directory / "bin").mkdir(parents=True)
        (directory / ".lambdaforge-environment.json").write_text("{}", encoding="utf-8")
    state.mkdir()
    (state / "active-environment").write_text(
        str(environments / "env-current" / "bin" / "python"), encoding="utf-8"
    )
    job = jobs / "job-running"
    job.mkdir(parents=True)
    (job / "state.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
    (job / "request.json").write_text(
        json.dumps({"command": [str(environments / "env-running" / "bin" / "python")]}),
        encoding="utf-8",
    )
    descriptor = {
        "state_root": str(state),
        "cache_root": str(cache),
        "run_root": str(jobs),
    }

    result = StorageOperations.prune_environments(descriptor, ("env-current",), apply=True)

    assert result["pruned"] == ["env-old"]
    assert (environments / "env-current").is_dir()
    assert (environments / "env-running").is_dir()
    assert not (environments / "env-old").exists()
