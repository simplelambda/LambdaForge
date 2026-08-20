"""Regress responsive monitoring and safe managed-environment maintenance."""

from __future__ import annotations

import json
import os
import time
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from lambdaforge.cli.LiveJobMonitor import MonitorRenderer, SnapshotProcess
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
        ProjectWheelBuilder.validate_framework_dependency(
            wheel, "0.9.1", project_root=tmp_path
        )

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
        command
        for command in transport.commands
        if command[1:4] == ("-m", "pip", "install")
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

    result = StorageOperations.prune_environments(
        descriptor, ("env-current",), apply=True
    )

    assert result["pruned"] == ["env-old"]
    assert (environments / "env-current").is_dir()
    assert (environments / "env-running").is_dir()
    assert not (environments / "env-old").exists()
