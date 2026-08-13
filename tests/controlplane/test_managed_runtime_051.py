"""Managed environments and lightweight remote result transfer for 0.5.1."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lambdaforge.artifacts import RemoteArtifactService
from lambdaforge.controlplane import (
    ClusterCatalog,
    ClusterProfile,
    CommandResult,
    EnvironmentIdentity,
    ExecutionBundle,
    JobRecord,
    JobService,
    JobState,
    JobStore,
    ManagedEnvironmentProvider,
    Transport,
)
from lambdaforge.results import RemoteResultService


class RuntimeTransport(Transport):
    """Model remote files and commands without network or package mutation."""

    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.files: dict[str, bytes] = {}

    def run(self, command: Sequence[str], *, cwd: str | Path | None = None) -> CommandResult:
        self.commands.append(tuple(command))
        if command[:2] == ("test", "-f"):
            return CommandResult(0 if command[2] in self.files else 1)
        if "rglob('*')" in " ".join(command):
            prefix = f"{str(cwd).rstrip('/')}"
            payload = [
                {"path": path.removeprefix(f"{prefix}/"), "size": len(value)}
                for path, value in sorted(self.files.items())
                if path.startswith(f"{prefix}/")
                and (
                    Path(path).name
                    in {"result.json", "metrics.csv", "environment.json", "config.yaml"}
                    or Path(path).name.endswith(".plot.json")
                )
            ]
            return CommandResult(0, json.dumps(payload))
        if len(command) >= 2 and "Path(sys.argv[1]).write_text" in " ".join(command):
            self.files[command[-2]] = command[-1].encode("utf-8")
        return CommandResult(0, "0.5.3\n")

    def put(self, source: str | Path, destination: str | Path) -> None:
        del source, destination

    def get(self, source: str | Path, destination: str | Path) -> None:
        Path(destination).write_bytes(self.files[str(source)])


class RuntimeFactory:
    """Return one deterministic transport to remote services."""

    def __init__(self, transport: RuntimeTransport) -> None:
        self.instance = transport

    def transport(self, profile: ClusterProfile) -> RuntimeTransport:
        del profile
        return self.instance


class TestManagedRuntime051:
    """Exercise idempotence, offline policy and explicit remote artifact selection."""

    def test_managed_environment_is_idempotent_and_offline(self, tmp_path: Path) -> None:
        package = tmp_path / "packages" / "lambdaforge-0.5.3-py3-none-any.whl"
        package.parent.mkdir()
        package.write_bytes(b"wheel")
        manifest = tmp_path / "manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        identity = EnvironmentIdentity.create(
            ({"name": package.name, "sha256": "sha256:x", "size_bytes": 5},),
            python_requirement=">=3.10",
            offline=True,
        )
        bundle = ExecutionBundle(
            "bundle-test",
            tmp_path,
            manifest,
            manifest,
            7,
            environment_id=identity.environment_id,
            package_names=(package.name,),
            offline=True,
        )
        profile = ClusterProfile(
            "cluster",
            transport="ssh",
            host="cluster",
            workspace="/work/user",
            environment="managed",
        )
        transport = RuntimeTransport()
        provider = ManagedEnvironmentProvider()

        first = provider.prepare(profile, transport, bundle, remote_bundle_dir="/remote/bundle")
        second = provider.prepare(profile, transport, bundle, remote_bundle_dir="/remote/bundle")

        assert not first.reused and second.reused
        pip = next(command for command in transport.commands if "pip" in command)
        assert "--no-index" in pip
        assert all(
            not any(argument.startswith(("cuda-", "nvidia-")) for argument in command)
            for command in transport.commands
        )
        assert sum("venv" in command for command in transport.commands) == 1
        assert transport.files["/work/user/.lambdaforge/active-environment"].decode().strip() == (
            first.python
        )

    def test_project_module_rejects_command_injection(self) -> None:
        with pytest.raises(ValueError, match="fully qualified"):
            ClusterProfile("cluster", project_module="project; import os")

    def test_remote_sync_is_small_and_checkpoint_fetch_is_explicit(self, tmp_path: Path) -> None:
        transport = RuntimeTransport()
        work = "/remote/work"
        result = {
            "name": "remote-study",
            "run_dir": work,
            "status": "ok",
            "best_model_path": "checkpoints/best.ckpt",
            "last_model_path": None,
        }
        transport.files[f"{work}/result.json"] = json.dumps(result).encode("utf-8")
        transport.files[f"{work}/metrics.csv"] = b"epoch,val_loss\n0,0.5\n"
        transport.files[f"{work}/checkpoints/best.ckpt"] = b"large-checkpoint"
        transport.files[f"{work}/plots/learning.png.plot.json"] = b"{}"
        profile = ClusterProfile("cluster", transport="ssh", host="cluster", workspace="/work/user")
        catalog = ClusterCatalog({"cluster": profile})
        store = JobStore(tmp_path / "jobs")
        now = datetime.now(timezone.utc).isoformat()
        store.write(
            JobRecord(
                job_id="job-remote",
                cluster="cluster",
                scheduler="slurm",
                scheduler_id="1",
                state=JobState.SUCCEEDED,
                command=("/remote/python", "-m", "lambdaforge", "run", "config.yaml"),
                work_dir=work,
                resources={},
                created_at_utc=now,
                updated_at_utc=now,
            )
        )
        factory = RuntimeFactory(transport)
        jobs = JobService(catalog, store, factory)  # type: ignore[arg-type]
        remote_results = RemoteResultService(
            jobs,
            catalog,
            factory,
            tmp_path / "synced",  # type: ignore[arg-type]
        )

        synced = remote_results.sync("job-remote")

        assert set(synced.files) == {
            "metrics.csv",
            "plots/learning.png.plot.json",
            "result.json",
        }
        assert "checkpoints/best.ckpt" not in synced.files
        artifacts = RemoteArtifactService(
            jobs,
            catalog,
            factory,
            remote_results,  # type: ignore[arg-type]
        )
        assert artifacts.list("job-remote")[0]["logical_name"] == "best-checkpoint"
        fetched = artifacts.fetch("job-remote", "best-checkpoint", tmp_path / "best.ckpt")
        assert fetched.read_bytes() == b"large-checkpoint"
