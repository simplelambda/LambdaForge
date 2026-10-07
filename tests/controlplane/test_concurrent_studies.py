"""Shared host admission and environment contention remain ordinary waiting."""

from __future__ import annotations

import importlib
import os
import subprocess
import time
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import pytest

from lambdaforge.controlplane import CommandResult, LocalTransport
from lambdaforge.controlplane.ManagedEnvironmentProvider import ManagedEnvironmentProvider
from lambdaforge.controlplane.ProcessIdentity import ProcessIdentity
from lambdaforge.controlplane.ProcessSupervisor import ProcessSupervisor
from lambdaforge.controlplane.StorageOperations import StorageOperations
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.runtime.SharedGPUAdmission import shared_gpu_launch


def identity(name: str) -> ProcessIdentity:
    import psutil

    return ProcessIdentity.create(os.getpid(), os.getpgrp(), psutil.Process().cmdline(), name)


def test_shared_jobs_get_overlapping_visibility_without_exclusive_leases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ProcessSupervisor, "_gpu_indices", staticmethod(lambda: (0, 1, 2)))
    monkeypatch.setattr(
        ProcessSupervisor,
        "_gpu_loads",
        staticmethod(lambda: {0: (90, 0.6), 1: (0, 0.01), 2: (0, 0)}),
    )
    for name in ("job-first", "job-second", "job-third"):
        assert ProcessSupervisor._acquire_gpus(
            tmp_path, identity(name), 1, allow_external_use=True
        ) == (2,)
    assert not (tmp_path / "gpu-2.json").exists()
    assert len(list(tmp_path.glob("gpu-2-shared-*.json"))) == 3
    monkeypatch.setattr(ProcessSupervisor, "_externally_occupied_gpus", staticmethod(set))
    # No compute process need exist yet: shared access must still prevent an exclusive race.
    assert ProcessSupervisor._acquire_gpus(tmp_path, identity("job-exclusive-new"), 3) is None
    ProcessSupervisor._release_gpus(tmp_path, "job-first", (2,))
    assert len(list(tmp_path.glob("gpu-2-shared-*.json"))) == 2
    # Shared mode must still respect genuinely exclusive jobs, including older workers.
    ProcessSupervisor._write_json(
        tmp_path / "gpu-2.json", {"supervisor_identity": identity("job-exclusive").to_dict()}
    )
    assert ProcessSupervisor._acquire_gpus(
        tmp_path, identity("job-fourth"), 1, allow_external_use=True
    ) == (1,)


def test_direct_gpu_inventory_never_broadens_inherited_numeric_grant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,0")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "0\n1\n2\n3\n", ""),
    )
    assert ProcessSupervisor._gpu_indices() == (0, 2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert ProcessSupervisor._gpu_indices() == ()


def test_shared_launch_staggers_same_gpu_but_not_sibling_gpu(tmp_path: Path) -> None:
    launches: list[str] = []

    def launch(token: str) -> str:
        launches.append(token)
        return token

    assert shared_gpu_launch(tmp_path, "0", lambda: launch("0"), stagger_seconds=60) == "0"
    assert shared_gpu_launch(tmp_path, "0", lambda: launch("0"), stagger_seconds=60) is None
    assert shared_gpu_launch(tmp_path, "1", lambda: launch("1"), stagger_seconds=60) == "1"
    assert launches == ["0", "1"]


def test_shared_launch_lock_contention_does_not_fail_or_start(tmp_path: Path) -> None:
    def launch() -> str:
        assert shared_gpu_launch(tmp_path, "0", lambda: "racing", stagger_seconds=0) is None
        return "owner"

    assert shared_gpu_launch(tmp_path, "0", launch, stagger_seconds=0) == "owner"


def test_shared_launch_failure_does_not_leave_lock_or_start_timestamp(tmp_path: Path) -> None:
    def fail() -> str:
        raise RuntimeError("worker failed to spawn")

    with pytest.raises(RuntimeError, match="failed to spawn"):
        shared_gpu_launch(tmp_path, "0", fail, stagger_seconds=60)
    assert not list(tmp_path.glob("*.json"))
    assert shared_gpu_launch(tmp_path, "0", lambda: "retry", stagger_seconds=60) == "retry"


def test_environment_waits_past_old_thirty_second_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    transport = LocalTransport()

    def run(command: Any, **kwargs: Any) -> CommandResult:
        del kwargs
        return CommandResult(0 if clock[0] >= 90 else 75)

    monkeypatch.setattr(transport, "run", run)
    monkeypatch.setattr(
        importlib.import_module("lambdaforge.controlplane.CacheBuildLease"),
        "time",
        SimpleNamespace(
            monotonic=lambda: clock[0],
            sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
            time=time.time,
        ),
    )
    assert (
        ManagedEnvironmentProvider._acquire(
            transport, PurePosixPath("/cache/build.lock"), PurePosixPath("/cache/complete")
        )
        is True
    )
    assert clock[0] >= 90


def test_environment_lock_still_has_bounded_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [0.0]
    transport = LocalTransport()
    monkeypatch.setattr(transport, "run", lambda *args, **kwargs: CommandResult(75))
    monkeypatch.setattr(
        importlib.import_module("lambdaforge.controlplane.CacheBuildLease"),
        "time",
        SimpleNamespace(
            monotonic=lambda: clock[0],
            sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
            time=time.time,
        ),
    )
    with pytest.raises(RuntimeError, match="Timed out waiting"):
        ManagedEnvironmentProvider._acquire(
            transport,
            PurePosixPath("/cache/build.lock"),
            PurePosixPath("/cache/complete"),
            timeout=1,
        )


def test_optional_environment_pruning_defers_while_cache_is_in_use(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    with CrossProcessFileLock(
        cache / ".gc.lock", shared=True, timeout_seconds=1, poll_interval_seconds=0.01
    ):
        result = StorageOperations.prune_environments(
            {
                "cache_root": str(cache),
                "state_root": str(tmp_path / "state"),
                "run_root": str(tmp_path / "jobs"),
                "dataset_root": str(tmp_path / "datasets"),
            },
            (),
            apply=True,
        )
    assert result["pruned"] == []
    assert result["applied"] is False
    assert "deferred" in result["blocked_reason"]
