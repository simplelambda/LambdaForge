"""Cluster GPU admission remains explicit, portable and argv-safe."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.CommandResult import CommandResult
from lambdaforge.controlplane.GpuAccessPolicy import GpuAccessPolicy
from lambdaforge.controlplane.ManagedEnvironmentProvider import ManagedEnvironmentProvider
from lambdaforge.controlplane.ProcessIdentity import ProcessIdentity
from lambdaforge.controlplane.ProcessSupervisor import ProcessSupervisor
from lambdaforge.controlplane.TorchInstallationPlan import TorchInstallationPlan
from lambdaforge.work.runner import _visible_gpu_tokens


def test_gpu_access_auto_follows_the_scheduler_boundary() -> None:
    assert ClusterProfile("local").gpu_access.effective_mode("local") == "exclusive"
    slurm = ClusterProfile(
        "slurm",
        transport="ssh",
        scheduler="slurm",
        host="cluster.invalid",
        workspace="/scratch/research",
    )
    assert slurm.gpu_access.effective_mode("slurm") == "scheduler"


def test_shared_direct_access_and_external_claim_prefix_round_trip() -> None:
    shared = ClusterProfile("shared", gpu_access=GpuAccessPolicy("shared"))
    assert ClusterProfile.from_mapping("shared", shared.to_dict()).gpu_access.mode == "shared"

    claimed = ClusterProfile(
        "claimed",
        gpu_access=GpuAccessPolicy(
            "command",
            ("gpu", "exec"),
            ("gpu", "claim", "--numgpus", "{gpu_count}"),
            ("gpu", "release"),
        ),
    )
    restored = ClusterProfile.from_mapping("claimed", claimed.to_dict())
    assert restored.gpu_access.command_prefix == ("gpu", "exec")
    assert restored.gpu_access.claim(2) == ("gpu", "claim", "--numgpus", "2")
    assert restored.gpu_access.release() == ("gpu", "release")
    assert restored.gpu_access.wrap(("/managed/bin/python", "train.py")) == (
        "gpu",
        "exec",
        "/managed/bin/python",
        "train.py",
    )


def test_gpu_access_rejects_ambiguous_or_incompatible_policies() -> None:
    with pytest.raises(ValueError, match="requires command_prefix"):
        GpuAccessPolicy("command")
    with pytest.raises(ValueError, match="configured together"):
        GpuAccessPolicy("command", ("gpu", "exec"), ("gpu", "claim"))
    with pytest.raises(ValueError, match="SLURM clusters"):
        ClusterProfile(
            "bad-slurm",
            transport="ssh",
            scheduler="slurm",
            host="cluster.invalid",
            workspace="/scratch/research",
            gpu_access=GpuAccessPolicy("shared"),
        )


def test_shared_direct_admission_ignores_only_external_occupancy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = ProcessIdentity.create(
        os.getpid(), os.getpgrp(), ("pytest", "gpu-policy"), "job-shared"
    )
    monkeypatch.setattr(ProcessSupervisor, "_gpu_indices", lambda: (0,))
    monkeypatch.setattr(ProcessSupervisor, "_externally_occupied_gpus", lambda: {0})

    assert ProcessSupervisor._acquire_gpus(tmp_path, identity, 1) is None
    assert ProcessSupervisor._acquire_gpus(tmp_path, identity, 1, allow_external_use=True) == (0,)


def test_external_gpu_allocations_are_narrowed_but_never_invented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LAMBDAFORGE_GPU_ACCESS_MODE", "command")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-allocated-a,GPU-allocated-b")

    assert _visible_gpu_tokens(2) == ("GPU-allocated-a", "GPU-allocated-b")
    assert _visible_gpu_tokens(1) == ("GPU-allocated-a",)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-allocated-a")
    with pytest.raises(RuntimeError, match="refusing to invent or broaden"):
        _visible_gpu_tokens(2)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with pytest.raises(RuntimeError, match="did not provide CUDA_VISIBLE_DEVICES"):
        _visible_gpu_tokens(1)


def test_direct_exclusive_mode_keeps_legacy_physical_discovery_without_a_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LAMBDAFORGE_GPU_ACCESS_MODE", "exclusive")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert _visible_gpu_tokens(2) == ("0", "1")


def test_persistent_external_claim_has_a_finally_safe_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = tmp_path / "job"
    work_dir = job_dir / "work"
    work_dir.mkdir(parents=True)
    (job_dir / "state.json").write_text('{"state": "failed"}', encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    def run(command: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
        del kwargs
        calls.append(tuple(command))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", run)
    ProcessSupervisor._release_external_gpu(
        job_dir,
        work_dir,
        {
            "claim_command": ["gpu", "claim", "--numgpus", "2"],
            "release_command": ["gpu", "release"],
        },
    )

    assert calls == [("gpu", "release")]


def test_cuda_environment_verification_uses_wrapper_and_absolute_python() -> None:
    commands: list[tuple[str, ...]] = []

    class Transport:
        def run(self, command: tuple[str, ...]) -> CommandResult:
            commands.append(tuple(command))
            return CommandResult(0, "verified")

    profile = ClusterProfile(
        "claimed",
        gpu_access=GpuAccessPolicy("command", ("gpu", "exec")),
    )
    plan = TorchInstallationPlan(
        "cu128",
        "2.7.0+cu128",
        "https://download.pytorch.org/whl/cu128",
        "cuda",
        require_cuda=True,
    )

    result = ManagedEnvironmentProvider._verify(
        profile,
        Transport(),  # type: ignore[arg-type]
        "/managed/environment/bin/python",
        plan,
    )

    assert result.returncode == 0
    assert commands[0][:3] == ("gpu", "exec", "/managed/environment/bin/python")
