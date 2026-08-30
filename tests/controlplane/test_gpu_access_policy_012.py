"""Cluster GPU admission remains explicit, portable and argv-safe."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.GpuAccessPolicy import GpuAccessPolicy
from lambdaforge.controlplane.ProcessIdentity import ProcessIdentity
from lambdaforge.controlplane.ProcessSupervisor import ProcessSupervisor


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
        gpu_access=GpuAccessPolicy("command", ("gpu", "run", "--")),
    )
    restored = ClusterProfile.from_mapping("claimed", claimed.to_dict())
    assert restored.gpu_access.command_prefix == ("gpu", "run", "--")


def test_gpu_access_rejects_ambiguous_or_incompatible_policies() -> None:
    with pytest.raises(ValueError, match="requires command_prefix"):
        GpuAccessPolicy("command")
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
    assert ProcessSupervisor._acquire_gpus(
        tmp_path, identity, 1, allow_external_use=True
    ) == (0,)
