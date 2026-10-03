"""Conservative preflight: only a live, unambiguous direct-host inventory can reject work."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.Transport import Transport


def validate_gpu_capacity(requested: int, inventory: Mapping[str, Any]) -> None:
    """Reject impossible requests, never confuse occupancy or unknown grants with capacity."""
    capacity = inventory.get("allocatable_gpus")
    if (
        inventory.get("reliable") is True
        and isinstance(capacity, int)
        and not isinstance(capacity, bool)
        and capacity >= 0
        and requested > capacity
    ):
        raise ValueError(
            f"Requested {requested} GPUs, but this execution target exposes only {capacity}. "
            "Reduce resources.gpu or select another target."
        )


def check_target_capacity(
    profile: ClusterProfile, transport: Transport, requested: int
) -> dict[str, Any]:
    """Observe direct capacity without initializing CUDA or claiming additional devices.

    SLURM login devices are not allocation capacity. Site-command grants may intentionally be
    smaller than an adaptive request and are checked by the inherited-token runtime contract.
    """
    unknown: dict[str, Any] = {"reliable": False, "allocatable_gpus": None}
    if requested == 0 or profile.gpu_access.effective_mode(profile.scheduler) in {
        "command",
        "scheduler",
    }:
        return unknown
    code = (
        "import json,os,subprocess\n"
        "p=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],"
        "capture_output=True,text=True,timeout=4)\n"
        "devices=[v.strip() for v in p.stdout.splitlines() if v.strip()]\n"
        "grant=os.environ.get('CUDA_VISIBLE_DEVICES')\n"
        "tokens=([v.strip() for v in grant.split(',') if v.strip()] "
        "if grant is not None else devices)\n"
        "valid=p.returncode==0 and all(v.startswith('GPU-') for v in devices)\n"
        "if grant is not None:\n"
        " valid=valid and len(tokens)==len(set(tokens)) and all("
        "v in devices or (v.isdigit() and int(v)<len(devices)) for v in tokens)\n"
        "print(json.dumps({'reliable':valid,'allocatable_gpus':len(tokens) if valid else None}))\n"
    )
    try:
        result = transport.run((*profile.command_prefix, profile.python, "-c", code), timeout=6.0)
        parsed = json.loads(result.stdout) if result.returncode == 0 else unknown
        inventory = dict(parsed) if isinstance(parsed, Mapping) else unknown
    except (OSError, RuntimeError, ValueError, TimeoutError):
        inventory = unknown
    validate_gpu_capacity(requested, inventory)
    return inventory
