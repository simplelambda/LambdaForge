"""Optional isolated representative CUDA memory preflight."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lambdaforge.hpo.MemoryPreflightWorker import MemoryPreflightWorker
from lambdaforge.training.orchestration.TrainingJob import TrainingJob
from lambdaforge.training.orchestration.TrainingOrchestrator import TrainingOrchestrator


class TorchMemoryPreflight:
    """Run a consumer-supplied representative step once per logical GPU in children.

    The object built from ``probe_spec`` must be callable without arguments. It owns creation of a
    representative model, batch, forward, backward and optimizer step. LambdaForge applies the
    declared allocator ceiling first and records peak public PyTorch allocator counters.
    """

    def run(
        self,
        probe_spec: dict[str, Any],
        *,
        devices: list[int],
        budget_bytes: int,
        output_dir: str | Path,
        grace_seconds: float = 15.0,
        configuration: dict[str, Any] | None = None,
        resource_context: dict[str, Any] | None = None,
        label: str = "probe",
    ) -> dict[int, dict[str, Any]]:
        """Execute isolated probes and fail if any process or telemetry reports failure."""
        if not devices:
            raise ValueError("CUDA memory preflight requires explicit execution.gpus.")
        directory = Path(output_dir)
        paths = {device: directory / f"{label}-gpu-{device}.json" for device in devices}
        jobs = [
            TrainingJob(
                f"memory-preflight-{device}",
                MemoryPreflightWorker(
                    probe_spec,
                    budget_bytes,
                    paths[device],
                    configuration,
                    {**(resource_context or {}), "logical_device": device},
                ),
            )
            for device in devices
        ]
        orchestrator = TrainingOrchestrator(grace_seconds=grace_seconds)
        codes = orchestrator.run_scheduled(jobs, [[device] for device in devices])
        output: dict[int, dict[str, Any]] = {}
        for device in devices:
            name = f"memory-preflight-{device}"
            if not paths[device].exists():
                raise RuntimeError(f"CUDA memory preflight failed on logical GPU {device}.")
            value = json.loads(paths[device].read_text(encoding="utf-8"))
            if not isinstance(value, dict) or value.get("status") not in {"ok", "oom"}:
                raise RuntimeError(f"Invalid CUDA memory preflight result for GPU {device}.")
            if codes.get(name) == 0 and value.get("status") != "ok":
                raise RuntimeError(f"Inconsistent CUDA memory preflight result for GPU {device}.")
            output[device] = value
        return output
