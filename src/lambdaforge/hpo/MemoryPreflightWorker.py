"""Pickle-safe child entry point for representative CUDA memory preflight."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from lambdaforge.experiments.ObjectFactory import ObjectFactory
from lambdaforge.hpo.CudaMemoryLimiter import CudaMemoryLimiter


class MemoryPreflightWorker:
    """Build and execute a consumer probe under a child-local allocator ceiling."""

    def __init__(self, spec: dict[str, Any], budget_bytes: int, output: Path) -> None:
        self.spec = spec
        self.budget_bytes = budget_bytes
        self.output = output

    def __call__(self, stop_event: Any) -> None:
        """Run the configured zero-argument probe and atomically publish peak telemetry."""
        del stop_event
        fraction = CudaMemoryLimiter().apply(self.budget_bytes)
        probe = ObjectFactory.build(self.spec)
        if not callable(probe):
            raise TypeError("The configured HPO memory probe must be callable.")
        probe()
        torch.cuda.synchronize()
        self._write(
            {
                "status": "ok",
                "allocator_fraction": fraction,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            }
        )

    def _write(self, payload: dict[str, Any]) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_name(f".{self.output.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.output)
        finally:
            temporary.unlink(missing_ok=True)
