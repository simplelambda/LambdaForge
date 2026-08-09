"""Pickle-safe child entry point for representative CUDA memory preflight."""

from __future__ import annotations

import inspect
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

    def __init__(
        self,
        spec: dict[str, Any],
        budget_bytes: int,
        output: Path,
        configuration: dict[str, Any] | None = None,
        resource_context: dict[str, Any] | None = None,
    ) -> None:
        self.spec = spec
        self.budget_bytes = budget_bytes
        self.output = output
        self.configuration = dict(configuration or {})
        self.resource_context = dict(resource_context or {})

    def __call__(self, stop_event: Any) -> None:
        """Run the configured zero-argument probe and atomically publish peak telemetry."""
        del stop_event
        fraction = CudaMemoryLimiter().apply(self.budget_bytes)
        probe = ObjectFactory.build(self.spec)
        if not callable(probe):
            raise TypeError("The configured HPO memory probe must be callable.")
        try:
            signature = inspect.signature(probe)
            positional = [
                parameter
                for parameter in signature.parameters.values()
                if parameter.kind
                in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
            ]
            variadic = any(
                parameter.kind is inspect.Parameter.VAR_POSITIONAL
                for parameter in signature.parameters.values()
            )
            if variadic or len(positional) >= 2:
                probe(self.configuration, self.resource_context)
            elif positional:
                probe(self.configuration)
            else:
                probe()
            torch.cuda.synchronize()
            self._write(
                {
                    "status": "ok",
                    "allocator_fraction": fraction,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                    "candidate_aware": bool(positional or variadic),
                }
            )
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            oom = "out of memory" in message.lower() or "cuda oom" in message.lower()
            self._write(
                {
                    "status": "oom" if oom else "failed",
                    "error": message,
                    "allocator_fraction": fraction,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                    "lower_bound_bytes": self.budget_bytes if oom else None,
                    "candidate_aware": bool(positional or variadic),
                }
            )
            raise

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
