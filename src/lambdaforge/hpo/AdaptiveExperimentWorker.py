"""Pickle-safe adaptive training process entry point."""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
from lambdaforge.experiments.RunResult import RunResult
from lambdaforge.experiments.RunStatus import RunStatus
from lambdaforge.hpo.CudaMemoryLimiter import CudaMemoryLimiter


class AdaptiveExperimentWorker:
    """Train through the normal runner and publish child-local CUDA telemetry."""

    def __init__(self, config: dict[str, Any], memory_budget_bytes: int = 0) -> None:
        self.config = config
        self.memory_budget_bytes = int(memory_budget_bytes)

    def __call__(self, stop_event: Any) -> None:
        from lambdaforge.experiments.ExperimentRunner import ExperimentRunner
        from lambdaforge.experiments.StdIOCapture import StdIOCapture

        runner = ExperimentRunner()
        run_dir = runner.experiment_run_dir(self.config)
        run_dir.mkdir(parents=True, exist_ok=True)
        fraction = 0.0
        allocator_error: str | None = None
        with StdIOCapture(run_dir / "train.log", echo=False):
            try:
                if self.memory_budget_bytes > 0:
                    try:
                        fraction = CudaMemoryLimiter().apply(self.memory_budget_bytes)
                    except Exception as error:
                        allocator_error = f"{type(error).__name__}: {error}"
                        raise
                runner._run_single_experiment_unlocked(self.config, stop_event=stop_event)
            except Exception:
                traceback.print_exc()
                RunResult(
                    name=ExperimentConfig.get_value(self.config, "experiment.name", "experiment"),
                    run_dir=run_dir,
                    variant=ExperimentConfig.get_value(self.config, "experiment.variant"),
                    seed=ExperimentConfig.get_value(self.config, "experiment.seed"),
                    status=RunStatus.FAILED,
                    error=traceback.format_exc().splitlines()[-1],
                ).write_json(run_dir / "result.json")
                raise
            finally:
                allocated = 0
                reserved = 0
                if torch.cuda.is_available():
                    allocated = int(torch.cuda.max_memory_allocated(0))
                    reserved = int(torch.cuda.max_memory_reserved(0))
                self._write_resource(
                    run_dir / "adaptive-resource.json",
                    {
                        "memory_budget_bytes": self.memory_budget_bytes,
                        "allocator_fraction": fraction,
                        "allocator_error": allocator_error,
                        "peak_allocated_bytes": allocated,
                        "peak_reserved_bytes": reserved,
                    },
                )

    @staticmethod
    def _write_resource(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
