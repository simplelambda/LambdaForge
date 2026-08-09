"""Validated resource policy for experiment execution."""

from __future__ import annotations

import copy
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

from lambdaforge.experiments.ExecutionMode import ExecutionMode
from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
from lambdaforge.training.orchestration.DeviceAssignment import DeviceAssignment


@dataclass(slots=True)
class ExecutionConfig:
    """Resolved scheduling and per-process resource limits.

    GPU indices are logical positions relative to the parent process's
    ``CUDA_VISIBLE_DEVICES``. ``None`` resource limits preserve the current
    environment; positive limits are applied inside each spawned job.
    """

    mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    gpus: list[int] | None = None
    jobs_per_gpu: int = 1
    cpu_jobs: int = 1
    devices_per_job: int = 1
    grace_seconds: float = 15.0
    cpu_threads_per_job: int | None = 1
    cpu_interop_threads_per_job: int | None = 1
    cpu_cores_per_job: int | None = 1
    dataloader_num_workers_per_job: int | None = 0

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any],
        overrides: Mapping[str, Any] | None = None,
    ) -> ExecutionConfig:
        """Resolve settings with precedence ``overrides > YAML > defaults``."""
        block = ExperimentConfig.get_value(config, "execution", {}) or {}
        if not isinstance(block, Mapping):
            raise TypeError("'execution' must be a mapping.")
        supplied = {key: value for key, value in (overrides or {}).items() if value is not None}

        def pick(key: str, default: Any) -> Any:
            return supplied[key] if key in supplied else block.get(key, default)

        raw_gpus = pick("gpus", None)
        if isinstance(raw_gpus, str):
            gpus = [
                cls._strict_int(token.strip(), "execution.gpus")
                for token in raw_gpus.split(",")
                if token.strip()
            ]
        elif raw_gpus is None:
            gpus = None
        else:
            normalized_gpus = DeviceAssignment.normalize(
                raw_gpus,
                label="execution.gpus",
            )
            gpus = list(normalized_gpus or ())

        instance = cls(
            mode=ExecutionMode(str(pick("mode", ExecutionMode.SEQUENTIAL.value))),
            gpus=gpus,
            jobs_per_gpu=cls._strict_int(
                pick("jobs_per_gpu", 1),
                "execution.jobs_per_gpu",
            ),
            cpu_jobs=cls._strict_int(pick("cpu_jobs", 1), "execution.cpu_jobs"),
            devices_per_job=cls._strict_int(
                pick("devices_per_job", 1),
                "execution.devices_per_job",
            ),
            grace_seconds=cls._finite_number(
                pick("grace_seconds", 15.0),
                "execution.grace_seconds",
            ),
            cpu_threads_per_job=cls._optional_int(
                pick("cpu_threads_per_job", 1),
                "execution.cpu_threads_per_job",
            ),
            cpu_interop_threads_per_job=cls._optional_int(
                pick("cpu_interop_threads_per_job", 1),
                "execution.cpu_interop_threads_per_job",
            ),
            cpu_cores_per_job=cls._optional_int(
                pick("cpu_cores_per_job", 1),
                "execution.cpu_cores_per_job",
            ),
            dataloader_num_workers_per_job=cls._optional_int(
                pick("dataloader_num_workers_per_job", 0),
                "execution.dataloader_num_workers_per_job",
            ),
        )
        instance.validate()
        return instance

    def validate(self) -> None:
        """Reject unsafe or internally inconsistent resource settings."""
        self.grace_seconds = self._finite_number(
            self.grace_seconds,
            "execution.grace_seconds",
        )
        if self.grace_seconds < 0:
            raise ValueError("execution.grace_seconds must be non-negative.")
        normalized_gpus = DeviceAssignment.normalize(
            self.gpus,
            label="execution.gpus",
        )
        self.gpus = None if normalized_gpus is None else list(normalized_gpus)
        self.jobs_per_gpu = self._strict_int(
            self.jobs_per_gpu,
            "execution.jobs_per_gpu",
        )
        self.devices_per_job = self._strict_int(
            self.devices_per_job,
            "execution.devices_per_job",
        )
        self.cpu_jobs = self._strict_int(self.cpu_jobs, "execution.cpu_jobs")
        if self.jobs_per_gpu < 1:
            raise ValueError("execution.jobs_per_gpu must be at least 1.")
        if self.devices_per_job < 1:
            raise ValueError("execution.devices_per_job must be at least 1.")
        if self.cpu_jobs < 1:
            raise ValueError("execution.cpu_jobs must be at least 1.")
        for key in (
            "cpu_threads_per_job",
            "cpu_interop_threads_per_job",
            "cpu_cores_per_job",
            "dataloader_num_workers_per_job",
        ):
            value = getattr(self, key)
            value = self._optional_int(value, f"execution.{key}")
            setattr(self, key, value)
            if value is not None and value < 0:
                raise ValueError(f"execution.{key} must be non-negative or null.")
            if key != "dataloader_num_workers_per_job" and value == 0:
                raise ValueError(f"execution.{key} must be positive or null.")
        if self.mode is ExecutionMode.SEQUENTIAL:
            return
        if self.mode is ExecutionMode.PARALLEL and not self.gpus:
            available = (
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else (os.cpu_count() or 1)
            )
            cores = self.cpu_cores_per_job or 1
            if self.cpu_jobs * cores > available:
                raise ValueError(
                    "CPU parallel execution oversubscribes available affinity: "
                    f"cpu_jobs={self.cpu_jobs}, cpu_cores_per_job={cores}, available={available}."
                )
            return
        if not self.gpus:
            raise ValueError(f"execution.mode={self.mode.value!r} requires a non-empty gpus list.")
        if self.mode is ExecutionMode.DDP and len(self.gpus) % self.devices_per_job:
            raise ValueError("The GPU count must be divisible by devices_per_job in DDP mode.")

    def slots(self) -> list[list[int]]:
        """Build the fixed logical-device slot pool used by the scheduler."""
        devices = self.gpus or []
        if self.mode is ExecutionMode.PARALLEL:
            if not devices:
                return [[] for _ in range(self.cpu_jobs)]
            return [[device] for device in devices for _ in range(self.jobs_per_gpu)]
        step = self.devices_per_job
        return [devices[index : index + step] for index in range(0, len(devices), step)]

    def patch_run(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """Return a subprocess-specific config without mutating the source."""
        patched = copy.deepcopy(dict(config))
        is_cpu = self.mode is ExecutionMode.PARALLEL and not self.gpus
        device_count = self.devices_per_job if self.mode is ExecutionMode.DDP else 1
        ExperimentConfig.set_value(patched, "trainer.accelerator", "cpu" if is_cpu else "gpu")
        ExperimentConfig.set_value(patched, "trainer.devices", device_count)
        ExperimentConfig.set_value(
            patched,
            "trainer.strategy",
            "ddp" if self.mode is ExecutionMode.DDP else "auto",
        )
        ExperimentConfig.set_value(patched, "trainer.enable_progress_bar", False)
        if self.dataloader_num_workers_per_job is not None:
            workers = self.dataloader_num_workers_per_job
            ExperimentConfig.set_value(
                patched,
                "data.datamodule.params.num_workers",
                workers,
            )
            if workers == 0:
                ExperimentConfig.set_value(
                    patched,
                    "data.datamodule.params.persistent_workers",
                    False,
                )
        return patched

    @staticmethod
    def _optional_int(value: Any, name: str) -> int | None:
        if value is None or str(value).strip().lower() in {"", "none", "null"}:
            return None
        return ExecutionConfig._strict_int(value, name)

    @staticmethod
    def _strict_int(value: Any, name: str) -> int:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be an integer, not a boolean.")
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError as error:
                raise TypeError(f"{name} must be an integer.") from error
        if isinstance(value, int):
            return value
        if isinstance(value, Real):
            resolved = float(value)
            if not math.isfinite(resolved):
                raise ValueError(f"{name} must be finite.")
            if not resolved.is_integer():
                raise ValueError(f"{name} must be an integer without a fractional part.")
            return int(resolved)
        raise TypeError(f"{name} must be an integer.")

    @staticmethod
    def _finite_number(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a finite number, not a boolean or string.")
        resolved = float(value)
        if not math.isfinite(resolved):
            raise ValueError(f"{name} must be finite.")
        return resolved
