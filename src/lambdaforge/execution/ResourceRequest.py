"""Portable resource declaration."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    """Declare capacity needed by one schedulable unit."""

    cpu_cores: int = 1
    ram_bytes: int = 0
    gpu_count: int = 0
    gpu_memory_bytes: int = 0
    runtime_seconds: float | None = None
    storage_bytes: int = 0
    processes: int = 1

    def __post_init__(self) -> None:
        for name in (
            "cpu_cores",
            "ram_bytes",
            "gpu_count",
            "gpu_memory_bytes",
            "storage_bytes",
            "processes",
        ):
            value = getattr(self, name)
            minimum = 1 if name in {"cpu_cores", "processes"} else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if self.runtime_seconds is not None and self.runtime_seconds <= 0:
            raise ValueError("runtime_seconds must be positive or null.")
        if self.processes > self.cpu_cores:
            raise ValueError("processes cannot exceed the requested cpu_cores.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> ResourceRequest:
        """Parse the user-facing portable ``resources`` mapping."""
        data = dict(value or {})
        aliases = {
            "cpus": "cpu_cores",
            "memory": "ram_bytes",
            "gpus": "gpu_count",
            "gpu_memory": "gpu_memory_bytes",
            "time": "runtime_seconds",
            "storage": "storage_bytes",
        }
        normalized = {aliases.get(str(key), str(key)): item for key, item in data.items()}
        for key in ("ram_bytes", "gpu_memory_bytes", "storage_bytes"):
            if isinstance(normalized.get(key), str):
                normalized[key] = cls._bytes(str(normalized[key]))
        if isinstance(normalized.get("runtime_seconds"), str):
            normalized["runtime_seconds"] = cls._duration(str(normalized["runtime_seconds"]))
        unknown = set(normalized) - {
            "cpu_cores",
            "ram_bytes",
            "gpu_count",
            "gpu_memory_bytes",
            "runtime_seconds",
            "storage_bytes",
            "processes",
        }
        if unknown:
            raise ValueError(f"Unknown resource keys: {sorted(unknown)}.")
        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        """Return a scheduler-neutral resource descriptor."""
        return {
            "cpu_cores": self.cpu_cores,
            "ram_bytes": self.ram_bytes,
            "gpu_count": self.gpu_count,
            "gpu_memory_bytes": self.gpu_memory_bytes,
            "runtime_seconds": self.runtime_seconds,
            "storage_bytes": self.storage_bytes,
            "processes": self.processes,
        }

    @staticmethod
    def _bytes(value: str) -> int:
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]i?B|B)\s*", value, re.I)
        if match is None:
            raise ValueError(f"Invalid byte quantity: {value!r}.")
        amount, unit = match.groups()
        powers = {
            "B": 1,
            "KB": 1000,
            "MB": 1000**2,
            "GB": 1000**3,
            "TB": 1000**4,
            "KIB": 1024,
            "MIB": 1024**2,
            "GIB": 1024**3,
            "TIB": 1024**4,
        }
        return int(float(amount) * powers[unit.upper()])

    @staticmethod
    def _duration(value: str) -> float:
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd])\s*", value, re.I)
        if match is None:
            raise ValueError(f"Invalid duration: {value!r}; use s, m, h or d.")
        amount, unit = match.groups()
        return float(amount) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit.lower()]
