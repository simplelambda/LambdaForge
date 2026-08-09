"""Portable resource declaration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    """Declare capacity needed by one schedulable unit."""

    cpu_cores: int = 1
    ram_bytes: int = 0
    gpu_count: int = 0
    gpu_memory_bytes: int = 0
    runtime_seconds: float | None = None
    storage_bytes: int = 0

    def __post_init__(self) -> None:
        for name in ("cpu_cores", "ram_bytes", "gpu_count", "gpu_memory_bytes", "storage_bytes"):
            value = getattr(self, name)
            minimum = 1 if name == "cpu_cores" else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if self.runtime_seconds is not None and self.runtime_seconds <= 0:
            raise ValueError("runtime_seconds must be positive or null.")
