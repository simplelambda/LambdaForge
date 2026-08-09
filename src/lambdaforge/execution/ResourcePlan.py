"""Immutable resource-packing plan."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResourcePlan:
    """Describe deterministic waves and aggregate resource estimates."""

    waves: tuple[tuple[str, ...], ...]
    peak_cpu_cores: int
    peak_ram_bytes: int
    peak_gpu_count: int
    storage_bytes: int
    estimated_runtime_seconds: float | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "waves": [list(wave) for wave in self.waves],
            "peak_cpu_cores": self.peak_cpu_cores,
            "peak_ram_bytes": self.peak_ram_bytes,
            "peak_gpu_count": self.peak_gpu_count,
            "storage_bytes": self.storage_bytes,
            "estimated_runtime_seconds": self.estimated_runtime_seconds,
        }
