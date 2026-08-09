"""One logical resource available to adaptive scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field

from lambdaforge.hpo.MemoryCapacity import MemoryCapacity
from lambdaforge.hpo.MemoryCapacityKind import MemoryCapacityKind


@dataclass(frozen=True, slots=True)
class AdaptiveResource:
    """Describe a CPU lane or logical visible GPU without physical-ID assumptions."""

    name: str
    device: int | None
    memory_capacity_bytes: int | None = None
    cpu_cores: int = 1
    max_jobs: int = 1
    memory_capacity_kind: MemoryCapacityKind | None = None
    memory_capacity: MemoryCapacity = field(init=False)

    def __post_init__(self) -> None:
        if not self.name or self.cpu_cores < 1 or self.max_jobs < 1:
            raise ValueError("Invalid adaptive resource declaration.")
        if self.device is not None and self.device < 0:
            raise ValueError("Adaptive GPU device indices must be non-negative logical indices.")
        if self.memory_capacity_bytes is not None and self.memory_capacity_bytes < 0:
            raise ValueError("Adaptive memory capacity cannot be negative.")
        kind = self.memory_capacity_kind
        if kind is None:
            if self.memory_capacity_bytes is not None:
                kind = MemoryCapacityKind.KNOWN
            elif self.device is None:
                kind = MemoryCapacityKind.UNBOUNDED
            else:
                kind = MemoryCapacityKind.UNKNOWN
        capacity = (
            MemoryCapacity.known(self.memory_capacity_bytes or 0)
            if kind is MemoryCapacityKind.KNOWN
            else MemoryCapacity.unbounded()
            if kind is MemoryCapacityKind.UNBOUNDED
            else MemoryCapacity.unknown()
        )
        object.__setattr__(self, "memory_capacity_kind", kind)
        object.__setattr__(self, "memory_capacity", capacity)
