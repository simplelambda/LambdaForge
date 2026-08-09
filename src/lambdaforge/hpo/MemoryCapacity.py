"""Typed available-memory capacity."""

from __future__ import annotations

from dataclasses import dataclass

from lambdaforge.hpo.MemoryCapacityKind import MemoryCapacityKind


@dataclass(frozen=True, slots=True)
class MemoryCapacity:
    """Represent UNKNOWN, UNBOUNDED or a known non-negative byte count."""

    kind: MemoryCapacityKind
    bytes: int | None = None

    def __post_init__(self) -> None:
        if self.kind is MemoryCapacityKind.KNOWN:
            if self.bytes is None or isinstance(self.bytes, bool) or self.bytes < 0:
                raise ValueError("Known memory capacity requires non-negative bytes.")
        elif self.bytes is not None:
            raise ValueError("Unknown/unbounded memory capacity cannot carry a byte value.")

    @classmethod
    def unknown(cls) -> MemoryCapacity:
        """Return a capacity whose physical limit cannot be established."""
        return cls(MemoryCapacityKind.UNKNOWN)

    @classmethod
    def unbounded(cls) -> MemoryCapacity:
        """Return an explicitly unconstrained resource such as a CPU lane."""
        return cls(MemoryCapacityKind.UNBOUNDED)

    @classmethod
    def known(cls, value: int) -> MemoryCapacity:
        """Return an exact usable byte capacity, including a meaningful zero."""
        return cls(MemoryCapacityKind.KNOWN, value)

    def remaining(self, reserved: int) -> MemoryCapacity:
        """Subtract active reservations while preserving state semantics."""
        if reserved < 0:
            raise ValueError("Memory reservations cannot be negative.")
        if self.kind is MemoryCapacityKind.KNOWN:
            assert self.bytes is not None
            return MemoryCapacity.known(max(0, self.bytes - reserved))
        return self
