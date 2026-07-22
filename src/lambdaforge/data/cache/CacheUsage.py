"""Atomic usage snapshot for a persistent cache namespace."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CacheUsage:
    """Store a coherent record count and byte total from one namespace scan."""

    entries: int
    bytes: int
