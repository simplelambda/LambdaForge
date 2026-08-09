"""Explicit memory-capacity states."""

from enum import Enum


class MemoryCapacityKind(str, Enum):
    """Distinguish unavailable knowledge from absence of a limit."""

    UNKNOWN = "unknown"
    UNBOUNDED = "unbounded"
    KNOWN = "known"
