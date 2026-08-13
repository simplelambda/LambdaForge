"""Explicit preview/apply result for internal cache collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class StorageGcPlan:
    """Describe exact reconstructible cache candidates without hiding application state."""

    cluster: str
    candidates: tuple[dict[str, Any], ...]
    reclaimable_bytes: int
    applied: bool = False
    blocked_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster": self.cluster,
            "candidates": list(self.candidates),
            "reclaimable_bytes": self.reclaimable_bytes,
            "applied": self.applied,
            "blocked_reason": self.blocked_reason,
            "protected": ["datasets", "results", "checkpoints", "active job workspaces"],
        }
