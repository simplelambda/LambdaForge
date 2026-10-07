"""Explicit preview/apply result for internal cache collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class StorageGcPlan:
    """Describe exact reconstructible or safely redundant storage candidates."""

    cluster: str
    candidates: tuple[dict[str, Any], ...]
    reclaimable_bytes: int
    applied: bool = False
    blocked_reason: str | None = None
    categories: dict[str, Any] | None = None
    protected_items: tuple[dict[str, Any], ...] = ()
    reclaimed_bytes: int = 0
    quota_unresolved_bytes: int = 0
    filesystems_before: dict[str, Any] | None = None
    filesystems_after: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster": self.cluster,
            "candidates": list(self.candidates),
            "reclaimable_bytes": self.reclaimable_bytes,
            "applied": self.applied,
            "blocked_reason": self.blocked_reason,
            "categories": self.categories or {},
            "protected_items": list(self.protected_items),
            "reclaimed_bytes": self.reclaimed_bytes,
            "quota_unresolved_bytes": self.quota_unresolved_bytes,
            "filesystems_before": self.filesystems_before or {},
            "filesystems_after": self.filesystems_after or {},
            "protected": [
                "published datasets",
                "results",
                "required, recoverable and pinned checkpoints",
                "active job workspaces",
                "Python runtimes referenced by active jobs or retained environments",
                "stage cache while a dataset build is active",
                "logs, metrics and unpublished successful artifacts",
            ],
        }
