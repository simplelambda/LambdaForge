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
    reference_issues: tuple[dict[str, Any], ...] = ()
    environment_count: int = 0
    environment_usage: dict[str, Any] | None = None
    reclaimable_allocated_bytes: int = 0
    outcome: str = "preview"
    hypothetical_after_repairs: bool = False
    observed_free_bytes_delta: dict[str, int] | None = None

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
            "reference_issues": list(self.reference_issues),
            "environment_count": self.environment_count,
            "environment_usage": self.environment_usage or {},
            "reclaimable_allocated_bytes": self.reclaimable_allocated_bytes,
            "outcome": self.outcome,
            "hypothetical_after_repairs": self.hypothetical_after_repairs,
            "observed_free_bytes_delta": self.observed_free_bytes_delta or {},
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
