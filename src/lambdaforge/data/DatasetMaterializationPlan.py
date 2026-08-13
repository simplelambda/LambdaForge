"""Immutable preview of dataset placement work."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DatasetMaterializationPlan:
    """Explain NOOP, REPLICATE or BUILD without silently moving data."""

    dataset: str
    target_cluster: str
    action: str
    source_cluster: str | None = None
    producer: str | None = None
    estimated_bytes: int | None = None
    reason: str = ""
    requires_controller_online: bool = False
    prerequisites: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "target_cluster": self.target_cluster,
            "action": self.action,
            "source_cluster": self.source_cluster,
            "producer": self.producer,
            "estimated_bytes": self.estimated_bytes,
            "reason": self.reason,
            "requires_controller_online": self.requires_controller_online,
            "prerequisites": [dict(value) for value in self.prerequisites],
        }
