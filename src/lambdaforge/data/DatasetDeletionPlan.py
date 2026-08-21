"""Preview-first physical dataset deletion plan."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DatasetDeletionPlan:
    """Describe one verified physical deletion while keeping preview as the default."""

    dataset: str
    cluster: str
    root: str
    size_bytes: int | None
    file_count: int | None
    safe: bool
    reasons: tuple[str, ...]
    applied: bool = False
    action: str = "DELETE_PLACEMENT"
    dataset_id: str | None = None
    placement_state: str = "available"
    active_consumers: tuple[str, ...] = ()
    logical_version_preserved: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "cluster": self.cluster,
            "root": self.root,
            "size_bytes": self.size_bytes,
            "file_count": self.file_count,
            "safe": self.safe,
            "reasons": list(self.reasons),
            "applied": self.applied,
            "action": self.action,
            "dataset_id": self.dataset_id,
            "placement_state": self.placement_state,
            "active_consumers": list(self.active_consumers),
            "logical_version_preserved": self.logical_version_preserved,
        }
