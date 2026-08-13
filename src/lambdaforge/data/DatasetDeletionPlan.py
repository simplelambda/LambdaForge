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
        }
