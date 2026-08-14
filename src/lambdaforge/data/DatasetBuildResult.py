"""Durable outcome of one dataset recipe build."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from lambdaforge.data.DatasetRecord import DatasetRecord


@dataclass(frozen=True, slots=True)
class DatasetBuildResult:
    """Keep stage execution evidence separate from the published DatasetVersion."""

    build_id: str
    dataset: str
    status: str
    stages: Mapping[str, Mapping[str, Any]]
    record: DatasetRecord | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return durable build evidence."""
        return {
            "kind": "dataset-build",
            "build_id": self.build_id,
            "dataset": self.dataset,
            "status": self.status,
            "stages": copy.deepcopy(dict(self.stages)),
            "record": self.record.to_dict() if self.record is not None else None,
            "error": self.error,
        }
