"""Typed plans, validation findings and outcomes for one dataset recipe build."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from lambdaforge.data.DatasetRecord import DatasetRecord


@dataclass(frozen=True, slots=True)
class DatasetStagePlan:
    """Explain whether one stage is reused, executed, missing or invalid."""

    stage: str
    action: str
    reason: str
    fingerprint: str | None = None
    required: bool = True

    def to_dict(self) -> dict[str, object]:
        """Return stable plan data for human and JSON renderers."""
        return {
            "stage": self.stage,
            "action": self.action,
            "reason": self.reason,
            "fingerprint": self.fingerprint,
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class DatasetBuildPlan:
    """Describe stage reuse and final publication before executing project code."""

    dataset: str
    recipe_fingerprint: str
    target_cluster: str
    stages: tuple[DatasetStagePlan, ...]
    publish_action: str
    publish_reason: str

    def to_dict(self) -> dict[str, object]:
        """Return the stable build-plan envelope."""
        return {
            "kind": "dataset-build-plan",
            "dataset": self.dataset,
            "recipe_fingerprint": self.recipe_fingerprint,
            "target_cluster": self.target_cluster,
            "stages": [stage.to_dict() for stage in self.stages],
            "publish": {"action": self.publish_action, "reason": self.publish_reason},
        }


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


@dataclass(frozen=True, slots=True)
class DatasetRecipeValidationReport:
    """Return all safe static recipe validation findings together."""

    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        """Return whether the recipe can be planned."""
        return not self.errors

    def summary(self) -> str:
        """Render a concise human validation result."""
        if self.is_valid:
            return "Dataset recipe validation: OK"
        return "Dataset recipe validation failed:\n- " + "\n- ".join(self.errors)
