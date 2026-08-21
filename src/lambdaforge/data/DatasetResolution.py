"""Exact result of resolving one logical dataset reference."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from lambdaforge.data.DatasetLocation import DatasetLocation
from lambdaforge.data.DatasetPlacement import DatasetPlacement
from lambdaforge.data.DatasetRecord import DatasetRecord
from lambdaforge.data.DatasetReference import DatasetReference


@dataclass(frozen=True, slots=True)
class DatasetResolution:
    """Pin a logical version/content identity while recording the placement used."""

    reference: DatasetReference
    exact_reference: str
    location: DatasetLocation
    identity: Mapping[str, Any]
    descriptor: Mapping[str, Any] = field(default_factory=dict)
    record: DatasetRecord | None = None
    managed: bool = False

    def to_binding(self, *, path: str) -> dict[str, Any]:
        """Return reproducibility evidence with physical placement outside identity."""
        return {
            "path": path,
            "reference": str(self.reference),
            "exact_reference": self.exact_reference,
            "identity": copy.deepcopy(dict(self.identity)),
            "resolved_path": self.location.uri,
            "environment": self.location.environment,
            "managed": self.managed,
        }


class DatasetPlacementState(str, Enum):
    """Describe observed target state without confusing uncertainty with absence."""

    AVAILABLE = "available"
    REGISTERED_BUT_MISSING = "registered_but_missing"
    DISCOVERED_UNREGISTERED = "discovered_unregistered"
    ABSENT = "absent"
    CONFLICT = "conflict"
    UNREACHABLE = "unreachable"


@dataclass(frozen=True, slots=True)
class DatasetPlacementResolution:
    """Combine immutable identity, target indexes and a bounded manifest observation."""

    record: DatasetRecord
    cluster: str
    state: DatasetPlacementState
    placement: DatasetPlacement | None = None
    controller_placement: DatasetPlacement | None = None
    target_placement: DatasetPlacement | None = None
    physical: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""
    repair: str | None = None

    @property
    def physically_available(self) -> bool:
        """Return whether an exact manifest-backed copy was observed on the target."""
        return self.state in {
            DatasetPlacementState.AVAILABLE,
            DatasetPlacementState.DISCOVERED_UNREGISTERED,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a stable machine-readable consistency view."""
        return {
            "dataset": self.record.key,
            "dataset_id": self.record.dataset_id,
            "cluster": self.cluster,
            "state": self.state.value,
            "placement": self.placement.to_dict() if self.placement is not None else None,
            "controller_placement": (
                self.controller_placement.to_dict()
                if self.controller_placement is not None
                else None
            ),
            "target_placement": (
                self.target_placement.to_dict()
                if self.target_placement is not None
                else None
            ),
            "physical": copy.deepcopy(dict(self.physical)),
            "reason": self.reason,
            "repair": self.repair,
        }
