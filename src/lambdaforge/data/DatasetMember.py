"""Logical member of an immutable dataset version."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from lambdaforge.data.DatasetAsset import DatasetAsset
from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping


@dataclass(frozen=True, slots=True)
class DatasetMember:
    """Bind one stable member ID to partitions, targets, metadata and arbitrary assets."""

    member_id: str
    partitions: Mapping[str, Any] = field(default_factory=dict)
    targets: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    display: Mapping[str, Any] = field(default_factory=dict)
    assets: Mapping[str, DatasetAsset] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.member_id.strip() or re.search(r"[\r\n\x00]", self.member_id):
            raise ValueError("Dataset member IDs must be non-empty single-line strings.")
        normalized_assets = {
            str(name): (
                value if isinstance(value, DatasetAsset) else DatasetAsset.from_mapping(value)
            )
            for name, value in self.assets.items()
        }
        if any(not name.strip() for name in normalized_assets):
            raise ValueError("Dataset asset logical names cannot be empty.")
        object.__setattr__(self, "partitions", FrozenJsonMapping(self.partitions))
        object.__setattr__(self, "targets", FrozenJsonMapping(self.targets))
        object.__setattr__(self, "metadata", FrozenJsonMapping(self.metadata))
        object.__setattr__(self, "display", FrozenJsonMapping(self.display))
        object.__setattr__(self, "assets", normalized_assets)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DatasetMember:
        """Parse one canonical index record."""
        unexpected = set(value) - {
            "id",
            "partitions",
            "targets",
            "metadata",
            "display",
            "assets",
        }
        if unexpected:
            raise ValueError(f"Unexpected dataset member keys: {sorted(unexpected)}.")
        assets = value.get("assets", {})
        if not isinstance(assets, Mapping):
            raise TypeError("Dataset member assets must be a mapping.")
        return cls(
            member_id=str(value["id"]),
            partitions=value.get("partitions", {}),
            targets=value.get("targets", {}),
            metadata=value.get("metadata", {}),
            display=value.get("display", {}),
            assets={str(name): DatasetAsset.from_mapping(item) for name, item in assets.items()},
        )

    def identity_dict(self) -> dict[str, Any]:
        """Return only scientifically identity-bearing member fields."""
        return {
            "id": self.member_id,
            "partitions": copy.deepcopy(self.partitions),
            "targets": copy.deepcopy(self.targets),
            "metadata": copy.deepcopy(self.metadata),
            "assets": {name: asset.identity_dict() for name, asset in sorted(self.assets.items())},
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the full persisted member record."""
        return {
            "id": self.member_id,
            "partitions": copy.deepcopy(self.partitions),
            "targets": copy.deepcopy(self.targets),
            "metadata": copy.deepcopy(self.metadata),
            "display": copy.deepcopy(self.display),
            "assets": {name: asset.to_dict() for name, asset in sorted(self.assets.items())},
        }
