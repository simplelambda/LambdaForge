"""First-class logical dataset registry record."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from lambdaforge.data.DatasetPlacement import DatasetPlacement
from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    """Describe one immutable version and every known physical placement."""

    name: str
    version: str
    dataset_id: str
    sample_count: int
    splits: Mapping[str, int]
    created_at_utc: str
    placements: tuple[DatasetPlacement, ...] = ()
    producer: Mapping[str, Any] = field(default_factory=dict)
    lineage: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "splits", FrozenJsonMapping(self.splits))
        object.__setattr__(self, "producer", FrozenJsonMapping(self.producer))
        object.__setattr__(self, "metadata", FrozenJsonMapping(self.metadata))

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DatasetRecord:
        raw_placements = value.get("placements", ())
        if not isinstance(raw_placements, Sequence) or isinstance(raw_placements, (str, bytes)):
            raise TypeError("Dataset placements must be a sequence.")
        return cls(
            str(value["name"]),
            str(value["version"]),
            str(value["dataset_id"]),
            int(value.get("sample_count", 0)),
            value.get("splits", {}),
            str(value["created_at_utc"]),
            tuple(DatasetPlacement.from_mapping(item) for item in raw_placements),
            value.get("producer", {}),
            tuple(str(item) for item in value.get("lineage", ())),
            value.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_record_version": 1,
            "name": self.name,
            "version": self.version,
            "dataset_id": self.dataset_id,
            "sample_count": self.sample_count,
            "splits": copy.deepcopy(self.splits),
            "created_at_utc": self.created_at_utc,
            "placements": [value.to_dict() for value in self.placements],
            "producer": copy.deepcopy(self.producer),
            "lineage": list(self.lineage),
            "metadata": copy.deepcopy(self.metadata),
        }
