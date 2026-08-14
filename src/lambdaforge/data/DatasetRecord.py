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
    build_id: str | None = None
    index: Mapping[str, Any] = field(default_factory=dict)
    partitions: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    target_schema: Mapping[str, Any] = field(default_factory=dict)
    global_assets: Mapping[str, Any] = field(default_factory=dict)
    lineage_graph: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "splits", FrozenJsonMapping(self.splits))
        object.__setattr__(self, "producer", FrozenJsonMapping(self.producer))
        object.__setattr__(self, "metadata", FrozenJsonMapping(self.metadata))
        object.__setattr__(self, "index", FrozenJsonMapping(self.index))
        object.__setattr__(self, "partitions", FrozenJsonMapping(self.partitions))
        object.__setattr__(self, "target_schema", FrozenJsonMapping(self.target_schema))
        object.__setattr__(self, "global_assets", FrozenJsonMapping(self.global_assets))
        object.__setattr__(self, "lineage_graph", FrozenJsonMapping(self.lineage_graph))

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
            str(value["build_id"]) if value.get("build_id") is not None else None,
            value.get("index", {}),
            value.get("partitions", {}),
            value.get("target_schema", {}),
            value.get("global_assets", {}),
            value.get("lineage_graph", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_record_version": 2,
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
            "content_id": self.dataset_id,
            "build_id": self.build_id,
            "index": copy.deepcopy(self.index),
            "partitions": copy.deepcopy(self.partitions),
            "target_schema": copy.deepcopy(self.target_schema),
            "global_assets": copy.deepcopy(self.global_assets),
            "lineage_graph": copy.deepcopy(self.lineage_graph),
        }
