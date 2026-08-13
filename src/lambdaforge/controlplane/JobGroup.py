"""Persistent group for one explicit multi-cluster submission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class JobGroup:
    """Keep related independent jobs visible without claiming distributed execution."""

    group_id: str
    name: str
    job_ids: tuple[str, ...]
    clusters: tuple[str, ...]
    created_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_group_version": 1,
            "group_id": self.group_id,
            "name": self.name,
            "job_ids": list(self.job_ids),
            "clusters": list(self.clusters),
            "created_at_utc": self.created_at_utc,
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> JobGroup:
        return cls(
            str(value["group_id"]),
            str(value["name"]),
            tuple(str(item) for item in value.get("job_ids", ())),
            tuple(str(item) for item in value.get("clusters", ())),
            str(value["created_at_utc"]),
        )
