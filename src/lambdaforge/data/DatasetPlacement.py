"""Physical placement of one immutable logical dataset version."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DatasetPlacement:
    """Bind dataset identity to one cluster/environment and registered root."""

    cluster: str
    root: str
    registered_at_utc: str
    size_bytes: int | None = None
    file_count: int | None = None
    verified: bool | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DatasetPlacement:
        return cls(
            str(value["cluster"]),
            str(value["root"]),
            str(value["registered_at_utc"]),
            int(value["size_bytes"]) if value.get("size_bytes") is not None else None,
            int(value["file_count"]) if value.get("file_count") is not None else None,
            bool(value["verified"]) if value.get("verified") is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster": self.cluster,
            "root": self.root,
            "registered_at_utc": self.registered_at_utc,
            "size_bytes": self.size_bytes,
            "file_count": self.file_count,
            "verified": self.verified,
        }
