"""One physical or provider-backed asset belonging to a logical dataset member."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping


@dataclass(frozen=True, slots=True)
class DatasetAsset:
    """Describe an asset without equating its physical path with scientific identity."""

    path: str
    kind: str = "file"
    sha256: str | None = None
    size_bytes: int | None = None
    media_type: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.path).strip():
            raise ValueError("Dataset asset paths/URIs cannot be empty.")
        if "://" not in self.path:
            relative = PurePosixPath(self.path)
            if relative.is_absolute() or not relative.parts or ".." in relative.parts:
                raise ValueError("Local dataset asset paths must be relative and contained.")
        if self.kind not in {"file", "directory", "record", "uri"}:
            raise ValueError("Dataset asset kind must be file, directory, record or uri.")
        if self.sha256 is not None:
            digest = self.sha256.removeprefix("sha256:")
            if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
                raise ValueError("Dataset asset sha256 must be a hexadecimal SHA-256.")
            object.__setattr__(self, "sha256", f"sha256:{digest}")
        if self.size_bytes is not None and (
            isinstance(self.size_bytes, bool) or self.size_bytes < 0
        ):
            raise ValueError("Dataset asset size_bytes must be non-negative.")
        object.__setattr__(self, "metadata", FrozenJsonMapping(self.metadata))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DatasetAsset:
        """Restore one asset descriptor from a persisted index record."""
        unexpected = set(value) - {
            "path",
            "kind",
            "sha256",
            "size_bytes",
            "media_type",
            "metadata",
        }
        if unexpected:
            raise ValueError(f"Unexpected dataset asset keys: {sorted(unexpected)}.")
        return cls(
            path=str(value["path"]),
            kind=str(value.get("kind", "file")),
            sha256=str(value["sha256"]) if value.get("sha256") is not None else None,
            size_bytes=(int(value["size_bytes"]) if value.get("size_bytes") is not None else None),
            media_type=(str(value["media_type"]) if value.get("media_type") is not None else None),
            metadata=value.get("metadata", {}),
        )

    def identity_dict(self) -> dict[str, Any]:
        """Return path-independent scientific asset identity."""
        return {
            "kind": self.kind,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "metadata": copy.deepcopy(self.metadata),
            **({"uri": self.path} if self.kind == "uri" and self.sha256 is None else {}),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the complete portable asset descriptor."""
        return {
            "path": self.path,
            "kind": self.kind,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "metadata": copy.deepcopy(self.metadata),
        }
