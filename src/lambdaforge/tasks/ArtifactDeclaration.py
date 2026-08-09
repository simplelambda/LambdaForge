"""Unmaterialized artifact declaration returned by a task."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping
from lambdaforge.tasks.ArtifactType import ArtifactType


@dataclass(frozen=True, slots=True)
class ArtifactDeclaration:
    """Declare a run-relative output for safe hashing after task completion."""

    path: str | Path
    kind: ArtifactType | str = ArtifactType.OTHER
    media_type: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        relative = Path(self.path)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("Artifact paths must be non-empty and relative to the task run.")
        object.__setattr__(self, "path", relative.as_posix())
        object.__setattr__(self, "kind", ArtifactType(self.kind))
        if self.media_type is not None and not str(self.media_type).strip():
            raise ValueError("Artifact media_type cannot be empty when provided.")
        object.__setattr__(self, "media_type", str(self.media_type) if self.media_type else None)
        object.__setattr__(self, "metadata", FrozenJsonMapping(self.metadata))

    @classmethod
    def from_value(
        cls, value: ArtifactDeclaration | str | Path | Mapping[str, Any]
    ) -> ArtifactDeclaration:
        """Normalize a concise path or mapping into an artifact declaration."""
        if isinstance(value, cls):
            return value
        if isinstance(value, (str, Path)):
            return cls(path=value)
        if not isinstance(value, Mapping):
            raise TypeError("Artifacts must be paths, mappings or ArtifactDeclaration objects.")
        if "path" not in value:
            raise ValueError("Artifact mappings require a 'path'.")
        unexpected = set(value) - {"path", "kind", "media_type", "metadata"}
        if unexpected:
            raise ValueError(f"Unexpected artifact declaration keys: {sorted(unexpected)}.")
        return cls(
            path=value["path"],
            kind=value.get("kind", ArtifactType.OTHER),
            media_type=value.get("media_type"),
            metadata=value.get("metadata", {}),
        )
