"""Structured artifact inspection result."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ArtifactInspection:
    """Describe safe metadata, bounded previews and optional warnings."""

    artifact_type: str
    path: str
    size_bytes: int
    items: tuple[Mapping[str, Any], ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible inspection report."""
        return {
            "artifact_type": self.artifact_type,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "items": [copy.deepcopy(dict(item)) for item in self.items],
            "warnings": list(self.warnings),
        }
