"""Strict configuration produced from a user-facing authoring document."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lambdaforge.configuration.ConfigurationKind import ConfigurationKind


@dataclass(frozen=True, slots=True)
class MaterializedConfig:
    """Hold normalized runner input and explainable authoring metadata."""

    kind: ConfigurationKind
    values: Mapping[str, Any]
    source: Path | None = None
    authoring_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON/YAML-compatible strict configuration."""
        return copy.deepcopy(dict(self.values))

    def explanation(self) -> dict[str, Any]:
        """Return the stable envelope shown by ``inspect --resolved``."""
        return {
            "authoring_version": self.authoring_version,
            "kind": self.kind.value,
            "source": str(self.source) if self.source is not None else None,
            "materialized": self.to_dict(),
        }
