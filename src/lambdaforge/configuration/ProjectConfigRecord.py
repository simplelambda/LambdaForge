"""Discovered project configuration descriptor."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ProjectConfigRecord:
    """Describe one YAML source without replacing it as the source of truth."""

    name: str
    kind: str
    path: Path
    valid: bool = True
    datasets: tuple[str, ...] = ()
    resources: dict[str, Any] = field(default_factory=dict)
    hpo_enabled: bool = False
    error: str | None = None
    active_jobs: tuple[str, ...] = ()
    active_clusters: tuple[str, ...] = ()
    last_result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "path": str(self.path),
            "valid": self.valid,
            "datasets": list(self.datasets),
            "resources": dict(self.resources),
            "hpo_enabled": self.hpo_enabled,
            "error": self.error,
            "active_jobs": list(self.active_jobs),
            "active_clusters": list(self.active_clusters),
            "last_result": dict(self.last_result) if self.last_result is not None else None,
        }
