"""Immutable descriptor for one applied configuration migration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lambdaforge.experiments.migrations.ExperimentSchemaVersion import (
    ExperimentSchemaVersion,
)


@dataclass(frozen=True, slots=True)
class ExperimentConfigMigrationStep:
    """Record one deterministic directed migration in a result envelope."""

    identifier: str
    source_version: ExperimentSchemaVersion
    target_version: ExperimentSchemaVersion
    description: str

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible descriptor."""
        return {
            "id": self.identifier,
            "from_version": self.source_version.to_json_value(),
            "to_version": self.target_version.to_json_value(),
            "description": self.description,
        }
