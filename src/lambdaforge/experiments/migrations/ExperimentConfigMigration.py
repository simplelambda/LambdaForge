"""Object contract for one directed experiment configuration migration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Any

from lambdaforge.experiments.migrations.ExperimentConfigMigrationStep import (
    ExperimentConfigMigrationStep,
)
from lambdaforge.experiments.migrations.ExperimentSchemaVersion import (
    ExperimentSchemaVersion,
)


class ExperimentConfigMigration(ABC):
    """Transform one exact Schema version into one later exact version."""

    @property
    @abstractmethod
    def identifier(self) -> str:
        """Return the stable migration identifier."""

    @property
    @abstractmethod
    def source_version(self) -> ExperimentSchemaVersion:
        """Return the only accepted input version."""

    @property
    @abstractmethod
    def target_version(self) -> ExperimentSchemaVersion:
        """Return the exact output version."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Describe the semantic transformation for users."""

    @abstractmethod
    def apply(self, config: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        """Transform an isolated mutable mapping and return the result."""

    def step(self) -> ExperimentConfigMigrationStep:
        """Return the immutable descriptor stored in migration results."""
        return ExperimentConfigMigrationStep(
            identifier=self.identifier,
            source_version=self.source_version,
            target_version=self.target_version,
            description=self.description,
        )
