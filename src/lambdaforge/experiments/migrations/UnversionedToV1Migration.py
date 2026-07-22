"""Migration that declares the Schema of historical unversioned configs."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from lambdaforge.experiments.migrations.ExperimentConfigMigration import (
    ExperimentConfigMigration,
)
from lambdaforge.experiments.migrations.ExperimentSchemaVersion import (
    ExperimentSchemaVersion,
)


class UnversionedToV1Migration(ExperimentConfigMigration):
    """Declare historical Schema 1.0 without changing experiment semantics."""

    @property
    def identifier(self) -> str:
        """Return the stable migration identifier."""
        return "unversioned_to_1_0"

    @property
    def source_version(self) -> ExperimentSchemaVersion:
        """Accept only historical YAML without `schema_version`."""
        return ExperimentSchemaVersion.unversioned()

    @property
    def target_version(self) -> ExperimentSchemaVersion:
        """Produce the exact historical 1.0 Schema declaration."""
        return ExperimentSchemaVersion("1.0")

    @property
    def description(self) -> str:
        """Describe the compatibility-only transformation."""
        return "Declare schema_version 1.0 for a historically unversioned experiment."

    def apply(self, config: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        """Insert `schema_version` first while preserving round-trip metadata."""
        if "schema_version" in config:
            raise ValueError("Unversioned migration received an already versioned mapping.")
        insertion = getattr(config, "insert", None)
        if callable(insertion):
            insertion(0, "schema_version", "1.0")
            return config
        migrated: dict[str, Any] = {"schema_version": "1.0"}
        migrated.update(config)
        return migrated
