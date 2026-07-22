"""Compatibility migration from experiment Schema 1.0 to 1.1."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from lambdaforge.experiments.migrations.ExperimentConfigMigration import (
    ExperimentConfigMigration,
)
from lambdaforge.experiments.migrations.ExperimentSchemaVersion import (
    ExperimentSchemaVersion,
)


class ExperimentV1ToV1_1Migration(ExperimentConfigMigration):
    """Declare Schema 1.1 without changing existing experiment semantics."""

    @property
    def identifier(self) -> str:
        """Return the stable migration identifier."""
        return "1_0_to_1_1"

    @property
    def source_version(self) -> ExperimentSchemaVersion:
        """Accept only explicit Schema 1.0 configurations."""
        return ExperimentSchemaVersion("1.0")

    @property
    def target_version(self) -> ExperimentSchemaVersion:
        """Produce the explicit Schema 1.1 declaration."""
        return ExperimentSchemaVersion("1.1")

    @property
    def description(self) -> str:
        """Describe the declaration-only compatibility transformation."""
        return "Declare schema_version 1.1; all Schema 1.0 fields keep their semantics."

    def apply(self, config: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        """Replace only the exact 1.0 version declaration."""
        if config.get("schema_version") != "1.0":
            raise ValueError("Schema 1.0 to 1.1 migration requires schema_version '1.0'.")
        config["schema_version"] = "1.1"
        return config
