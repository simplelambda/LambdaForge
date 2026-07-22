"""Pure orchestrator for versioned configuration migration previews."""

from __future__ import annotations

import copy
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

from lambdaforge.experiments.migrations.ExperimentConfigMigrationRegistry import (
    ExperimentConfigMigrationRegistry,
)
from lambdaforge.experiments.migrations.ExperimentConfigMigrationResult import (
    ExperimentConfigMigrationResult,
)
from lambdaforge.experiments.migrations.ExperimentSchemaCatalog import (
    ExperimentSchemaCatalog,
)
from lambdaforge.experiments.migrations.ExperimentSchemaVersion import (
    ExperimentSchemaVersion,
)
from lambdaforge.experiments.migrations.RoundTripYamlCodec import RoundTripYamlCodec


class ExperimentConfigMigrator:
    """Plan, apply and validate migrations without importing experiment objects."""

    def __init__(
        self,
        registry: ExperimentConfigMigrationRegistry | None = None,
        schema_catalog: ExperimentSchemaCatalog | None = None,
        codec: RoundTripYamlCodec | None = None,
    ) -> None:
        self.registry = registry or ExperimentConfigMigrationRegistry.default()
        self.schema_catalog = schema_catalog or ExperimentSchemaCatalog()
        self.codec = codec or RoundTripYamlCodec()

    @classmethod
    def default(cls) -> ExperimentConfigMigrator:
        """Build the framework's default immutable migration pipeline."""
        return cls()

    def preview_file(
        self,
        path: str | Path,
        *,
        target_version: ExperimentSchemaVersion | str | None = None,
        validate: bool = True,
    ) -> ExperimentConfigMigrationResult:
        """Preview one YAML file without modifying it or importing user targets."""
        path = Path(path)
        config, original_yaml = self.codec.load_file(path)
        return self._preview(
            config,
            target_version=target_version,
            source=str(path),
            original_yaml=original_yaml,
            validate=validate,
            detach_round_trip=True,
        )

    def preview_mapping(
        self,
        config: Mapping[str, Any],
        *,
        target_version: ExperimentSchemaVersion | str | None = None,
        validate: bool = True,
    ) -> ExperimentConfigMigrationResult:
        """Preview an isolated mapping and leave every input object untouched."""
        if not isinstance(config, Mapping):
            raise TypeError("Experiment configuration must be a mapping.")
        working: MutableMapping[str, Any] = copy.deepcopy(dict(config))
        original_yaml = self.codec.dump_preview(working)
        return self._preview(
            working,
            target_version=target_version,
            source=None,
            original_yaml=original_yaml,
            validate=validate,
            detach_round_trip=False,
        )

    def _preview(
        self,
        config: MutableMapping[str, Any],
        *,
        target_version: ExperimentSchemaVersion | str | None,
        source: str | None,
        original_yaml: str,
        validate: bool,
        detach_round_trip: bool,
    ) -> ExperimentConfigMigrationResult:
        source_version = ExperimentSchemaVersion.from_config(config)
        target = self._target_version(target_version)
        migrations = self.registry.path(source_version, target)
        working = copy.deepcopy(config)
        steps = []
        for migration in migrations:
            current = ExperimentSchemaVersion.from_config(working)
            if current != migration.source_version:
                raise ValueError(
                    f"Migration {migration.identifier!r} expected Schema "
                    f"{migration.source_version}, found {current}."
                )
            if validate and not current.is_unversioned:
                source_errors = self.schema_catalog.validation_errors(working, current)
                if source_errors:
                    joined = "\n  - ".join(source_errors)
                    raise ValueError(
                        f"Configuration is invalid before migration "
                        f"{migration.identifier!r} from Schema {current}:\n  - {joined}"
                    )
            migrated = migration.apply(working)
            if not isinstance(migrated, MutableMapping):
                raise TypeError(
                    f"Migration {migration.identifier!r} must return a mutable mapping."
                )
            working = migrated
            actual = ExperimentSchemaVersion.from_config(working)
            if actual != migration.target_version:
                raise ValueError(
                    f"Migration {migration.identifier!r} produced Schema {actual}, "
                    f"expected {migration.target_version}."
                )
            errors = self.schema_catalog.validation_errors(working, actual) if validate else ()
            if errors:
                joined = "\n  - ".join(errors)
                raise ValueError(
                    f"Migration {migration.identifier!r} produced invalid Schema "
                    f"{actual}:\n  - {joined}"
                )
            steps.append(migration.step())

        if not migrations and validate:
            errors = self.schema_catalog.validation_errors(working, target)
            if errors:
                joined = "\n  - ".join(errors)
                raise ValueError(f"Configuration is invalid for Schema {target}:\n  - {joined}")

        migrated_yaml = (
            (
                self.codec.dump(working, newline=self.codec.newline_for(original_yaml))
                if detach_round_trip
                else self.codec.dump_preview(
                    working,
                    newline=self.codec.newline_for(original_yaml),
                )
            )
            if steps
            else original_yaml
        )
        warnings: tuple[str, ...] = ()
        if steps:
            warnings = (
                "Round-trip YAML preserves comments, order and anchors where "
                "structure permits; unmodified presentation may still be normalized.",
            )
        return ExperimentConfigMigrationResult(
            source=source,
            source_version=source_version,
            target_version=target,
            steps=tuple(steps),
            config=(
                self.codec.to_plain_mapping(working)
                if detach_round_trip
                else copy.deepcopy(dict(working))
            ),
            original_yaml=original_yaml,
            migrated_yaml=migrated_yaml,
            warnings=warnings,
        )

    def _target_version(
        self,
        value: ExperimentSchemaVersion | str | None,
    ) -> ExperimentSchemaVersion:
        target = (
            self.schema_catalog.current_version
            if value is None
            else value
            if isinstance(value, ExperimentSchemaVersion)
            else ExperimentSchemaVersion.from_value(value)
        )
        if target.is_unversioned:
            raise ValueError("A migration target must be an explicit Schema version.")
        if target not in self.schema_catalog.supported_versions:
            raise ValueError(f"No packaged JSON Schema for migration target {target}.")
        return target
