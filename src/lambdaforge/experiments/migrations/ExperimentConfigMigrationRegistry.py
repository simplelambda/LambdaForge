"""Immutable registry and deterministic path planner for migrations."""

from __future__ import annotations

from collections.abc import Iterable

from lambdaforge.experiments.migrations.ExperimentConfigMigration import (
    ExperimentConfigMigration,
)
from lambdaforge.experiments.migrations.ExperimentSchemaVersion import (
    ExperimentSchemaVersion,
)
from lambdaforge.experiments.migrations.ExperimentV1ToV1_1Migration import (
    ExperimentV1ToV1_1Migration,
)
from lambdaforge.experiments.migrations.UnversionedToV1Migration import (
    UnversionedToV1Migration,
)


class ExperimentConfigMigrationRegistry:
    """Own a linear, forward-only migration graph without global mutation."""

    def __init__(self, migrations: Iterable[ExperimentConfigMigration] = ()) -> None:
        ordered = tuple(migrations)
        by_source: dict[ExperimentSchemaVersion, ExperimentConfigMigration] = {}
        identifiers: set[str] = set()
        for migration in ordered:
            if migration.identifier in identifiers:
                raise ValueError(f"Duplicate migration identifier: {migration.identifier!r}.")
            if migration.source_version in by_source:
                raise ValueError(f"Ambiguous migrations from Schema {migration.source_version}.")
            if migration.target_version <= migration.source_version:
                raise ValueError(f"Migration {migration.identifier!r} must move strictly forward.")
            identifiers.add(migration.identifier)
            by_source[migration.source_version] = migration
        self._migrations = tuple(sorted(ordered, key=lambda item: item.source_version.sort_key()))
        self._by_source = by_source

    @classmethod
    def default(cls) -> ExperimentConfigMigrationRegistry:
        """Return the built-in immutable migration registry."""
        return cls((UnversionedToV1Migration(), ExperimentV1ToV1_1Migration()))

    @property
    def migrations(self) -> tuple[ExperimentConfigMigration, ...]:
        """Return the registered steps in deterministic source order."""
        return self._migrations

    def with_migration(
        self,
        migration: ExperimentConfigMigration,
    ) -> ExperimentConfigMigrationRegistry:
        """Return a new registry containing one additional migration."""
        return ExperimentConfigMigrationRegistry((*self._migrations, migration))

    def path(
        self,
        source: ExperimentSchemaVersion,
        target: ExperimentSchemaVersion,
    ) -> tuple[ExperimentConfigMigration, ...]:
        """Resolve every consecutive step or fail without guessing."""
        if target < source:
            raise ValueError(f"Schema downgrade is not supported: {source} -> {target}.")
        if source == target:
            return ()

        current = source
        steps: list[ExperimentConfigMigration] = []
        visited: set[ExperimentSchemaVersion] = set()
        while current != target:
            if current in visited:
                raise ValueError(f"Migration cycle detected at Schema {current}.")
            visited.add(current)
            migration = self._by_source.get(current)
            if migration is None or migration.target_version > target:
                raise ValueError(f"No migration path from Schema {source} to {target}.")
            steps.append(migration)
            current = migration.target_version
        return tuple(steps)
