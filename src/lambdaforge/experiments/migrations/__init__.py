"""Versioned, preview-first experiment configuration migrations."""

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from lambdaforge.experiments.migrations.ExperimentConfigMigration import (
        ExperimentConfigMigration,
    )
    from lambdaforge.experiments.migrations.ExperimentConfigMigrationRegistry import (
        ExperimentConfigMigrationRegistry,
    )
    from lambdaforge.experiments.migrations.ExperimentConfigMigrationResult import (
        ExperimentConfigMigrationResult,
    )
    from lambdaforge.experiments.migrations.ExperimentConfigMigrationStep import (
        ExperimentConfigMigrationStep,
    )
    from lambdaforge.experiments.migrations.ExperimentConfigMigrator import (
        ExperimentConfigMigrator,
    )
    from lambdaforge.experiments.migrations.ExperimentSchemaCatalog import (
        ExperimentSchemaCatalog,
    )
    from lambdaforge.experiments.migrations.ExperimentSchemaVersion import (
        ExperimentSchemaVersion,
    )
    from lambdaforge.experiments.migrations.ExperimentV1ToV1_1Migration import (
        ExperimentV1ToV1_1Migration,
    )
    from lambdaforge.experiments.migrations.MigrationPreviewFormat import (
        MigrationPreviewFormat,
    )
    from lambdaforge.experiments.migrations.RoundTripYamlCodec import RoundTripYamlCodec
    from lambdaforge.experiments.migrations.UnversionedToV1Migration import (
        UnversionedToV1Migration,
    )

LazyExports.install(
    __name__,
    {
        name: (f"lambdaforge.experiments.migrations.{name}", name)
        for name in (
            "ExperimentConfigMigration",
            "ExperimentConfigMigrationRegistry",
            "ExperimentConfigMigrationResult",
            "ExperimentConfigMigrationStep",
            "ExperimentConfigMigrator",
            "ExperimentSchemaCatalog",
            "ExperimentSchemaVersion",
            "ExperimentV1ToV1_1Migration",
            "MigrationPreviewFormat",
            "RoundTripYamlCodec",
            "UnversionedToV1Migration",
        )
    },
)

__all__ = [
    "ExperimentConfigMigration",
    "ExperimentConfigMigrationRegistry",
    "ExperimentConfigMigrationResult",
    "ExperimentConfigMigrationStep",
    "ExperimentConfigMigrator",
    "ExperimentSchemaCatalog",
    "ExperimentSchemaVersion",
    "ExperimentV1ToV1_1Migration",
    "MigrationPreviewFormat",
    "RoundTripYamlCodec",
    "UnversionedToV1Migration",
]
