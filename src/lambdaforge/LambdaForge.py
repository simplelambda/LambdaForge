"""Single discoverable facade for LambdaForge's main workflows."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lambdaforge.experiments.Experiment import Experiment
from lambdaforge.experiments.ObjectFactory import ObjectFactory
from lambdaforge.LambdaForgeVersion import LambdaForgeVersion

if TYPE_CHECKING:
    from lambdaforge.experiments.migrations.ExperimentConfigMigrationResult import (
        ExperimentConfigMigrationResult,
    )
    from lambdaforge.experiments.results.ResultRecord import ResultRecord
    from lambdaforge.experiments.retention.ArtifactRetentionPlan import ArtifactRetentionPlan
    from lambdaforge.experiments.retention.ArtifactRetentionResult import ArtifactRetentionResult
    from lambdaforge.experiments.RunResult import RunResult
    from lambdaforge.experiments.ValidationReport import ValidationReport


class LambdaForge:
    """Framework facade for experiments and YAML object construction.

    Most users start with :meth:`experiment` or :meth:`run`. Lower-level
    components remain available from the documented ``lambdaforge.data``,
    ``lambdaforge.nn``, ``lambdaforge.metrics``, ``lambdaforge.plugins``,
    ``lambdaforge.training`` and ``lambdaforge.experiments`` namespaces.
    """

    VERSION = LambdaForgeVersion.CURRENT

    @staticmethod
    def experiment(path: str | Path) -> Experiment:
        """Load a YAML experiment into the object API."""
        return Experiment.from_yaml(path)

    @staticmethod
    def run(
        path: str | Path,
        *,
        dry_run: bool = False,
        execution_overrides: Mapping[str, Any] | None = None,
        aggregate_plots: bool = True,
    ) -> list[RunResult]:
        """Load and execute a YAML experiment in one call."""
        return LambdaForge.experiment(path).run(
            dry_run=dry_run,
            execution_overrides=execution_overrides,
            aggregate_plots=aggregate_plots,
        )

    @staticmethod
    def validate(path: str | Path, *, check_imports: bool = True) -> ValidationReport:
        """Validate one experiment file without creating run artifacts."""
        from lambdaforge.experiments.ExperimentValidator import ExperimentValidator

        return ExperimentValidator().validate_file(path, check_imports=check_imports)

    @staticmethod
    def preview_migration(
        path: str | Path,
        *,
        target_version: str | None = None,
    ) -> ExperimentConfigMigrationResult:
        """Preview a validated version migration without modifying the source."""
        from lambdaforge.experiments.migrations.ExperimentConfigMigrator import (
            ExperimentConfigMigrator,
        )

        return ExperimentConfigMigrator.default().preview_file(
            path,
            target_version=target_version,
        )

    @staticmethod
    def preview_retention(path: str | Path) -> ArtifactRetentionPlan:
        """Preview post-aggregation retention without writing any artifact."""
        return LambdaForge.experiment(path).preview_retention()

    @staticmethod
    def apply_retention(path: str | Path) -> ArtifactRetentionResult:
        """Explicitly apply an eligible artifact-retention transaction."""
        return LambdaForge.experiment(path).apply_retention()

    @staticmethod
    def results(
        path: str | Path,
        *,
        status: str | None = None,
        include_archived: bool = True,
    ) -> tuple[ResultRecord, ...]:
        """List canonical and historical attempts for one experiment YAML."""
        return LambdaForge.experiment(path).results(
            status=status,
            include_archived=include_archived,
        )

    @staticmethod
    def build(spec: Any) -> Any:
        """Construct an object from a YAML ``target``, ``ref`` or plugin spec."""
        return ObjectFactory.build(spec)
