"""Single discoverable facade for LambdaForge's main workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lambdaforge.experiments.Experiment import Experiment
from lambdaforge.experiments.ObjectFactory import ObjectFactory
from lambdaforge.LambdaForgeVersion import LambdaForgeVersion

if TYPE_CHECKING:
    from lambdaforge.configuration.MaterializedConfig import MaterializedConfig
    from lambdaforge.controlplane.jobs import JobHandle
    from lambdaforge.data.build_models import (
        DatasetBuildPlan,
        DatasetBuildResult,
        DatasetRecipeValidationReport,
    )
    from lambdaforge.data.DatasetRecipe import DatasetRecipe
    from lambdaforge.execution.ResourceRequest import ResourceRequest
    from lambdaforge.experiments.migrations.ExperimentConfigMigrationResult import (
        ExperimentConfigMigrationResult,
    )
    from lambdaforge.experiments.results.ResultRecord import ResultRecord
    from lambdaforge.experiments.retention.ArtifactRetentionPlan import ArtifactRetentionPlan
    from lambdaforge.experiments.retention.ArtifactRetentionResult import ArtifactRetentionResult
    from lambdaforge.experiments.RunResult import RunResult
    from lambdaforge.experiments.ValidationReport import ValidationReport
    from lambdaforge.hpo.AdaptiveExperimentPlan import AdaptiveExperimentPlan
    from lambdaforge.hpo.AdaptiveExperimentResult import AdaptiveExperimentResult
    from lambdaforge.tasks.TaskExecutionPlan import TaskExecutionPlan
    from lambdaforge.tasks.TaskResult import TaskResult
    from lambdaforge.tasks.TaskRun import TaskRun
    from lambdaforge.tasks.TaskValidator import TaskValidationReport
    from lambdaforge.workflows.models import WorkflowPlan, WorkflowResult, WorkflowValidationReport
    from lambdaforge.workflows.Workflow import Workflow


class LambdaForge:
    """Framework facade for experiments and YAML object construction.

    Most users start with :meth:`experiment`, :meth:`task`, :meth:`dataset` or :meth:`run`.
    Lower-level components remain available from the documented ``lambdaforge.data``,
    ``lambdaforge.nn``, ``lambdaforge.metrics``, ``lambdaforge.plugins``,
    ``lambdaforge.training``, ``lambdaforge.experiments``, ``lambdaforge.tasks`` and
    ``lambdaforge.preprocessing`` namespaces.
    """

    VERSION = LambdaForgeVersion.CURRENT

    @staticmethod
    def experiment(path: str | Path) -> Experiment:
        """Load a YAML experiment into the object API."""
        return Experiment.from_yaml(path)

    @staticmethod
    def task(path: str | Path) -> TaskRun:
        """Load a concise or strict task YAML document into the generic task API."""
        from lambdaforge.tasks.TaskRun import TaskRun

        return TaskRun.from_yaml(path)

    @staticmethod
    def workflow(path: str | Path) -> Workflow:
        """Load a concise or strict workflow YAML document into the DAG API."""
        from lambdaforge.workflows.Workflow import Workflow

        return Workflow.from_yaml(path)

    @staticmethod
    def dataset(path: str | Path) -> DatasetRecipe:
        """Load a `kind: dataset` YAML into the recipe/build API."""
        from lambdaforge.data.DatasetRecipe import DatasetRecipe

        return DatasetRecipe.from_yaml(path)

    @staticmethod
    def load(path: str | Path) -> Experiment | TaskRun | Workflow | DatasetRecipe:
        """Dispatch YAML to its dataset, workflow, task or training experiment API."""
        from lambdaforge.configuration.AuthoringConfig import AuthoringConfig
        from lambdaforge.configuration.ConfigurationKind import ConfigurationKind
        from lambdaforge.tasks.TaskConfig import TaskConfig

        if TaskConfig.is_task_file(path):
            return LambdaForge.task(path)
        resolved_kind = AuthoringConfig.from_yaml(path).materialize().kind
        if resolved_kind is ConfigurationKind.DATASET:
            return LambdaForge.dataset(path)
        return (
            LambdaForge.workflow(path)
            if resolved_kind is ConfigurationKind.WORKFLOW
            else LambdaForge.experiment(path)
        )

    @staticmethod
    def run(
        path: str | Path,
        *,
        dry_run: bool = False,
        execution_overrides: Mapping[str, Any] | None = None,
        aggregate_plots: bool = True,
    ) -> (
        list[RunResult]
        | AdaptiveExperimentPlan
        | AdaptiveExperimentResult
        | TaskResult
        | TaskExecutionPlan
        | WorkflowResult
        | WorkflowPlan
        | DatasetBuildResult
        | DatasetBuildPlan
    ):
        """Load and execute an experiment, task, workflow or dataset recipe in one call."""
        configured = LambdaForge.load(path)
        if isinstance(configured, Experiment):
            return configured.run(
                dry_run=dry_run,
                execution_overrides=execution_overrides,
                aggregate_plots=aggregate_plots,
            )
        if execution_overrides and any(value is not None for value in execution_overrides.values()):
            raise ValueError(
                "Experiment execution overrides cannot be applied to tasks, workflows or "
                "dataset recipes."
            )
        from lambdaforge.data.DatasetRecipe import DatasetRecipe

        if isinstance(configured, DatasetRecipe):
            return configured.inspect() if dry_run else configured.run()
        return configured.run(dry_run=dry_run)

    @staticmethod
    def validate(
        path: str | Path,
        *,
        check_imports: bool = True,
    ) -> (
        ValidationReport
        | TaskValidationReport
        | WorkflowValidationReport
        | DatasetRecipeValidationReport
    ):
        """Validate one experiment, task, workflow or dataset recipe without artifacts."""
        from lambdaforge.tasks.TaskConfig import TaskConfig
        from lambdaforge.tasks.TaskValidator import TaskValidator

        if TaskConfig.is_task_file(path):
            return TaskValidator().validate_file(path, check_imports=check_imports)
        from lambdaforge.configuration.AuthoringConfig import AuthoringConfig
        from lambdaforge.configuration.ConfigurationKind import ConfigurationKind

        kind = AuthoringConfig.from_yaml(path).materialize().kind
        if kind is ConfigurationKind.DATASET:
            return LambdaForge.dataset(path).validate(check_imports=check_imports)
        if kind is ConfigurationKind.WORKFLOW:
            from lambdaforge.workflows.WorkflowValidator import WorkflowValidator

            return WorkflowValidator().validate_file(path, check_imports=check_imports)
        from lambdaforge.experiments.ExperimentValidator import ExperimentValidator

        return ExperimentValidator().validate_file(path, check_imports=check_imports)

    @staticmethod
    def inspect(
        path: str | Path,
    ) -> (
        list[dict[str, Any]]
        | AdaptiveExperimentPlan
        | TaskExecutionPlan
        | WorkflowPlan
        | DatasetBuildPlan
    ):
        """Expand an experiment or return a task/workflow/dataset plan without running."""
        configured = LambdaForge.load(path)
        return configured.inspect()

    @staticmethod
    def materialize(path: str | Path) -> MaterializedConfig:
        """Compile concise authoring YAML to the exact strict runner configuration."""
        from lambdaforge.configuration.AuthoringConfig import AuthoringConfig

        return AuthoringConfig.from_yaml(path).materialize()

    @staticmethod
    def submit(
        path: str | Path,
        *,
        on: str,
        resources: ResourceRequest | None = None,
        dry_run: bool = False,
        run_arguments: Sequence[str] = (),
    ) -> JobHandle:
        """Submit through the persistent local control plane and return a job handle."""
        from lambdaforge.controlplane.ControlPlane import ControlPlane

        handle, _ = ControlPlane().submit(
            path,
            cluster=on,
            resources=resources,
            dry_run=dry_run,
            run_arguments=run_arguments,
        )
        return handle

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
        """List canonical and historical attempts for an experiment or task YAML."""
        from lambdaforge.tasks.TaskRun import TaskRun

        configured = LambdaForge.load(path)
        if isinstance(configured, Experiment):
            return configured.results(status=status, include_archived=include_archived)
        if not isinstance(configured, TaskRun):
            raise ValueError(
                "Workflow/dataset results are stored per node/build; query their service layer."
            )
        return configured.result_catalog().records(
            status=status,
            include_archived=include_archived,
        )

    @staticmethod
    def build(spec: Any) -> Any:
        """Construct an object from a YAML ``target``, ``ref`` or plugin spec."""
        return ObjectFactory.build(spec)
