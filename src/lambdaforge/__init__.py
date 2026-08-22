"""LambdaForge: declarative infrastructure for reproducible AI workflows."""

from typing import TYPE_CHECKING

from lambdaforge.LambdaForgeVersion import LambdaForgeVersion
from lambdaforge.LazyExports import LazyExports
from lambdaforge.runtime.api import artifact, current, metric, publish_dataset

if TYPE_CHECKING:
    from lambdaforge.artifacts.ArtifactService import ArtifactService
    from lambdaforge.configuration.MaterializedConfig import MaterializedConfig
    from lambdaforge.controlplane.jobs import JobHandle
    from lambdaforge.data.build_models import DatasetBuildPlan, DatasetBuildResult
    from lambdaforge.data.DatasetRecipe import DatasetRecipe
    from lambdaforge.experiments.AggregateResult import AggregateResult
    from lambdaforge.experiments.Experiment import Experiment
    from lambdaforge.experiments.results.ResultCatalog import ResultCatalog
    from lambdaforge.experiments.results.ResultRecord import ResultRecord
    from lambdaforge.experiments.retention.ArtifactRetentionPlan import ArtifactRetentionPlan
    from lambdaforge.experiments.retention.ArtifactRetentionResult import ArtifactRetentionResult
    from lambdaforge.experiments.RunResult import RunResult
    from lambdaforge.hpo.AdaptiveExperimentPlan import AdaptiveExperimentPlan
    from lambdaforge.hpo.AdaptiveExperimentResult import AdaptiveExperimentResult
    from lambdaforge.LambdaForge import LambdaForge
    from lambdaforge.results.ResultService import ResultService
    from lambdaforge.tasks.TaskExecutionPlan import TaskExecutionPlan
    from lambdaforge.tasks.TaskResult import TaskResult
    from lambdaforge.tasks.TaskRun import TaskRun
    from lambdaforge.visualization.PlotSpec import PlotSpec
    from lambdaforge.visualization.VisualizationService import VisualizationService
    from lambdaforge.workflows.models import WorkflowPlan, WorkflowResult, WorkflowValidationReport
    from lambdaforge.workflows.Workflow import Workflow

LazyExports.install(
    __name__,
    {
        "AdaptiveExperimentPlan": (
            "lambdaforge.hpo.AdaptiveExperimentPlan",
            "AdaptiveExperimentPlan",
        ),
        "AdaptiveExperimentResult": (
            "lambdaforge.hpo.AdaptiveExperimentResult",
            "AdaptiveExperimentResult",
        ),
        "AggregateResult": (
            "lambdaforge.experiments.AggregateResult",
            "AggregateResult",
        ),
        "ArtifactService": ("lambdaforge.artifacts.ArtifactService", "ArtifactService"),
        "ArtifactRetentionPlan": (
            "lambdaforge.experiments.retention.ArtifactRetentionPlan",
            "ArtifactRetentionPlan",
        ),
        "ArtifactRetentionResult": (
            "lambdaforge.experiments.retention.ArtifactRetentionResult",
            "ArtifactRetentionResult",
        ),
        "Experiment": ("lambdaforge.experiments.Experiment", "Experiment"),
        "DatasetBuildPlan": ("lambdaforge.data.build_models", "DatasetBuildPlan"),
        "DatasetBuildResult": ("lambdaforge.data.build_models", "DatasetBuildResult"),
        "DatasetRecipe": ("lambdaforge.data.DatasetRecipe", "DatasetRecipe"),
        "LambdaForge": ("lambdaforge.LambdaForge", "LambdaForge"),
        "JobHandle": ("lambdaforge.controlplane.jobs", "JobHandle"),
        "MaterializedConfig": (
            "lambdaforge.configuration.MaterializedConfig",
            "MaterializedConfig",
        ),
        "RunResult": ("lambdaforge.experiments.RunResult", "RunResult"),
        "ResultCatalog": (
            "lambdaforge.experiments.results.ResultCatalog",
            "ResultCatalog",
        ),
        "ResultRecord": (
            "lambdaforge.experiments.results.ResultRecord",
            "ResultRecord",
        ),
        "ResultService": ("lambdaforge.results.ResultService", "ResultService"),
        "PlotSpec": ("lambdaforge.visualization.PlotSpec", "PlotSpec"),
        "TaskExecutionPlan": (
            "lambdaforge.tasks.TaskExecutionPlan",
            "TaskExecutionPlan",
        ),
        "TaskResult": ("lambdaforge.tasks.TaskResult", "TaskResult"),
        "TaskRun": ("lambdaforge.tasks.TaskRun", "TaskRun"),
        "Workflow": ("lambdaforge.workflows.Workflow", "Workflow"),
        "WorkflowPlan": ("lambdaforge.workflows.models", "WorkflowPlan"),
        "WorkflowResult": ("lambdaforge.workflows.models", "WorkflowResult"),
        "WorkflowValidationReport": (
            "lambdaforge.workflows.models",
            "WorkflowValidationReport",
        ),
        "VisualizationService": (
            "lambdaforge.visualization.VisualizationService",
            "VisualizationService",
        ),
    },
)

__version__ = LambdaForgeVersion.CURRENT
__all__ = [
    "AdaptiveExperimentPlan",
    "AdaptiveExperimentResult",
    "AggregateResult",
    "ArtifactService",
    "ArtifactRetentionPlan",
    "ArtifactRetentionResult",
    "Experiment",
    "DatasetBuildPlan",
    "DatasetBuildResult",
    "DatasetRecipe",
    "LambdaForge",
    "JobHandle",
    "MaterializedConfig",
    "RunResult",
    "ResultCatalog",
    "ResultRecord",
    "ResultService",
    "PlotSpec",
    "TaskExecutionPlan",
    "TaskResult",
    "TaskRun",
    "Workflow",
    "WorkflowPlan",
    "WorkflowResult",
    "WorkflowValidationReport",
    "VisualizationService",
    "artifact",
    "current",
    "metric",
    "publish_dataset",
    "__version__",
]
