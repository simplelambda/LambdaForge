"""LambdaForge: declarative infrastructure for reproducible AI workflows."""

from typing import TYPE_CHECKING

from lambdaforge.LambdaForgeVersion import LambdaForgeVersion
from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from lambdaforge.configuration.MaterializedConfig import MaterializedConfig
    from lambdaforge.controlplane.JobHandle import JobHandle
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
    from lambdaforge.tasks.TaskExecutionPlan import TaskExecutionPlan
    from lambdaforge.tasks.TaskResult import TaskResult
    from lambdaforge.tasks.TaskRun import TaskRun
    from lambdaforge.workflows.Workflow import Workflow
    from lambdaforge.workflows.WorkflowPlan import WorkflowPlan
    from lambdaforge.workflows.WorkflowResult import WorkflowResult
    from lambdaforge.workflows.WorkflowValidationReport import WorkflowValidationReport

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
        "ArtifactRetentionPlan": (
            "lambdaforge.experiments.retention.ArtifactRetentionPlan",
            "ArtifactRetentionPlan",
        ),
        "ArtifactRetentionResult": (
            "lambdaforge.experiments.retention.ArtifactRetentionResult",
            "ArtifactRetentionResult",
        ),
        "Experiment": ("lambdaforge.experiments.Experiment", "Experiment"),
        "LambdaForge": ("lambdaforge.LambdaForge", "LambdaForge"),
        "JobHandle": ("lambdaforge.controlplane.JobHandle", "JobHandle"),
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
        "TaskExecutionPlan": (
            "lambdaforge.tasks.TaskExecutionPlan",
            "TaskExecutionPlan",
        ),
        "TaskResult": ("lambdaforge.tasks.TaskResult", "TaskResult"),
        "TaskRun": ("lambdaforge.tasks.TaskRun", "TaskRun"),
        "Workflow": ("lambdaforge.workflows.Workflow", "Workflow"),
        "WorkflowPlan": ("lambdaforge.workflows.WorkflowPlan", "WorkflowPlan"),
        "WorkflowResult": ("lambdaforge.workflows.WorkflowResult", "WorkflowResult"),
        "WorkflowValidationReport": (
            "lambdaforge.workflows.WorkflowValidationReport",
            "WorkflowValidationReport",
        ),
    },
)

__version__ = LambdaForgeVersion.CURRENT
__all__ = [
    "AdaptiveExperimentPlan",
    "AdaptiveExperimentResult",
    "AggregateResult",
    "ArtifactRetentionPlan",
    "ArtifactRetentionResult",
    "Experiment",
    "LambdaForge",
    "JobHandle",
    "MaterializedConfig",
    "RunResult",
    "ResultCatalog",
    "ResultRecord",
    "TaskExecutionPlan",
    "TaskResult",
    "TaskRun",
    "Workflow",
    "WorkflowPlan",
    "WorkflowResult",
    "WorkflowValidationReport",
    "__version__",
]
