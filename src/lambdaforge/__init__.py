"""LambdaForge: object-oriented infrastructure for reproducible ML training."""

from typing import TYPE_CHECKING

from lambdaforge.LambdaForgeVersion import LambdaForgeVersion
from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from lambdaforge.experiments.AggregateResult import AggregateResult
    from lambdaforge.experiments.Experiment import Experiment
    from lambdaforge.experiments.results.ResultCatalog import ResultCatalog
    from lambdaforge.experiments.results.ResultRecord import ResultRecord
    from lambdaforge.experiments.retention.ArtifactRetentionPlan import ArtifactRetentionPlan
    from lambdaforge.experiments.retention.ArtifactRetentionResult import ArtifactRetentionResult
    from lambdaforge.experiments.RunResult import RunResult
    from lambdaforge.LambdaForge import LambdaForge

LazyExports.install(
    __name__,
    {
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
        "RunResult": ("lambdaforge.experiments.RunResult", "RunResult"),
        "ResultCatalog": (
            "lambdaforge.experiments.results.ResultCatalog",
            "ResultCatalog",
        ),
        "ResultRecord": (
            "lambdaforge.experiments.results.ResultRecord",
            "ResultRecord",
        ),
    },
)

__version__ = LambdaForgeVersion.CURRENT
__all__ = [
    "AggregateResult",
    "ArtifactRetentionPlan",
    "ArtifactRetentionResult",
    "Experiment",
    "LambdaForge",
    "RunResult",
    "ResultCatalog",
    "ResultRecord",
    "__version__",
]
