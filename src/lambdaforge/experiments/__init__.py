"""Configuration, execution, aggregation and reload APIs."""

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from lambdaforge.experiments.AggregateResult import AggregateResult
    from lambdaforge.experiments.CheckpointChoice import CheckpointChoice
    from lambdaforge.experiments.DetachedRunState import DetachedRunState
    from lambdaforge.experiments.ExecutionConfig import ExecutionConfig
    from lambdaforge.experiments.ExecutionMode import ExecutionMode
    from lambdaforge.experiments.Experiment import Experiment
    from lambdaforge.experiments.ExperimentAggregator import ExperimentAggregator
    from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
    from lambdaforge.experiments.ExperimentExecutor import ExperimentExecutor
    from lambdaforge.experiments.ExperimentRunner import ExperimentRunner
    from lambdaforge.experiments.ExperimentValidator import ExperimentValidator
    from lambdaforge.experiments.JsonResult import JsonResult
    from lambdaforge.experiments.migrations.ExperimentConfigMigration import (
        ExperimentConfigMigration,
    )
    from lambdaforge.experiments.migrations.ExperimentConfigMigrationRegistry import (
        ExperimentConfigMigrationRegistry,
    )
    from lambdaforge.experiments.migrations.ExperimentConfigMigrationResult import (
        ExperimentConfigMigrationResult,
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
    from lambdaforge.experiments.migrations.MigrationPreviewFormat import (
        MigrationPreviewFormat,
    )
    from lambdaforge.experiments.ObjectFactory import ObjectFactory
    from lambdaforge.experiments.postrun.PostRunAction import PostRunAction
    from lambdaforge.experiments.postrun.PostRunActionReceipt import PostRunActionReceipt
    from lambdaforge.experiments.postrun.PostRunCheckpoint import PostRunCheckpoint
    from lambdaforge.experiments.postrun.PostRunContext import PostRunContext
    from lambdaforge.experiments.postrun.PostRunResult import PostRunResult
    from lambdaforge.experiments.postrun.PostRunService import PostRunService
    from lambdaforge.experiments.results.ResultCatalog import ResultCatalog
    from lambdaforge.experiments.results.ResultRecord import ResultRecord
    from lambdaforge.experiments.results.RunFingerprint import RunFingerprint
    from lambdaforge.experiments.retention.AggregationReceipt import AggregationReceipt
    from lambdaforge.experiments.retention.ArtifactCompressionOptions import (
        ArtifactCompressionOptions,
    )
    from lambdaforge.experiments.retention.ArtifactRetentionAction import (
        ArtifactRetentionAction,
    )
    from lambdaforge.experiments.retention.ArtifactRetentionManager import (
        ArtifactRetentionManager,
    )
    from lambdaforge.experiments.retention.ArtifactRetentionMode import ArtifactRetentionMode
    from lambdaforge.experiments.retention.ArtifactRetentionPlan import ArtifactRetentionPlan
    from lambdaforge.experiments.retention.ArtifactRetentionPolicy import ArtifactRetentionPolicy
    from lambdaforge.experiments.retention.ArtifactRetentionResult import ArtifactRetentionResult
    from lambdaforge.experiments.retention.ArtifactRetentionRule import ArtifactRetentionRule
    from lambdaforge.experiments.retention.ArtifactRetentionStatus import (
        ArtifactRetentionStatus,
    )
    from lambdaforge.experiments.retention.CheckpointRetention import CheckpointRetention
    from lambdaforge.experiments.RunLoader import RunLoader
    from lambdaforge.experiments.RunResult import RunResult
    from lambdaforge.experiments.RunStatus import RunStatus
    from lambdaforge.experiments.statistics.intervals.ConfidenceIntervalMethod import (
        ConfidenceIntervalMethod,
    )
    from lambdaforge.experiments.statistics.intervals.ConfidenceIntervalResult import (
        ConfidenceIntervalResult,
    )
    from lambdaforge.experiments.statistics.paired.PairedAlternative import (
        PairedAlternative,
    )
    from lambdaforge.experiments.statistics.paired.PairedTestMethod import PairedTestMethod
    from lambdaforge.experiments.statistics.paired.PairedTestResult import PairedTestResult
    from lambdaforge.experiments.statistics.paired.WilcoxonCalculation import (
        WilcoxonCalculation,
    )
    from lambdaforge.experiments.statistics.paired.WilcoxonZeroMethod import (
        WilcoxonZeroMethod,
    )
    from lambdaforge.experiments.statistics.StatisticalComparisonConfig import (
        StatisticalComparisonConfig,
    )
    from lambdaforge.experiments.ValidationReport import ValidationReport
    from lambdaforge.experiments.VariantAggregateResult import VariantAggregateResult

LazyExports.install(
    __name__,
    {
        **{
            name: (f"lambdaforge.experiments.{name}", name)
            for name in (
                "AggregateResult",
                "CheckpointChoice",
                "DetachedRunState",
                "ExecutionConfig",
                "ExecutionMode",
                "Experiment",
                "ExperimentAggregator",
                "ExperimentConfig",
                "ExperimentExecutor",
                "ExperimentRunner",
                "ExperimentValidator",
                "JsonResult",
                "ObjectFactory",
                "RunLoader",
                "RunResult",
                "RunStatus",
                "ValidationReport",
                "VariantAggregateResult",
            )
        },
        "PostRunAction": (
            "lambdaforge.experiments.postrun.PostRunAction",
            "PostRunAction",
        ),
        "PostRunActionReceipt": (
            "lambdaforge.experiments.postrun.PostRunActionReceipt",
            "PostRunActionReceipt",
        ),
        "PostRunCheckpoint": (
            "lambdaforge.experiments.postrun.PostRunCheckpoint",
            "PostRunCheckpoint",
        ),
        "PostRunContext": (
            "lambdaforge.experiments.postrun.PostRunContext",
            "PostRunContext",
        ),
        "PostRunResult": (
            "lambdaforge.experiments.postrun.PostRunResult",
            "PostRunResult",
        ),
        "PostRunService": (
            "lambdaforge.experiments.postrun.PostRunService",
            "PostRunService",
        ),
        "ResultCatalog": (
            "lambdaforge.experiments.results.ResultCatalog",
            "ResultCatalog",
        ),
        "ResultRecord": (
            "lambdaforge.experiments.results.ResultRecord",
            "ResultRecord",
        ),
        "RunFingerprint": (
            "lambdaforge.experiments.results.RunFingerprint",
            "RunFingerprint",
        ),
        "StatisticalComparisonConfig": (
            "lambdaforge.experiments.statistics.StatisticalComparisonConfig",
            "StatisticalComparisonConfig",
        ),
        "ConfidenceIntervalMethod": (
            "lambdaforge.experiments.statistics.intervals.ConfidenceIntervalMethod",
            "ConfidenceIntervalMethod",
        ),
        "ConfidenceIntervalResult": (
            "lambdaforge.experiments.statistics.intervals.ConfidenceIntervalResult",
            "ConfidenceIntervalResult",
        ),
        "PairedAlternative": (
            "lambdaforge.experiments.statistics.paired.PairedAlternative",
            "PairedAlternative",
        ),
        "PairedTestMethod": (
            "lambdaforge.experiments.statistics.paired.PairedTestMethod",
            "PairedTestMethod",
        ),
        "PairedTestResult": (
            "lambdaforge.experiments.statistics.paired.PairedTestResult",
            "PairedTestResult",
        ),
        "WilcoxonCalculation": (
            "lambdaforge.experiments.statistics.paired.WilcoxonCalculation",
            "WilcoxonCalculation",
        ),
        "WilcoxonZeroMethod": (
            "lambdaforge.experiments.statistics.paired.WilcoxonZeroMethod",
            "WilcoxonZeroMethod",
        ),
        "ExperimentConfigMigration": (
            "lambdaforge.experiments.migrations.ExperimentConfigMigration",
            "ExperimentConfigMigration",
        ),
        "ExperimentConfigMigrationRegistry": (
            "lambdaforge.experiments.migrations.ExperimentConfigMigrationRegistry",
            "ExperimentConfigMigrationRegistry",
        ),
        "ExperimentConfigMigrationResult": (
            "lambdaforge.experiments.migrations.ExperimentConfigMigrationResult",
            "ExperimentConfigMigrationResult",
        ),
        "ExperimentConfigMigrator": (
            "lambdaforge.experiments.migrations.ExperimentConfigMigrator",
            "ExperimentConfigMigrator",
        ),
        "ExperimentSchemaCatalog": (
            "lambdaforge.experiments.migrations.ExperimentSchemaCatalog",
            "ExperimentSchemaCatalog",
        ),
        "ExperimentSchemaVersion": (
            "lambdaforge.experiments.migrations.ExperimentSchemaVersion",
            "ExperimentSchemaVersion",
        ),
        "MigrationPreviewFormat": (
            "lambdaforge.experiments.migrations.MigrationPreviewFormat",
            "MigrationPreviewFormat",
        ),
        "AggregationReceipt": (
            "lambdaforge.experiments.retention.AggregationReceipt",
            "AggregationReceipt",
        ),
        "ArtifactCompressionOptions": (
            "lambdaforge.experiments.retention.ArtifactCompressionOptions",
            "ArtifactCompressionOptions",
        ),
        "ArtifactRetentionAction": (
            "lambdaforge.experiments.retention.ArtifactRetentionAction",
            "ArtifactRetentionAction",
        ),
        "ArtifactRetentionManager": (
            "lambdaforge.experiments.retention.ArtifactRetentionManager",
            "ArtifactRetentionManager",
        ),
        "ArtifactRetentionMode": (
            "lambdaforge.experiments.retention.ArtifactRetentionMode",
            "ArtifactRetentionMode",
        ),
        "ArtifactRetentionPlan": (
            "lambdaforge.experiments.retention.ArtifactRetentionPlan",
            "ArtifactRetentionPlan",
        ),
        "ArtifactRetentionPolicy": (
            "lambdaforge.experiments.retention.ArtifactRetentionPolicy",
            "ArtifactRetentionPolicy",
        ),
        "ArtifactRetentionResult": (
            "lambdaforge.experiments.retention.ArtifactRetentionResult",
            "ArtifactRetentionResult",
        ),
        "ArtifactRetentionRule": (
            "lambdaforge.experiments.retention.ArtifactRetentionRule",
            "ArtifactRetentionRule",
        ),
        "ArtifactRetentionStatus": (
            "lambdaforge.experiments.retention.ArtifactRetentionStatus",
            "ArtifactRetentionStatus",
        ),
        "CheckpointRetention": (
            "lambdaforge.experiments.retention.CheckpointRetention",
            "CheckpointRetention",
        ),
    },
)

__all__ = [
    "AggregateResult",
    "AggregationReceipt",
    "ArtifactCompressionOptions",
    "ArtifactRetentionAction",
    "ArtifactRetentionManager",
    "ArtifactRetentionMode",
    "ArtifactRetentionPlan",
    "ArtifactRetentionPolicy",
    "ArtifactRetentionResult",
    "ArtifactRetentionRule",
    "ArtifactRetentionStatus",
    "CheckpointChoice",
    "CheckpointRetention",
    "ConfidenceIntervalMethod",
    "ConfidenceIntervalResult",
    "DetachedRunState",
    "ExecutionConfig",
    "ExecutionMode",
    "Experiment",
    "ExperimentAggregator",
    "ExperimentConfig",
    "ExperimentConfigMigration",
    "ExperimentConfigMigrationRegistry",
    "ExperimentConfigMigrationResult",
    "ExperimentConfigMigrator",
    "ExperimentExecutor",
    "ExperimentRunner",
    "ExperimentSchemaCatalog",
    "ExperimentSchemaVersion",
    "ExperimentValidator",
    "JsonResult",
    "MigrationPreviewFormat",
    "ObjectFactory",
    "PairedAlternative",
    "PairedTestMethod",
    "PairedTestResult",
    "RunLoader",
    "RunResult",
    "RunStatus",
    "ResultCatalog",
    "ResultRecord",
    "RunFingerprint",
    "StatisticalComparisonConfig",
    "PostRunAction",
    "PostRunActionReceipt",
    "PostRunCheckpoint",
    "PostRunContext",
    "PostRunResult",
    "PostRunService",
    "ValidationReport",
    "VariantAggregateResult",
    "WilcoxonCalculation",
    "WilcoxonZeroMethod",
]
