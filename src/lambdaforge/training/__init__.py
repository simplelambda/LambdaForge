"""Lightning training, callbacks and safe multi-process orchestration."""

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from lambdaforge.training.callbacks.EpochLogPrinter import EpochLogPrinter
    from lambdaforge.training.callbacks.EpochMetricsCSV import EpochMetricsCSV
    from lambdaforge.training.callbacks.EpochStats import EpochStats
    from lambdaforge.training.CheckpointPolicy import CheckpointPolicy
    from lambdaforge.training.LightningRunner import LightningRunner
    from lambdaforge.training.LightningTask import LightningTask
    from lambdaforge.training.LightningTrainConfig import LightningTrainConfig
    from lambdaforge.training.LoggerMode import LoggerMode
    from lambdaforge.training.MatmulPrecision import MatmulPrecision
    from lambdaforge.training.MonitorMode import MonitorMode
    from lambdaforge.training.orchestration.ProcessGuard import ProcessGuard
    from lambdaforge.training.orchestration.TrainingJob import TrainingJob
    from lambdaforge.training.orchestration.TrainingOrchestrator import TrainingOrchestrator
    from lambdaforge.training.TaskLoggingConfig import TaskLoggingConfig

LazyExports.install(
    __name__,
    {
        "CheckpointPolicy": (
            "lambdaforge.training.CheckpointPolicy",
            "CheckpointPolicy",
        ),
        "EpochLogPrinter": (
            "lambdaforge.training.callbacks.EpochLogPrinter",
            "EpochLogPrinter",
        ),
        "EpochMetricsCSV": (
            "lambdaforge.training.callbacks.EpochMetricsCSV",
            "EpochMetricsCSV",
        ),
        "EpochStats": ("lambdaforge.training.callbacks.EpochStats", "EpochStats"),
        "LightningRunner": ("lambdaforge.training.LightningRunner", "LightningRunner"),
        "LightningTask": ("lambdaforge.training.LightningTask", "LightningTask"),
        "LightningTrainConfig": (
            "lambdaforge.training.LightningTrainConfig",
            "LightningTrainConfig",
        ),
        "LoggerMode": ("lambdaforge.training.LoggerMode", "LoggerMode"),
        "MatmulPrecision": ("lambdaforge.training.MatmulPrecision", "MatmulPrecision"),
        "MonitorMode": ("lambdaforge.training.MonitorMode", "MonitorMode"),
        "ProcessGuard": (
            "lambdaforge.training.orchestration.ProcessGuard",
            "ProcessGuard",
        ),
        "TrainingJob": (
            "lambdaforge.training.orchestration.TrainingJob",
            "TrainingJob",
        ),
        "TrainingOrchestrator": (
            "lambdaforge.training.orchestration.TrainingOrchestrator",
            "TrainingOrchestrator",
        ),
        "TaskLoggingConfig": (
            "lambdaforge.training.TaskLoggingConfig",
            "TaskLoggingConfig",
        ),
    },
)

__all__ = [
    "CheckpointPolicy",
    "EpochLogPrinter",
    "EpochMetricsCSV",
    "EpochStats",
    "LightningRunner",
    "LightningTask",
    "LightningTrainConfig",
    "LoggerMode",
    "MatmulPrecision",
    "MonitorMode",
    "ProcessGuard",
    "TrainingJob",
    "TrainingOrchestrator",
    "TaskLoggingConfig",
]
