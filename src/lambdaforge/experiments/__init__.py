"""Configuration, execution, aggregation and reload APIs."""

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from lambdaforge.experiments.CheckpointChoice import CheckpointChoice
    from lambdaforge.experiments.DetachedRunState import DetachedRunState
    from lambdaforge.experiments.ExecutionConfig import ExecutionConfig
    from lambdaforge.experiments.ExecutionMode import ExecutionMode
    from lambdaforge.experiments.Experiment import Experiment
    from lambdaforge.experiments.ExperimentAggregator import ExperimentAggregator
    from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
    from lambdaforge.experiments.ExperimentExecutor import ExperimentExecutor
    from lambdaforge.experiments.ExperimentRunner import ExperimentRunner
    from lambdaforge.experiments.ObjectFactory import ObjectFactory
    from lambdaforge.experiments.RunLoader import RunLoader
    from lambdaforge.experiments.RunStatus import RunStatus

LazyExports.install(
    __name__,
    {
        name: (f"lambdaforge.experiments.{name}", name)
        for name in (
            "CheckpointChoice",
            "DetachedRunState",
            "ExecutionConfig",
            "ExecutionMode",
            "Experiment",
            "ExperimentAggregator",
            "ExperimentConfig",
            "ExperimentExecutor",
            "ExperimentRunner",
            "ObjectFactory",
            "RunLoader",
            "RunStatus",
        )
    },
)

__all__ = [
    "CheckpointChoice",
    "DetachedRunState",
    "ExecutionConfig",
    "ExecutionMode",
    "Experiment",
    "ExperimentAggregator",
    "ExperimentConfig",
    "ExperimentExecutor",
    "ExperimentRunner",
    "ObjectFactory",
    "RunLoader",
    "RunStatus",
]
