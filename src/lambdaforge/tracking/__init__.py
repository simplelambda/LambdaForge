"""Lazy public API for optional experiment-tracking infrastructure."""

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from lambdaforge.tracking.MLflowTrackingLogger import MLflowTrackingLogger
    from lambdaforge.tracking.TensorBoardTrackingLogger import TensorBoardTrackingLogger
    from lambdaforge.tracking.TrackingBackend import TrackingBackend
    from lambdaforge.tracking.TrackingDependencyError import TrackingDependencyError
    from lambdaforge.tracking.TrackingDependencyGuard import TrackingDependencyGuard
    from lambdaforge.tracking.WeightsAndBiasesTrackingLogger import (
        WeightsAndBiasesTrackingLogger,
    )

LazyExports.install(
    __name__,
    {
        "MLflowTrackingLogger": (
            "lambdaforge.tracking.MLflowTrackingLogger",
            "MLflowTrackingLogger",
        ),
        "TensorBoardTrackingLogger": (
            "lambdaforge.tracking.TensorBoardTrackingLogger",
            "TensorBoardTrackingLogger",
        ),
        "TrackingBackend": (
            "lambdaforge.tracking.TrackingBackend",
            "TrackingBackend",
        ),
        "TrackingDependencyError": (
            "lambdaforge.tracking.TrackingDependencyError",
            "TrackingDependencyError",
        ),
        "TrackingDependencyGuard": (
            "lambdaforge.tracking.TrackingDependencyGuard",
            "TrackingDependencyGuard",
        ),
        "WeightsAndBiasesTrackingLogger": (
            "lambdaforge.tracking.WeightsAndBiasesTrackingLogger",
            "WeightsAndBiasesTrackingLogger",
        ),
    },
)

__all__ = [
    "MLflowTrackingLogger",
    "TensorBoardTrackingLogger",
    "TrackingBackend",
    "TrackingDependencyError",
    "TrackingDependencyGuard",
    "WeightsAndBiasesTrackingLogger",
]
