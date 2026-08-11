"""Public declarative scientific visualization services."""

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from lambdaforge.visualization.PlotSpec import PlotSpec
    from lambdaforge.visualization.VisualizationService import VisualizationService

LazyExports.install(
    __name__,
    {
        "PlotSpec": ("lambdaforge.visualization.PlotSpec", "PlotSpec"),
        "VisualizationService": (
            "lambdaforge.visualization.VisualizationService",
            "VisualizationService",
        ),
    },
)

__all__ = ["PlotSpec", "VisualizationService"]
