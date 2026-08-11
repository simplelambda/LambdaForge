"""Public result-query and standard metric-series services."""

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from lambdaforge.results.MetricPoint import MetricPoint
    from lambdaforge.results.MetricSeries import MetricSeries
    from lambdaforge.results.RemoteResultService import RemoteResultService
    from lambdaforge.results.ResultSelectionError import ResultSelectionError
    from lambdaforge.results.ResultService import ResultService
    from lambdaforge.results.ResultSyncResult import ResultSyncResult

_NAMES = (
    "MetricPoint",
    "MetricSeries",
    "RemoteResultService",
    "ResultSelectionError",
    "ResultService",
    "ResultSyncResult",
)
LazyExports.install(__name__, {name: (f"lambdaforge.results.{name}", name) for name in _NAMES})

__all__ = [
    "MetricPoint",
    "MetricSeries",
    "RemoteResultService",
    "ResultSelectionError",
    "ResultService",
    "ResultSyncResult",
]
