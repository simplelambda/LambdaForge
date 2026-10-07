"""Reproducible post-study analysis shared by CLI and Research Console."""

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from lambdaforge.analysis.AnalysisProfile import AnalysisProfile
    from lambdaforge.analysis.MetricCatalog import MetricCatalog
    from lambdaforge.analysis.StudyAnalysis import StudyAnalysis

LazyExports.install(
    __name__,
    {
        name: (f"lambdaforge.analysis.{name}", name)
        for name in ("AnalysisProfile", "MetricCatalog", "StudyAnalysis")
    },
)

__all__ = ["AnalysisProfile", "MetricCatalog", "StudyAnalysis"]
