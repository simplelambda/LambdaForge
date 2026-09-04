"""Reusable visual widgets for the LambdaForge Research Console."""

from lambdaforge.tui.widgets.CoverageDashboard import CoverageDashboard
from lambdaforge.tui.widgets.HpoParameterDashboard import HpoParameterDashboard
from lambdaforge.tui.widgets.InspectablePlotWidget import (
    InspectablePlotWidget,
    InspectionPoint,
)
from lambdaforge.tui.widgets.MetricDashboard import MetricDashboard
from lambdaforge.tui.widgets.ResourceDashboard import ResourceDashboard
from lambdaforge.tui.widgets.StudyOverviewDashboard import StudyOverviewDashboard

__all__ = [
    "CoverageDashboard",
    "HpoParameterDashboard",
    "InspectablePlotWidget",
    "InspectionPoint",
    "MetricDashboard",
    "ResourceDashboard",
    "StudyOverviewDashboard",
]
