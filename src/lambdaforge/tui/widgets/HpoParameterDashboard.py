"""Visual response and coverage evidence for one HPO parameter."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

from textual.app import ComposeResult
from textual.containers import Grid, Vertical
from textual.widgets import Label
from textual_plot import HiResMode
from textual_plot.axis_formatter import NumericAxisFormatter

from lambdaforge.tui.widgets.InspectablePlotWidget import (
    InspectablePlotWidget,
    InspectionPoint,
)


class HpoParameterDashboard(Vertical):
    """Show surrogate response with uncertainty and observed support."""

    def compose(self) -> ComposeResult:
        with Grid(classes="hpo-plot-grid"):
            with Vertical(classes="hpo-plot-card"):
                yield Label(
                    "Predictive response",
                    classes="hpo-plot-title hpo-response-title",
                )
                yield InspectablePlotWidget(
                    classes="hpo-plot hpo-response-plot", allow_pan_and_zoom=True
                )
            with Vertical(classes="hpo-plot-card"):
                yield Label("Observed support", classes="hpo-plot-title")
                yield InspectablePlotWidget(
                    classes="hpo-plot hpo-coverage-plot", allow_pan_and_zoom=True
                )

    def show_parameter(self, name: str, response: Mapping[str, Any]) -> None:
        points = [
            value for value in response.get("points", ()) if isinstance(value, Mapping)
        ]
        live_observed = response.get("source") == "live-observed"
        response_plot: Any = self.query_one(".hpo-response-plot")
        coverage_plot: Any = self.query_one(".hpo-coverage-plot")
        response_plot.clear()
        coverage_plot.clear()
        response_plot.set_xlimits(None, None)
        response_plot.set_ylimits(None, None)
        coverage_plot.set_xlimits(None, None)
        coverage_plot.set_ylimits(None, None)
        self.query_one(".hpo-response-title", Label).update(
            "Observed response · live" if live_observed else "Predictive response"
        )
        numeric = bool(points) and all(
            isinstance(point.get("x"), int | float) and not isinstance(point.get("x"), bool)
            for point in points
        )
        coordinates = [
            point.get("x")
            if numeric
            else str(point.get("category", point.get("x", "?")))
            for point in points
        ]
        predicted = [self._number(point.get("predicted_objective")) for point in points]
        usable = [index for index, value in enumerate(predicted) if value is not None]
        x: list[Any] = [coordinates[index] for index in usable]
        y = [cast(float, predicted[index]) for index in usable]
        if x:
            if numeric:
                response_plot.set_x_formatter(NumericAxisFormatter())
                response_plot.set_xticks(None)
                response_plot.plot(x, y, line_style="bright_cyan", hires_mode=HiResMode.BRAILLE)
                response_plot.scatter(x, y, marker="●", marker_style="bright_cyan")
            else:
                response_plot.bar(x, y, bar_style="bright_cyan")
            response_plot.set_inspection_points(
                tuple(
                    InspectionPoint(
                        float(cast(int | float, value)) if numeric else float(position + 1),
                        objective,
                        f"{name}={value}",
                        f"objective={objective:.6g}",
                        "response",
                    )
                    for position, (value, objective) in enumerate(zip(x, y, strict=True))
                )
            )
        response_plot.set_xlabel(name)
        response_plot.set_ylabel("mean objective" if live_observed else "predicted objective")

        support = [
            int(point.get("support_count", point.get("support", 0)) or 0)
            for point in points
        ]
        if coordinates:
            if numeric:
                coverage_plot.set_x_formatter(NumericAxisFormatter())
                coverage_plot.set_xticks(None)
            coverage_plot.bar(coordinates, support, bar_style="bright_magenta")
            coverage_plot.set_inspection_points(
                tuple(
                    InspectionPoint(
                        float(cast(int | float, value)) if numeric else float(position + 1),
                        float(count),
                        f"{name}={value}",
                        f"observations={count}",
                        "support",
                    )
                    for position, (value, count) in enumerate(
                        zip(coordinates, support, strict=True)
                    )
                )
            )
        coverage_plot.set_xlabel(name)
        coverage_plot.set_ylabel("observations")

    @staticmethod
    def _number(value: Any) -> float | None:
        if isinstance(value, int | float) and not isinstance(value, bool):
            numeric = float(value)
            return numeric if math.isfinite(numeric) else None
        return None


__all__ = ["HpoParameterDashboard"]
