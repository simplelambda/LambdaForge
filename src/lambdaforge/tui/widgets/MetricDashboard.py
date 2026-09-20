"""Native learning-curve plots for one persisted Study Run."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from textual.app import ComposeResult
from textual.containers import Grid, Vertical
from textual.widgets import Label
from textual_plot import HiResMode

from lambdaforge.tui.viewmodels import metric_display_name
from lambdaforge.tui.widgets.InspectablePlotWidget import (
    InspectablePlotWidget,
    InspectionPoint,
)


class MetricDashboard(Vertical):
    """Render at most four interactive scalar curves without owning telemetry state."""

    PAGE_SIZE = 4

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._page_signature: tuple[str, ...] = ()

    def compose(self) -> ComposeResult:
        with Grid(classes="metric-plot-grid"):
            for index in range(self.PAGE_SIZE):
                with Vertical(classes=f"metric-plot-card metric-plot-card-{index}"):
                    yield Label(classes=f"metric-plot-title metric-plot-title-{index}")
                    yield InspectablePlotWidget(
                        classes=f"metric-plot metric-plot-{index}",
                        allow_pan_and_zoom=True,
                    )

    def show_curves(
        self,
        curves: Mapping[str, Any],
        names: Sequence[str],
        *,
        selected_step: int | None,
        best_step: int | None,
        display_names: Mapping[str, Any] | None = None,
    ) -> None:
        """Replace the current page while retaining truthful selected/best markers."""
        signature = tuple(names)
        same_page = signature == self._page_signature
        self._page_signature = signature
        for index in range(self.PAGE_SIZE):
            card = self.query_one(f".metric-plot-card-{index}", Vertical)
            if index >= len(names):
                card.display = False
                continue
            card.display = True
            name = names[index]
            points = self._points(curves.get(name))
            self.query_one(f".metric-plot-title-{index}", Label).update(
                metric_display_name(name, display_names)
            )
            plot: Any = self.query_one(f".metric-plot-{index}")
            viewport = plot.user_viewport() if same_page else None
            plot.clear()
            plot.restore_viewport(viewport)
            if not points:
                continue
            x = [point[0] for point in points]
            y = [point[1] for point in points]
            plot.plot(
                x,
                y,
                line_style="bright_cyan",
                hires_mode=HiResMode.BRAILLE,
                label="observed",
            )
            plot.set_inspection_points(
                tuple(
                    InspectionPoint(
                        float(step),
                        value,
                        f"epoch={step}",
                        f"{metric_display_name(name, display_names)}={value:.6g}",
                        metric_display_name(name, display_names),
                    )
                    for step, value in points
                )
            )
            self._mark(plot, points, best_step, "bright_green", "best")
            self._mark(plot, points, selected_step, "bright_red", "selected")
            plot.set_xlabel("epoch")
            plot.set_ylabel("value")
            plot.show_legend()

    @staticmethod
    def _reset_viewport(plot: Any) -> None:
        """Return a reused plot to data-driven limits before drawing another metric."""
        plot.set_xlimits(None, None)
        plot.set_ylimits(None, None)

    @classmethod
    def _mark(
        cls,
        plot: Any,
        points: Sequence[tuple[int, float]],
        step: int | None,
        style: str,
        label: str,
    ) -> None:
        value = next((value for candidate, value in points if candidate == step), None)
        if value is not None and step is not None:
            plot.scatter([step], [value], marker="●", marker_style=style, label=label)

    @staticmethod
    def _raw_points(value: Any) -> Sequence[Mapping[str, Any]]:
        if not isinstance(value, Sequence) or isinstance(value, str | bytes):
            return ()
        return tuple(point for point in value if isinstance(point, Mapping))

    @classmethod
    def _points(cls, value: Any) -> list[tuple[int, float]]:
        points: list[tuple[int, float]] = []
        for point in cls._raw_points(value):
            step, observed = point.get("step"), point.get("value")
            if (
                isinstance(step, int)
                and not isinstance(step, bool)
                and isinstance(observed, int | float)
                and not isinstance(observed, bool)
                and math.isfinite(float(observed))
            ):
                points.append((step, float(observed)))
        return sorted(points)


__all__ = ["MetricDashboard"]
