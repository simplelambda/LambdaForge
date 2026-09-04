"""A tiny interaction adapter over textual-plot's native plotting widget."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from textual.events import Click
from textual_plot import PlotWidget


@dataclass(frozen=True, slots=True)
class InspectionPoint:
    """One real plotted coordinate and its human-facing labels."""

    x: float
    y: float
    x_label: str
    y_label: str
    series: str = "value"


class InspectablePlotWidget(PlotWidget):
    """Keep chart rendering in textual-plot while exposing exact values on click."""

    def __init__(
        self,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
        *,
        allow_pan_and_zoom: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
            allow_pan_and_zoom=allow_pan_and_zoom,
        )
        self._inspection_points: tuple[InspectionPoint, ...] = ()

    def set_inspection_points(self, points: Sequence[InspectionPoint]) -> None:
        self._inspection_points = tuple(points)

    def on_click(self, event: Click) -> None:
        if event.button != 1 or not self._inspection_points:
            return
        offset = event.get_content_offset(self)
        if offset is None:
            return
        clicked_x, clicked_y = self.get_coordinate_from_pixel(offset.x, offset.y)
        x_values = [point.x for point in self._inspection_points]
        y_values = [point.y for point in self._inspection_points]
        x_span = max(max(x_values) - min(x_values), 1e-12)
        y_span = max(max(y_values) - min(y_values), 1e-12)
        selected = min(
            self._inspection_points,
            key=lambda point: math.hypot(
                (point.x - clicked_x) / x_span,
                (point.y - clicked_y) / y_span,
            ),
        )
        self.notify(
            f"{selected.series}: {selected.x_label} · {selected.y_label}",
            title="Chart value",
        )


__all__ = ["InspectablePlotWidget", "InspectionPoint"]
