"""A tiny interaction adapter over textual-plot's native plotting widget."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from rich.segment import Segment
from textual.app import ComposeResult
from textual.containers import Grid
from textual.events import Click
from textual.strip import Strip
from textual.widgets import Static
from textual_hires_canvas import Canvas
from textual_plot import PlotWidget


class _NoColorSafeCanvas(Canvas):
    """Normalize a dependency's unstyled empty segments before Textual filters see them."""

    def render_line(self, y: int) -> Strip:
        strip = super().render_line(y)
        base = self.rich_style
        return Strip(
            [Segment(segment.text, segment.style or base, segment.control) for segment in strip],
            strip.cell_length,
        )


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

    def compose(self) -> ComposeResult:
        """Use textual-plot's canvas contract with a NO_COLOR-safe compatibility subclass."""
        with Grid():
            yield _NoColorSafeCanvas(1, 1, id="margin-top")
            yield _NoColorSafeCanvas(1, 1, id="margin-left")
            yield _NoColorSafeCanvas(1, 1, id="plot")
            yield _NoColorSafeCanvas(1, 1, id="margin-bottom")
        yield Static(id="legend")

    def set_inspection_points(self, points: Sequence[InspectionPoint]) -> None:
        self._inspection_points = tuple(points)

    def user_viewport(self) -> tuple[float, float, float, float] | None:
        """Return an explicitly panned/zoomed viewport, or ``None`` for auto limits.

        ``textual-plot`` currently exposes limit setters but no matching public getter.  Keep
        that small compatibility detail in this adapter so live telemetry refreshes do not throw
        away a researcher's position.
        """
        if all(
            bool(getattr(self, field, True))
            for field in ("_auto_x_min", "_auto_x_max", "_auto_y_min", "_auto_y_max")
        ):
            return None
        return (
            float(self._x_min),
            float(self._x_max),
            float(self._y_min),
            float(self._y_max),
        )

    def restore_viewport(self, viewport: tuple[float, float, float, float] | None) -> None:
        """Restore a user's viewport after redrawing the same semantic series."""
        if viewport is None:
            self.set_xlimits(None, None)
            self.set_ylimits(None, None)
            return
        self.set_xlimits(viewport[0], viewport[1])
        self.set_ylimits(viewport[2], viewport[3])

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
