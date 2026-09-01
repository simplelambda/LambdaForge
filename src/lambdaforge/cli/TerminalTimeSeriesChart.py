"""Small Unicode time-series charts for LambdaForge's dependency-light TUI."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeGuard


class TerminalTimeSeriesChart:
    """Render a bounded scalar series as a high-resolution Braille line chart.

    Braille cells provide a 2-by-4 pixel grid without a plotting dependency.  The
    renderer is deliberately pure: callers own sampling, terminal I/O and colour.
    """

    _DOT_BITS = ((0, 1, 2, 6), (3, 4, 5, 7))

    @classmethod
    def render(
        cls,
        label: str,
        values: Sequence[float | None],
        *,
        width: int,
        rows: int = 4,
        minimum: float | None = None,
        maximum: float | None = None,
        suffix: str = "",
        x_label: str = "older ← time → now",
        selected_index: int | None = None,
        best_index: int | None = None,
    ) -> list[str]:
        """Return a framed chart whose visible width never exceeds ``width``.

        ``selected_index`` marks one original observation with a solid dot and
        ``best_index`` marks the objective-selected epoch with a diamond. The
        monitor colours those symbols separately, which keeps this pure renderer
        usable in plain-text snapshots and terminals without colour support.
        """
        width = max(24, width)
        rows = max(2, rows)
        axis_width = 8
        columns = max(8, width - axis_width - 2)
        finite = [float(value) for value in values if cls._finite(value)]
        low = float(minimum) if minimum is not None else (min(finite) if finite else 0.0)
        high = float(maximum) if maximum is not None else (max(finite) if finite else 1.0)
        if high < low:
            low, high = high, low
        if math.isclose(high, low):
            padding = max(1.0, abs(high) * 0.05)
            low, high = low - padding, high + padding

        pixels_w, pixels_h = columns * 2, rows * 4
        sampled = cls._resample(values, pixels_w)
        raster = [[False for _ in range(pixels_w)] for _ in range(pixels_h)]
        previous: tuple[int, int] | None = None
        for x, value in enumerate(sampled):
            if value is None:
                previous = None
                continue
            ratio = min(1.0, max(0.0, (value - low) / (high - low)))
            point = (x, round((1.0 - ratio) * (pixels_h - 1)))
            if previous is not None:
                cls._line(raster, previous, point)
            else:
                raster[point[1]][point[0]] = True
            previous = point

        marker = cls._marker(
            values,
            selected_index,
            pixels_w=pixels_w,
            pixels_h=pixels_h,
            low=low,
            high=high,
        )
        best_marker = cls._marker(
            values,
            best_index,
            pixels_w=pixels_w,
            pixels_h=pixels_h,
            low=low,
            high=high,
        )

        current = next((float(value) for value in reversed(values) if cls._finite(value)), None)
        current_text = cls._number(current, suffix) if current is not None else "unknown"
        title = f"{label}  current={current_text}"
        output = [title[:width]]
        for cell_y in range(rows):
            value = high - (high - low) * cell_y / max(1, rows - 1)
            axis = cls._number(value, suffix) if cell_y in {0, rows - 1} else ""
            cells: list[str] = []
            for cell_x in range(columns):
                if marker == best_marker == (cell_x, cell_y):
                    cells.append("◈")
                    continue
                if marker == (cell_x, cell_y):
                    cells.append("●")
                    continue
                if best_marker == (cell_x, cell_y):
                    cells.append("◆")
                    continue
                bits = 0
                for local_x in range(2):
                    for local_y in range(4):
                        if raster[cell_y * 4 + local_y][cell_x * 2 + local_x]:
                            bits |= 1 << cls._DOT_BITS[local_x][local_y]
                cells.append(chr(0x2800 + bits) if bits else " ")
            output.append(f"{axis:>{axis_width - 1}} {'│'}{''.join(cells)}│")
        axis_label = x_label if len(x_label) <= columns else x_label[:columns]
        output.append(f"{'':>{axis_width}}└{axis_label:─<{columns}}┘")
        return output

    @classmethod
    def _marker(
        cls,
        values: Sequence[float | None],
        index: int | None,
        *,
        pixels_w: int,
        pixels_h: int,
        low: float,
        high: float,
    ) -> tuple[int, int] | None:
        if index is None or not values:
            return None
        index = min(max(0, index), len(values) - 1)
        value = values[index]
        if not cls._finite(value):
            return None
        marker_x = round(index * (pixels_w - 1) / max(1, len(values) - 1))
        marker_ratio = min(1.0, max(0.0, (float(value) - low) / (high - low)))
        marker_y = round((1.0 - marker_ratio) * (pixels_h - 1))
        return marker_x // 2, marker_y // 4

    @staticmethod
    def _finite(value: object) -> TypeGuard[int | float]:
        return (
            isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
        )

    @classmethod
    def _resample(cls, values: Sequence[float | None], count: int) -> tuple[float | None, ...]:
        clean = tuple(float(value) if cls._finite(value) else None for value in values)
        if not clean:
            return (None,) * count
        if len(clean) == 1:
            return (clean[0],) * count
        return tuple(
            clean[round(index * (len(clean) - 1) / max(1, count - 1))] for index in range(count)
        )

    @staticmethod
    def _line(raster: list[list[bool]], start: tuple[int, int], end: tuple[int, int]) -> None:
        """Draw one Bresenham segment into the chart raster."""
        x0, y0 = start
        x1, y1 = end
        dx, sx = abs(x1 - x0), 1 if x0 < x1 else -1
        dy, sy = -abs(y1 - y0), 1 if y0 < y1 else -1
        error = dx + dy
        while True:
            raster[y0][x0] = True
            if x0 == x1 and y0 == y1:
                return
            doubled = 2 * error
            if doubled >= dy:
                error += dy
                x0 += sx
            if doubled <= dx:
                error += dx
                y0 += sy

    @staticmethod
    def _number(value: float | None, suffix: str) -> str:
        if value is None:
            return "-"
        magnitude = abs(value)
        if value.is_integer():
            text = f"{value:.0f}"
        elif magnitude >= 100:
            text = f"{value:.0f}"
        elif magnitude >= 10:
            text = f"{value:.1f}"
        else:
            text = f"{value:.3g}"
        return f"{text}{suffix}"


__all__ = ["TerminalTimeSeriesChart"]
