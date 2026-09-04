"""Interactive cluster resource plots backed by :mod:`textual_plot`."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from datetime import datetime
from time import monotonic
from typing import Any

from textual.app import ComposeResult
from textual.containers import Grid, Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Label, Static
from textual_plot import HiResMode

from lambdaforge.tui.widgets.InspectablePlotWidget import (
    InspectablePlotWidget,
    InspectionPoint,
)


class _RelativeTimeFormatter:
    """Format a fixed trailing window without exposing negative second offsets."""

    TICKS = (-300.0, -240.0, -180.0, -120.0, -60.0, 0.0)

    def get_ticks(self, _minimum: float, _maximum: float, _max_ticks: int = 8) -> list[float]:
        return list(self.TICKS)

    def get_labels_for_ticks(self, ticks: Sequence[float]) -> list[str]:
        return [ResourceDashboard._format_relative_time(value) for value in ticks]


class ResourceDashboard(Vertical):
    """Show CPU, RAM and GPU history without owning resource-probe policy.

    The widget only receives already-redacted snapshots from the control plane.  Its
    bounded in-memory history disappears with the console and can never become a
    second monitoring database.
    """

    HISTORY_SECONDS = 300.0
    MAX_SAMPLES = 180

    class ExportRequested(Message):
        """Request an optional interactive rendering of the currently selected history."""

        def __init__(self, cluster: str, series: Mapping[str, Any]) -> None:
            super().__init__()
            self.cluster = cluster
            self.series = dict(series)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._history: dict[str, dict[str, deque[tuple[float, float | None]]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.MAX_SAMPLES))
        )
        self._selected: str | None = None
        self._sample_ids: dict[str, str] = {}
        self._summary = "Select a cluster to inspect its resource history."

    @property
    def native_plots(self) -> bool:
        """Whether the installed environment contains the declared plot widget."""
        return True

    def compose(self) -> ComposeResult:
        with Horizontal(classes="resource-summary-row"):
            yield Static(self._summary, classes="resource-summary")
            yield Button("↗ HTML", classes="resource-export", flat=True)
        with Grid(classes="resource-plot-grid"):
            for metric in ("cpu", "ram", "gpu"):
                with Vertical(classes="resource-plot-card"):
                    yield Label(
                        "GPU MEMORY" if metric == "gpu" else metric.upper(),
                        classes="resource-plot-title",
                    )
                    yield InspectablePlotWidget(
                        classes=f"resource-plot resource-plot-{metric}",
                        allow_pan_and_zoom=True,
                    )

    def ingest(
        self,
        cluster: str,
        observed: Mapping[str, Any],
        personal: Mapping[str, Any] | None = None,
        *,
        sample_id: str | None = None,
    ) -> None:
        """Append one bounded sample without changing the visible cluster."""
        if sample_id and self._sample_ids.get(cluster) == sample_id:
            return
        if sample_id:
            self._sample_ids[cluster] = sample_id
        now = self._sample_time(sample_id)
        mine = self._mapping(self._mapping(personal).get("observed"))
        values = self._metric_values(observed, mine)
        for metric, pair in values.items():
            self._history[cluster][metric].append((now, pair[0]))
            self._history[cluster][f"my_{metric}"].append((now, pair[1]))

    def show_cluster(
        self,
        cluster: str,
        observed: Mapping[str, Any],
        personal: Mapping[str, Any] | None = None,
        *,
        ingest: bool = True,
        sample_id: str | None = None,
    ) -> None:
        """Select and render one cluster using total and personal LambdaForge usage."""
        if ingest:
            self.ingest(cluster, observed, personal, sample_id=sample_id)
        self._selected = cluster
        self.display = True
        personal_value = self._mapping(personal)
        mine = self._mapping(personal_value.get("observed"))
        requested = self._mapping(personal_value.get("requested"))
        gpus = self._sequence(observed.get("gpus"))
        self._summary = (
            f"{cluster}  ·  {observed.get('cpu_total', '?')} CPU cores  ·  "
            f"{self._bytes(observed.get('ram_total_bytes'))} RAM  ·  {len(gpus)} GPU\n"
            f"LambdaForge active jobs {personal_value.get('active_jobs', 0)}  ·  requested "
            f"{requested.get('cpu_cores', 0)} CPU / {self._bytes(requested.get('ram_bytes'))} "
            f"RAM / {requested.get('gpu_count', 0)} GPU  ·  observed jobs "
            f"{mine.get('job_count', 0)}\n"
            "[bright_cyan]━ cluster total[/]  ·  "
            "[bright_magenta]━ my LambdaForge jobs[/]  ·  fixed five-minute window"
        )
        if self.is_mounted:
            self.query_one(".resource-summary", Static).update(self._summary)
            self._render_plots()

    def hide_dashboard(self) -> None:
        self._selected = None
        self.display = False

    def on_mount(self) -> None:
        if self._selected is not None:
            self.query_one(".resource-summary", Static).update(self._summary)
            self._render_plots()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if "resource-export" not in event.button.classes:
            return
        if self._selected is None:
            self.notify("Select a cluster before exporting resource history.", severity="warning")
            return
        newest = max(
            (
                item[0]
                for metric in ("cpu", "ram", "gpu")
                for owner in (metric, f"my_{metric}")
                for item in self._history[self._selected][owner]
            ),
            default=monotonic(),
        )
        series: dict[str, Any] = {}
        for metric in ("cpu", "ram", "gpu"):
            total_x, total_y = self._series(self._history[self._selected][metric], newest)
            mine_x, mine_y = self._series(self._history[self._selected][f"my_{metric}"], newest)
            series[f"total_{metric}"] = list(zip(total_x, total_y, strict=True))
            series[f"mine_{metric}"] = list(zip(mine_x, mine_y, strict=True))
        self.post_message(self.ExportRequested(self._selected, series))

    def _render_plots(self) -> None:
        if self._selected is None:
            return
        for metric in ("cpu", "ram", "gpu"):
            plot: Any = self.query_one(f".resource-plot-{metric}")
            total = tuple(self._history[self._selected][metric])
            mine = tuple(self._history[self._selected][f"my_{metric}"])
            newest = max((item[0] for item in (*total, *mine)), default=monotonic())
            plot.clear()
            total_x, total_y = self._series(total, newest)
            mine_x, mine_y = self._series(mine, newest)
            if total_y:
                plot.plot(
                    total_x,
                    total_y,
                    line_style="bright_cyan",
                    hires_mode=HiResMode.BRAILLE,
                )
            if mine_y:
                plot.plot(
                    mine_x,
                    mine_y,
                    line_style="bright_magenta",
                    hires_mode=HiResMode.BRAILLE,
                )
            inspection = [
                InspectionPoint(x, y, self._format_relative_time(x), f"total={y:.1f}%", "cluster")
                for x, y in zip(total_x, total_y, strict=True)
            ]
            inspection.extend(
                InspectionPoint(x, y, self._format_relative_time(x), f"mine={y:.1f}%", "mine")
                for x, y in zip(mine_x, mine_y, strict=True)
            )
            plot.set_inspection_points(tuple(inspection))
            plot.set_xlimits(-self.HISTORY_SECONDS, 0.0)
            plot.set_xticks(_RelativeTimeFormatter.TICKS)
            plot.set_x_formatter(_RelativeTimeFormatter())
            plot.set_ylimits(0.0, 100.0)
            plot.set_xlabel("five-minute history")
            plot.set_ylabel("%")

    @classmethod
    def _series(
        cls, values: Sequence[tuple[float, float | None]], newest: float
    ) -> tuple[list[float], list[float]]:
        finite = [
            (stamp - newest, value)
            for stamp, value in values
            if value is not None and stamp - newest >= -cls.HISTORY_SECONDS
        ]
        return [item[0] for item in finite], [float(item[1]) for item in finite]

    @staticmethod
    def _format_relative_time(value: float) -> str:
        seconds = max(0, int(round(-value)))
        if seconds == 0:
            return "now"
        minutes, remainder = divmod(seconds, 60)
        return f"{minutes}m" if remainder == 0 else f"{minutes}m{remainder:02d}s"

    @staticmethod
    def _sample_time(sample_id: str | None) -> float:
        if sample_id:
            try:
                return datetime.fromisoformat(sample_id.replace("Z", "+00:00")).timestamp()
            except ValueError:
                pass
        return monotonic()

    @classmethod
    def _metric_values(
        cls, observed: Mapping[str, Any], mine: Mapping[str, Any]
    ) -> dict[str, tuple[float | None, float | None]]:
        cpu_total = cls._number(observed.get("cpu_total"))
        ram_total = cls._number(observed.get("ram_total_bytes"))
        ram_available = cls._number(observed.get("ram_available_bytes"))
        gpus = cls._sequence(observed.get("gpus"))
        gpu_memory_total = sum(
            cls._number(cls._mapping(item).get("memory_total_bytes")) or 0 for item in gpus
        )
        gpu_memory_used = sum(
            cls._number(cls._mapping(item).get("memory_used_bytes")) or 0 for item in gpus
        )
        return {
            "cpu": (
                cls._number(observed.get("cpu_load")),
                (cls._number(mine.get("cpu_percent")) or 0) / cpu_total
                if cpu_total and mine.get("job_count")
                else None,
            ),
            "ram": (
                100 * (ram_total - ram_available) / ram_total
                if ram_total and ram_available is not None
                else None,
                100 * (cls._number(mine.get("rss_bytes")) or 0) / ram_total
                if ram_total and mine.get("job_count")
                else None,
            ),
            "gpu": (
                100 * gpu_memory_used / gpu_memory_total if gpu_memory_total else None,
                100 * (cls._number(mine.get("gpu_memory_bytes")) or 0) / gpu_memory_total
                if gpu_memory_total and mine.get("job_count")
                else None,
            ),
        }

    @staticmethod
    def _mapping(value: Any) -> Mapping[str, Any]:
        return value if isinstance(value, Mapping) else {}

    @staticmethod
    def _sequence(value: Any) -> Sequence[Any]:
        return value if isinstance(value, Sequence) and not isinstance(value, str | bytes) else ()

    @staticmethod
    def _number(value: Any) -> float | None:
        return (
            float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None
        )

    @staticmethod
    def _bytes(value: Any) -> str:
        if not isinstance(value, int | float):
            return "unknown"
        size = float(value)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if abs(size) < 1024 or unit == "TiB":
                return f"{size:.1f} {unit}"
            size /= 1024
        return "unknown"


__all__ = ["ResourceDashboard"]
