"""Compact visual summary for a persisted adaptive Study."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from textual.app import ComposeResult
from textual.containers import Grid, Vertical
from textual.widgets import Label, Static
from textual_plot import HiResMode

from lambdaforge.tui.viewmodels import objective_display_name
from lambdaforge.tui.widgets.InspectablePlotWidget import (
    InspectablePlotWidget,
    InspectionPoint,
)


class StudyOverviewDashboard(Vertical):
    """Plot candidate quality and state distribution from persisted evidence only."""

    def compose(self) -> ComposeResult:
        with Grid(classes="study-plot-grid"):
            with Vertical(classes="study-plot-card"):
                yield Label(
                    "Candidate objective",
                    classes="study-plot-title",
                    id="study-objective-title",
                )
                yield InspectablePlotWidget(
                    classes="study-plot study-objective-plot", allow_pan_and_zoom=True
                )
            with Vertical(classes="study-plot-card"):
                yield Label("Run states", classes="study-plot-title")
                yield InspectablePlotWidget(
                    classes="study-plot study-state-plot", allow_pan_and_zoom=False
                )
                yield Static("No Runs observed yet.", classes="study-state-counts")

    def show_study(self, study: Mapping[str, Any]) -> None:
        candidates = [
            value for value in study.get("candidates", ()) if isinstance(value, Mapping)
        ]
        states = Counter(
            str(run.get("state", "unknown"))
            for candidate in candidates
            for run in candidate.get("runs", ())
            if isinstance(run, Mapping)
        )
        objective = study.get("objective", {})
        self.query_one("#study-objective-title", Label).update(
            f"{objective_display_name(objective)} by trial"
        )
        objective_plot: Any = self.query_one(".study-objective-plot")
        objective_plot.clear()
        points = [
            (int(item.get("trial", index)), float(item["selection_objective"]))
            for index, item in enumerate(candidates)
            if isinstance(item.get("selection_objective"), int | float)
        ]
        if points:
            points.sort()
            objective_plot.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                line_style="bright_cyan",
                hires_mode=HiResMode.BRAILLE,
            )
            objective_plot.scatter(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="●",
                marker_style="bright_cyan",
            )
            objective_plot.set_inspection_points(
                tuple(
                    InspectionPoint(
                        float(trial),
                        objective_value,
                        f"trial={trial}",
                        f"objective={objective_value:.6g}",
                        "candidate",
                    )
                    for trial, objective_value in points
                )
            )
        objective_plot.set_xlabel("trial")
        objective_plot.set_ylabel("selection")

        state_plot: Any = self.query_one(".study-state-plot")
        state_plot.clear()
        if states:
            labels = list(states)
            state_plot.bar(
                labels,
                [states[label] for label in labels],
                bar_style="bright_magenta",
            )
            state_plot.set_inspection_points(
                tuple(
                    InspectionPoint(
                        float(index + 1),
                        float(states[label]),
                        f"state={label}",
                        f"runs={states[label]}",
                        "run state",
                    )
                    for index, label in enumerate(labels)
                )
            )
        self._show_state_counts(states)
        state_plot.set_xlabel("state")
        state_plot.set_ylabel("runs")

    def _show_state_counts(self, states: Counter[str]) -> None:
        self.query_one(".study-state-counts", Static).update(
            "  ·  ".join(f"{label.replace('_', ' ').title()} {states[label]}" for label in states)
            or "No Runs observed yet."
        )

__all__ = ["StudyOverviewDashboard"]
