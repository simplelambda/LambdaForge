"""Visual, exact search-space coverage summary for Study Analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from textual.app import ComposeResult
from textual.containers import Grid, Vertical
from textual.widgets import DataTable, Label, Static

from lambdaforge.tui.viewmodels import format_value
from lambdaforge.tui.widgets.InspectablePlotWidget import (
    InspectablePlotWidget,
    InspectionPoint,
)


class CoverageDashboard(Vertical):
    """Show what has actually been sampled without implying full pool coverage."""

    def compose(self) -> ComposeResult:
        with Grid(classes="coverage-summary-grid"):
            yield Static(id="coverage-quality", classes="metric-card")
            yield Static(id="coverage-distance", classes="metric-card")
            yield Static(id="coverage-reference", classes="metric-card")
        yield Label("Marginal coverage by parameter", classes="section-title")
        yield InspectablePlotWidget(
            id="coverage-plot", classes="coverage-plot", allow_pan_and_zoom=True
        )
        yield DataTable(id="coverage-table", cursor_type="row", zebra_stripes=True)
        yield Static(
            "Coverage describes observed candidates only. Click a bar for its exact percentage; "
            "open contextual help for interpretation.",
            classes="legend",
        )

    def show_coverage(self, coverage: Mapping[str, Any]) -> None:
        joint = coverage.get("joint", {})
        joint = joint if isinstance(joint, Mapping) else {}
        attempted = joint.get("attempted", {})
        attempted = attempted if isinstance(attempted, Mapping) else {}
        self.query_one("#coverage-quality", Static).update(
            "[b]JOINT COVERAGE[/b]\n"
            f"{str(joint.get('quality', 'insufficient')).upper()}\n"
            "Compared with same-size space-filling designs"
        )
        self.query_one("#coverage-distance", Static).update(
            "[b]UNCOVERED DISTANCE[/b]\n"
            f"median {format_value(attempted.get('median_nearest_observed_distance'))}\n"
            f"p90 {format_value(attempted.get('p90_distance'))} · "
            f"max {format_value(attempted.get('maximum_distance'))}"
        )
        percentile = joint.get("coverage_percentile_vs_same_size_baseline")
        self.query_one("#coverage-reference", Static).update(
            "[b]REFERENCE CHECK[/b]\n"
            f"percentile {format_value(percentile)}\n"
            f"{joint.get('reference_points', 0)} deterministic reference points"
        )

        marginal = coverage.get("marginal", {})
        marginal = marginal if isinstance(marginal, Mapping) else {}
        rows: list[tuple[str, str, float, str, str, str]] = []
        for name, raw in marginal.items():
            if not isinstance(raw, Mapping):
                continue
            kind = str(raw.get("kind", "unknown"))
            if kind == "categorical":
                authored_values = list(raw.get("authored_levels", ()))
                observed_values = list(raw.get("observed_levels", ()))
                ratio = len(observed_values) / len(authored_values) if authored_values else 0.0
                observed_text = ", ".join(map(str, observed_values)) or "none"
                authored_text = ", ".join(map(str, authored_values)) or "unknown"
                support = ", ".join(
                    f"{label}:{count}"
                    for label, count in (raw.get("level_counts", {}) or {}).items()
                )
            else:
                occupied = int(raw.get("occupied_bins", 0) or 0)
                total = int(raw.get("total_bins", 0) or 0)
                ratio = occupied / total if total else 0.0
                observed_range = raw.get("observed_range")
                authored_range = raw.get("authored_range")
                observed_text = self._range(observed_range)
                authored_text = self._range(authored_range)
                edges = raw.get("edge_support", {})
                support = (
                    f"bins {occupied}/{total} · edges "
                    f"{edges.get('lower', 0)}/{edges.get('upper', 0)}"
                    if isinstance(edges, Mapping)
                    else f"bins {occupied}/{total}"
                )
            rows.append(
                (
                    str(name),
                    kind,
                    min(1.0, max(0.0, ratio)),
                    observed_text,
                    authored_text,
                    support,
                )
            )

        plot = self.query_one("#coverage-plot", InspectablePlotWidget)
        plot.clear()
        plot.set_xlimits(None, None)
        plot.set_ylimits(0.0, 1.05)
        if rows:
            labels = [name.replace("_", " ") for name, *_rest in rows]
            values = [row[2] for row in rows]
            plot.bar(labels, values, bar_style="bright_cyan")
            plot.set_inspection_points(
                tuple(
                    InspectionPoint(
                        float(index + 1),
                        value,
                        f"parameter={label}",
                        f"coverage={value:.1%}",
                        "marginal coverage",
                    )
                    for index, (label, value) in enumerate(zip(labels, values, strict=True))
                )
            )
        plot.set_xlabel("parameter")
        plot.set_ylabel("observed fraction")

        table = self.query_one("#coverage-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Parameter", "Kind", "Coverage", "Observed", "Authored", "Support")
        for name, kind, ratio, observed, authored, support in rows:
            table.add_row(
                name.replace("_", " ").title(),
                kind.title(),
                f"{ratio:.0%}",
                observed,
                authored,
                support,
            )

    @staticmethod
    def _range(value: Any) -> str:
        if isinstance(value, Sequence) and not isinstance(value, str | bytes) and len(value) >= 2:
            return f"{format_value(value[0])} … {format_value(value[1])}"
        return "unknown"


__all__ = ["CoverageDashboard"]
