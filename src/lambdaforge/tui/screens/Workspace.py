"""Hierarchical entity workspaces for the Textual Research Console."""

# Human-facing terminal copy is intentionally kept as complete strings for translation/readability.
# ruff: noqa: E501

from __future__ import annotations

import json
import re
import statistics
import time
import webbrowser
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from threading import Lock, Thread
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.events import MouseDown, MouseMove, MouseUp, Resize
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    LoadingIndicator,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from lambdaforge.analysis.Report import write_html, write_metric_html, write_parameter_html
from lambdaforge.tui.viewmodels import (
    confirmation_text,
    epoch_rows,
    format_duration,
    format_value,
    metric_display_name,
    objective_display_name,
    structured_text,
    visible_metrics,
)
from lambdaforge.tui.widgets import (
    CoverageDashboard,
    HpoParameterDashboard,
    MetricDashboard,
    ResourceDashboard,
    StudyOverviewDashboard,
)


def _structured_text(value: Any, *, heading: str | None = None, limit: int = 160) -> str:
    """Compatibility alias for workspace call sites using the shared view-model formatter."""
    return structured_text(value, heading=heading, limit=limit)


class ResearchWorkspace(Screen[None]):
    """Base screen with real stack navigation and human-readable breadcrumbs."""

    BINDINGS = [
        ("escape", "back", "Back"),
        ("left", "back", "Back"),
        ("home", "home", "Root"),
        ("question_mark", "context_help", "Help"),
    ]

    def __init__(self, breadcrumb: str) -> None:
        super().__init__()
        self.breadcrumb = breadcrumb

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(classes="workspace-navigation"):
            yield Button(
                f"← Back to {self._parent_label()}",
                id="workspace-back",
                classes="workspace-back",
                action="screen.back",
                flat=True,
            )
            with Horizontal(classes="breadcrumb-trail"):
                parts = self._breadcrumb_parts()
                for index, part in enumerate(parts):
                    if index:
                        yield Label("/", classes="breadcrumb-separator")
                    yield Button(
                        part,
                        id=f"breadcrumb-{index}",
                        classes=(
                            "breadcrumb-button breadcrumb-current"
                            if index == len(parts) - 1
                            else "breadcrumb-button"
                        ),
                        action=f"screen.breadcrumb({index})",
                        flat=True,
                        compact=True,
                        disabled=index == len(parts) - 1,
                    )
        yield from self.compose_workspace()
        yield Footer()

    def compose_workspace(self) -> ComposeResult:
        raise NotImplementedError

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_home(self) -> None:
        while len(self.app.screen_stack) > 1:
            self.app.pop_screen()

    def action_breadcrumb(self, index: int) -> None:
        """Pop real workspace levels when an ancestor breadcrumb is clicked."""
        parts = self._breadcrumb_parts()
        if not 0 <= index < len(parts) - 1:
            return
        for _ in range(len(parts) - index - 1):
            if len(self.app.screen_stack) <= 1:
                break
            self.app.pop_screen()

    def _breadcrumb_parts(self) -> list[str]:
        return [part.strip() for part in self.breadcrumb.split("/") if part.strip()]

    def _parent_label(self) -> str:
        parts = self._breadcrumb_parts()
        return parts[-2] if len(parts) > 1 else "Overview"

    def action_context_help(self) -> None:
        self.notify(
            "Enter/right opens the selected entity. Esc/left returns one level. "
            "† is censored partial evidence; ★ is the best comparable objective; "
            "◆ marks a Pareto-optimal candidate.",
            title="Contextual help",
        )

    def _show_explanation(self, title: str, text: str) -> None:
        self.app.push_screen(ExplanationDialog(title, text))

    def _write_and_open_report(self, title: str, writer: Any) -> None:
        """Create an explicitly requested local report without blocking the TUI."""
        self.notify("Generating the interactive report…", title=title)

        def generate() -> None:
            try:
                path = writer()
                opened = webbrowser.open_new_tab(path.as_uri())
            except Exception as error:
                self.app.call_from_thread(
                    self.notify,
                    f"{type(error).__name__}: {error}",
                    title="Interactive report unavailable",
                    severity="error",
                )
                return
            message = f"Saved to {path}"
            if not opened:
                message += " (no graphical browser was available; open this file manually)"
            self.app.call_from_thread(self.notify, message, title=title)

        Thread(target=generate, daemon=True, name="lambdaforge-tui-report").start()

    @staticmethod
    def _report_path(name: str, suffix: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", name).strip("-") or "study"
        directory = Path.cwd() / ".lambdaforge" / "reports"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{safe}-{suffix}.html"


class ExplanationDialog(ModalScreen[None]):
    """Readable contextual help for one evidence view."""

    def __init__(self, title: str, text: str) -> None:
        super().__init__()
        self.title_text = title
        self.help_text = text

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-card wide-modal"):
            yield Label(self.title_text, classes="modal-title")
            yield VerticalScroll(Static(self.help_text, classes="workspace-panel"))
            with Horizontal(classes="modal-actions"):
                yield Button("Close", id="explanation-close", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "explanation-close":
            self.dismiss(None)


class StudyWorkspace(ResearchWorkspace):
    """Scientific Study overview, Trials, HPO, analysis, resources and logs."""

    BINDINGS = [*ResearchWorkspace.BINDINGS, ("right", "open_selected", "Open selected")]

    def __init__(self, work: Mapping[str, Any], services: Any) -> None:
        self.work = dict(work)
        self.study = dict(work.get("study") or {})
        self.services = services
        self.selected_trial: int | None = None
        self.analysis: dict[str, Any] | None = None
        self._visible_candidates: list[Mapping[str, Any]] = []
        self._hpo_parameters: list[str] = []
        self._hpo_actions: list[Mapping[str, Any]] = []
        self._analysis_findings: list[Mapping[str, Any]] = []
        self._persisted_hpo_actions: list[Mapping[str, Any]] | None = None
        self._refreshing = False
        self._analysis_loading = False
        self._analysis_loaded_at = 0.0
        self._cancel_running = False
        self._last_refresh_error: str | None = None
        super().__init__(f"Studies / {work.get('name', 'Study')}")

    @property
    def job_id(self) -> str:
        return str(self.work.get("study_job_id") or self.work.get("primary_job_id", ""))

    def compose_workspace(self) -> ComposeResult:
        yield Static(id="study-header", classes="workspace-header")
        with Horizontal(id="study-control-bar"):
            yield Button("Cancel Study", id="study-cancel", variant="warning")
            yield Static("", id="study-action-status", classes="freshness-line")
        with Vertical(id="study-loading", classes="workspace-loading"):
            yield LoadingIndicator()
            yield Label(
                "Loading the persisted Study snapshot from the execution cluster…",
                id="study-loading-message",
            )
        with TabbedContent(initial="study-overview", id="study-tabs"):
            with TabPane("Overview", id="study-overview"):
                with VerticalScroll():
                    with Horizontal(classes="workspace-actions compact-actions"):
                        yield Button("? Explain", id="study-overview-help", flat=True)
                        yield Button("↗ Interactive HTML", id="study-overview-export", flat=True)
                    with Grid(classes="study-overview-grid"):
                        yield Static(id="study-overview-state", classes="metric-card")
                        yield Static(id="study-overview-trials", classes="metric-card")
                        yield Static(id="study-overview-best", classes="metric-card")
                        yield Static(id="study-overview-runtime", classes="metric-card")
                    yield StudyOverviewDashboard(id="study-overview-dashboard")
                    yield Label("Leading parameters", classes="section-title")
                    yield DataTable(
                        id="study-best-parameters", cursor_type="row", zebra_stripes=True
                    )
                    yield Static(id="study-overview-content", classes="detail-panel")
            with TabPane("Trials", id="study-trials"):
                with Horizontal(classes="workspace-actions"):
                    yield Input(
                        placeholder="Filter trial, state or parameter…",
                        id="trial-filter",
                    )
                    yield Select(
                        (
                            ("Objective", "objective"),
                            ("Trial number", "trial"),
                            ("State", "state"),
                            ("Seed count", "seeds"),
                        ),
                        value="objective",
                        id="trial-sort",
                    )
                yield DataTable(id="trial-table", cursor_type="row", zebra_stripes=True)
                yield Static(
                    "★ best · ◆ Pareto · † partial/censored (not a final seed score)",
                    classes="legend",
                )
            with TabPane("HPO", id="study-hpo"):
                with Horizontal(classes="workspace-actions compact-actions"):
                    yield Button("? Explain HPO", id="study-hpo-help", flat=True)
                    yield Button("↗ Interactive HTML", id="study-hpo-export", flat=True)
                with Grid(classes="hpo-summary-grid"):
                    yield Static(id="hpo-strategy-card", classes="metric-card")
                    yield Static(id="hpo-objective-card", classes="metric-card")
                    yield Static(id="hpo-evidence-card", classes="metric-card")
                    yield Static(id="hpo-scheduler-card", classes="metric-card")
                with TabbedContent(initial="hpo-parameters-pane", id="hpo-tabs"):
                    with TabPane("Parameters", id="hpo-parameters-pane"):
                        with Vertical(id="hpo-analysis-loading", classes="inline-loading"):
                            yield LoadingIndicator()
                            yield Label(
                                "Loading persisted response, uncertainty and coverage evidence…",
                                id="hpo-loading-message",
                            )
                        yield DataTable(
                            id="hpo-parameter-table", cursor_type="row", zebra_stripes=True
                        )
                        yield Static(
                            "Open a parameter for its response curve, uncertainty, coverage and interactions.",
                            classes="legend",
                        )
                    with TabPane("Action history", id="hpo-actions-pane"):
                        yield DataTable(
                            id="hpo-action-table", cursor_type="row", zebra_stripes=True
                        )
                        yield Static(id="hpo-action-preview", classes="detail-panel")
                    with TabPane("Controller", id="hpo-controller-pane"):
                        yield VerticalScroll(Static(id="study-hpo-content"))
            with TabPane("Analysis", id="study-analysis"):
                with Horizontal(classes="workspace-actions compact-actions"):
                    yield Button("? Explain analysis", id="study-analysis-help", flat=True)
                    yield Button("↗ Interactive HTML", id="study-analysis-export", flat=True)
                with TabbedContent(initial="analysis-summary"):
                    with TabPane("Summary", id="analysis-summary"):
                        yield VerticalScroll(Static(id="study-analysis-content"))
                    with TabPane("Parameters", id="analysis-parameters"):
                        yield DataTable(
                            id="analysis-parameter-table",
                            cursor_type="row",
                            zebra_stripes=True,
                        )
                        yield Static(id="analysis-parameter-preview", classes="detail-panel")
                    with TabPane("Interactions", id="analysis-interactions"):
                        yield DataTable(
                            id="analysis-interaction-table",
                            cursor_type="row",
                            zebra_stripes=True,
                        )
                        yield Label("Joint predictive-gain matrix", classes="section-title")
                        yield DataTable(
                            id="analysis-interaction-matrix",
                            cursor_type="cell",
                            zebra_stripes=True,
                        )
                        yield Static(
                            "Select a pair above to generate an interactive heatmap and numeric 3D "
                            f"surface for {objective_display_name(self.study.get('objective', {}))}.",
                            classes="legend",
                        )
                    with TabPane("Coverage", id="analysis-coverage"):
                        yield CoverageDashboard(id="analysis-coverage-dashboard")
                    with TabPane("Seeds", id="analysis-seeds"):
                        with Grid(classes="analysis-seed-grid"):
                            yield Static(id="analysis-seed-status", classes="metric-card")
                            yield Static(id="analysis-seed-winner", classes="metric-card")
                            yield Static(id="analysis-seed-evidence", classes="metric-card")
                        yield Static(id="analysis-seed-content", classes="detail-panel")
                    with TabPane("Pareto", id="analysis-pareto"):
                        yield Label("Scientific objective front", classes="section-title")
                        yield DataTable(
                            id="analysis-scientific-pareto",
                            cursor_type="row",
                            zebra_stripes=True,
                        )
                        yield Label(
                            "Objective / per-Run resource trade-offs", classes="section-title"
                        )
                        yield DataTable(
                            id="analysis-resource-pareto",
                            cursor_type="row",
                            zebra_stripes=True,
                        )
                        yield Static(id="analysis-pareto-content", classes="legend")
                    with TabPane("Findings", id="analysis-findings"):
                        yield DataTable(
                            id="analysis-finding-table",
                            cursor_type="row",
                            zebra_stripes=True,
                        )
                        yield Static(id="analysis-finding-content", classes="detail-panel")
            with TabPane("Resources", id="study-resources"):
                yield VerticalScroll(Static(id="study-resource-content"))
            with TabPane("Logs", id="study-logs"):
                yield RichLog(id="study-log-content", wrap=False, auto_scroll=True)

    def on_mount(self) -> None:
        terminal = str(self.work.get("state", "unknown")) in {
            "succeeded",
            "failed",
            "cancelled",
            "timeout",
        }
        self.query_one("#study-cancel", Button).disabled = terminal or not bool(self._selector)
        self.query_one("#study-loading").display = not bool(self.study)
        self.query_one("#study-tabs").display = bool(self.study)
        self.query_one("#hpo-parameter-table").display = False
        self._render_workspace()
        self.set_interval(2.0, self._refresh_if_active)
        if not self.study:
            self._refresh_study(force=True)
        self._load_analysis(force=True)
        self._load_study_logs()

    def _render_workspace(self) -> None:
        if not self.study:
            self.query_one("#study-header", Static).update(
                f"{self.work.get('name', 'Study')}  ·  "
                f"{str(self.work.get('state', 'unknown')).upper()}\n"
                "Retrieving persisted Study telemetry; no empty table is being interpreted yet."
            )
            return
        counts = self.study.get("counts", {})
        objective = self.study.get("objective", {})
        candidates = [
            value for value in self.study.get("candidates", ()) if isinstance(value, Mapping)
        ]
        best_candidate = next(
            (value for value in candidates if value.get("selection_objective") is not None), None
        )
        ranked = sorted(
            (value for value in candidates if value.get("selection_objective") is not None),
            key=lambda value: float(value["selection_objective"]),
            reverse=str(objective.get("mode", "max")) == "max",
        )
        best_candidate = ranked[0] if ranked else best_candidate
        self.query_one("#study-header", Static).update(
            f"{self.work.get('name', 'Study')}  ·  {str(self.work.get('state', 'unknown')).upper()}\n"
            f"{self.study.get('strategy', 'Study')} · {self.work.get('cluster', 'local')}   "
            f"best {objective_display_name(objective)} "
            f"{format_value((best_candidate or {}).get('selection_objective'))}   "
            f"candidates {counts.get('candidates', len(candidates))}   "
            f"active {counts.get('active_runs', 0)}   waiting {counts.get('queued_runs', 0)}   "
            f"pruned {counts.get('pruned_runs', 0)}   "
            f"GPU time {format_duration(self.study.get('cost', {}).get('gpu_seconds'))}"
        )
        attention = []
        if counts.get("queued_runs", 0):
            attention.append(f"⚠ {counts['queued_runs']} Run(s) waiting for admission")
        if best_candidate is not None:
            attention.append(f"★ Trial {best_candidate.get('trial')} currently leads selection")
        attention.append(f"† {counts.get('pruned_runs', 0)} censored/pruned Runs retained")
        self.query_one("#study-overview-state", Static).update(
            "[b]STATE[/b]\n"
            f"{str(self.work.get('state', 'unknown')).upper()}\n"
            f"{self.study.get('strategy', 'Study')} · {self.work.get('cluster', 'local')}"
        )
        self.query_one("#study-overview-trials", Static).update(
            "[b]TRIALS & RUNS[/b]\n"
            f"{counts.get('candidates', len(candidates))} candidates · "
            f"{counts.get('completed_runs', 0)} completed\n"
            f"{counts.get('active_runs', 0)} active · {counts.get('queued_runs', 0)} waiting"
        )
        self.query_one("#study-overview-best", Static).update(
            "[b]CURRENT LEADER[/b]\n"
            f"Trial {(best_candidate or {}).get('trial', 'unavailable')}\n"
            f"{objective_display_name(objective)}: "
            f"{format_value((best_candidate or {}).get('selection_objective'))}"
        )
        cost = self.study.get("cost", {})
        self.query_one("#study-overview-runtime", Static).update(
            "[b]COMPUTE[/b]\n"
            f"GPU time {format_duration(cost.get('gpu_seconds'))}\n"
            f"Wall time {format_duration(cost.get('duration_seconds'))} · "
            f"{counts.get('pruned_runs', 0)} pruned"
        )
        self.query_one("#study-overview-content", Static).update(
            "ATTENTION\n" + "\n".join(attention)
        )
        self.query_one("#study-overview-dashboard", StudyOverviewDashboard).show_study(self.study)
        best_parameters = self.query_one("#study-best-parameters", DataTable)
        best_parameters.clear(columns=True)
        best_parameters.add_columns("Parameter", "Value")
        for name, value in sorted((best_candidate or {}).get("parameters", {}).items()):
            best_parameters.add_row(str(name).replace("_", " ").title(), format_value(value))
        self._populate_trials(candidates)
        self.query_one("#study-hpo-content", Static).update(self._hpo_text())
        self._update_hpo_summary()
        self._populate_hpo_actions()
        self.query_one("#study-resource-content", Static).update(self._resource_text())

    def _populate_trials(self, candidates: Sequence[Mapping[str, Any]]) -> None:
        query = self.query_one("#trial-filter", Input).value.strip().lower()
        if query:
            candidates = [
                value
                for value in candidates
                if query
                in " ".join(
                    (
                        str(value.get("trial", "")),
                        str(value.get("state", "")),
                        json.dumps(value.get("parameters", {}), sort_keys=True),
                    )
                ).lower()
            ]
        order = str(self.query_one("#trial-sort", Select).value)
        objective_mode = str(self.study.get("objective", {}).get("mode", "max"))
        if order == "trial":
            candidates = sorted(candidates, key=lambda value: int(value.get("trial", -1)))
        elif order == "state":
            candidates = sorted(candidates, key=lambda value: str(value.get("state", "")))
        elif order == "seeds":
            candidates = sorted(
                candidates,
                key=lambda value: int(value.get("selection_seed_count", 0)),
                reverse=True,
            )
        else:
            comparable_candidates = [
                value for value in candidates if value.get("selection_objective") is not None
            ]
            incomplete_candidates = [
                value for value in candidates if value.get("selection_objective") is None
            ]
            candidates = [
                *sorted(
                    comparable_candidates,
                    key=lambda value: float(value["selection_objective"]),
                    reverse=objective_mode == "max",
                ),
                *incomplete_candidates,
            ]
        self._visible_candidates = list(candidates)
        comparable = [value for value in candidates if value.get("selection_objective") is not None]
        best_trial = None
        if comparable:
            best_trial = (max if objective_mode == "max" else min)(
                comparable, key=lambda value: float(value["selection_objective"])
            ).get("trial")
        table = self.query_one("#trial-table", DataTable)
        table.clear(columns=True)
        table.add_columns(
            "Mark", "Trial", "Selection", "Current", "Best", "SE", "Seeds", "Epoch", "GPU", "State"
        )
        for candidate in candidates:
            runs = [value for value in candidate.get("runs", ()) if isinstance(value, Mapping)]
            active = next((run for run in runs if run.get("state") == "running"), None)
            markers = []
            if candidate.get("trial") == best_trial:
                markers.append("★")
            if candidate.get("pareto_optimal"):
                markers.append("◆")
            if candidate.get("partially_censored"):
                markers.append("†")
            censored = "†" if candidate.get("partially_censored") else ""
            table.add_row(
                " ".join(markers),
                str(candidate.get("trial")),
                f"{format_value(candidate.get('selection_objective'))}{censored}",
                format_value(candidate.get("current_objective")),
                format_value(candidate.get("best_objective")),
                format_value(candidate.get("selection_standard_error")),
                str(candidate.get("selection_seed_count", len(runs))),
                str(max((int(run.get("latest_step", 0) or 0) for run in runs), default=0)),
                str((active or {}).get("gpu_index", "-")),
                str(candidate.get("state", "unknown")).upper(),
                key=str(candidate.get("trial")),
            )
        if self.selected_trial is not None:
            for row, candidate in enumerate(candidates):
                if int(candidate.get("trial", -1)) == self.selected_trial:
                    table.move_cursor(row=row)
                    break

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "trial-filter":
            candidates = [
                value for value in self.study.get("candidates", ()) if isinstance(value, Mapping)
            ]
            self._populate_trials(candidates)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "trial-sort":
            candidates = [
                value for value in self.study.get("candidates", ()) if isinstance(value, Mapping)
            ]
            self._populate_trials(candidates)

    def _hpo_text(self) -> str:
        controller = self.study.get("controller", {})
        latest = controller.get("last", {}) if isinstance(controller, Mapping) else {}
        scheduler = controller.get("scheduler", {}) if isinstance(controller, Mapping) else {}
        belief = self.study.get("surrogate_belief") or controller.get("surrogate_belief", {})
        return (
            "CONTROLLER\n"
            f"Last action          {latest.get('action', 'unavailable')}\n"
            f"Reason               {latest.get('reason', latest.get('why', 'unavailable'))}\n"
            f"Controller value     {format_value(latest.get('controller_value'))}\n"
            f"Expected cost        {format_value(latest.get('expected_cost'))}\n\n"
            "SCHEDULER\n"
            f"Slots total          {scheduler.get('slots_total', 'unavailable')}\n"
            f"Active               {scheduler.get('slots_active', 0)}\n"
            f"Available            {scheduler.get('slots_available', 'unavailable')}\n"
            f"Queued               {scheduler.get('queued', 0)}\n"
            f"Paused / preempted   {scheduler.get('paused', 0)} / {scheduler.get('preempted', 0)}\n\n"
            "SURROGATE BELIEF\n"
            f"Backend              {(belief or {}).get('backend', 'unavailable')}\n"
            f"Observations         {(belief or {}).get('observations', 'unavailable')}\n"
            f"Target fidelity      {(belief or {}).get('target_fidelity', 'unavailable')}\n\n"
            "The controller view exposes exact persisted state. Action history is shown in its "
            "own table; parameter effects are predictive associations, not causal claims."
        )

    def _update_hpo_summary(self) -> None:
        controller = self.study.get("controller", {})
        latest = controller.get("last", {}) if isinstance(controller, Mapping) else {}
        scheduler = controller.get("scheduler", {}) if isinstance(controller, Mapping) else {}
        belief = self.study.get("surrogate_belief") or controller.get("surrogate_belief", {})
        counts = self.study.get("counts", {})
        objective = self.study.get("objective", {})
        stop_summary = self._controller_stop_summary(controller)
        controller_status = stop_summary or f"Last: {latest.get('action', 'No decision yet')}"
        self.query_one("#hpo-strategy-card", Static).update(
            "[b]STRATEGY[/b]\n"
            f"{self.study.get('strategy', 'adaptive')}\n"
            f"{controller_status}"
        )
        self.query_one("#hpo-objective-card", Static).update(
            "[b]OBJECTIVE[/b]\n"
            f"{objective_display_name(objective)}\n"
            f"Direction: {objective.get('mode', 'unavailable')}"
        )
        self.query_one("#hpo-evidence-card", Static).update(
            "[b]EVIDENCE[/b]\n"
            f"{counts.get('candidates', 0)} candidates · {counts.get('completed_runs', 0)} complete\n"
            f"{(belief or {}).get('observations', 'unavailable')} surrogate observations"
        )
        self.query_one("#hpo-scheduler-card", Static).update(
            "[b]SCHEDULER[/b]\n"
            f"{scheduler.get('slots_active', 0)} active / "
            f"{scheduler.get('slots_total', 'unavailable')} slots\n"
            f"{scheduler.get('queued', counts.get('queued_runs', 0))} waiting"
        )

    @staticmethod
    def _controller_stop_summary(controller: Any) -> str | None:
        """Render a durable terminal reason, including older snapshots missing FINISH."""
        if not isinstance(controller, Mapping):
            return None
        recent = [value for value in controller.get("recent", ()) if isinstance(value, Mapping)]
        last = controller.get("last", {})
        candidates = [last] if isinstance(last, Mapping) else []
        candidates.extend(reversed(recent))
        terminal = next(
            (
                value
                for value in candidates
                if value.get("action") in {"FINISH", "STOP_PROPOSING"}
            ),
            None,
        )
        if terminal is None:
            return None
        reason = str(terminal.get("reason", "terminal controller decision"))
        return "Stop: " + reason.replace("-", " ").replace("_", " ").capitalize()

    def _populate_hpo_actions(self) -> None:
        controller = self.study.get("controller", {})
        recent = (
            self._persisted_hpo_actions
            if self._persisted_hpo_actions is not None
            else controller.get("recent", ()) if isinstance(controller, Mapping) else ()
        )
        self._hpo_actions = [value for value in recent if isinstance(value, Mapping)]
        table = self.query_one("#hpo-action-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Action", "Trial", "Reason")
        for index, action in enumerate(self._hpo_actions):
            table.add_row(
                str(action.get("action", action.get("decision", "unavailable"))).replace("_", " ").title(),
                str(action.get("trial", action.get("candidate", "—"))),
                str(action.get("reason", action.get("why", "No reason persisted."))),
                key=str(index),
            )
        if self._hpo_actions:
            self._render_hpo_action_preview(0)
        else:
            self.query_one("#hpo-action-preview", Static).update(
                "No controller decisions have been persisted yet."
            )

    def _populate_hpo_parameters(self) -> None:
        if self.analysis is None:
            return
        space = self.analysis.get("search_space", {})
        importance = self.analysis.get("parameter_importance", {})
        coverage = self.analysis.get("coverage", {})
        marginal = coverage.get("marginal", {}) if isinstance(coverage, Mapping) else {}
        responses = self.analysis.get("response_curves", {})
        live = self.analysis.get("live_hpo", {})
        live_parameters = live.get("parameters", ()) if isinstance(live, Mapping) else ()
        live_by_name = {
            str(value.get("parameter")): value
            for value in live_parameters
            if isinstance(value, Mapping) and value.get("parameter") is not None
        }
        names = list(space) if isinstance(space, Mapping) else []
        if not names and isinstance(importance, Mapping):
            names = list(importance)
        names.extend(name for name in live_by_name if name not in names)
        self._hpo_parameters = [str(name) for name in names]
        table = self.query_one("#hpo-parameter-table", DataTable)
        table.clear(columns=True)
        table.add_columns(
            "Parameter", "Search space", "Observed", "Promising", "Importance", "Confidence"
        )
        for name in self._hpo_parameters:
            rule = space.get(name, {}) if isinstance(space, Mapping) else {}
            observed = marginal.get(name, {}) if isinstance(marginal, Mapping) else {}
            response = responses.get(name, {}) if isinstance(responses, Mapping) else {}
            detail = importance.get(name, {}) if isinstance(importance, Mapping) else {}
            live_detail = live_by_name.get(name, {})
            confidence = live_detail.get("confidence")
            confidence_label = live_detail.get(
                "confidence_label",
                detail.get("reliability", "low") if isinstance(detail, Mapping) else "low",
            )
            table.add_row(
                name.replace("_", " ").title(),
                self._parameter_domain(rule),
                self._observed_domain(observed),
                self._promising_region(
                    response,
                    self.study.get("objective", {}),
                    live_detail=live_detail,
                ),
                format_value(detail.get("importance") if isinstance(detail, Mapping) else None),
                (
                    f"{float(confidence):.0%} · {str(confidence_label).title()}"
                    if isinstance(confidence, int | float) and not isinstance(confidence, bool)
                    else str(confidence_label).title()
                ),
                key=name,
            )
        loading = self.query_one("#hpo-analysis-loading")
        loading.display = False
        table.display = True

    @staticmethod
    def _parameter_domain(rule: Any) -> str:
        if not isinstance(rule, Mapping):
            return "Not persisted"
        values = rule.get("values")
        if isinstance(values, Sequence) and not isinstance(values, str | bytes):
            return ", ".join(map(str, values))
        if rule.get("kind") == "numeric":
            scale = f" · {rule.get('scale')}" if rule.get("scale") not in {None, "linear"} else ""
            return f"{format_value(rule.get('low'))} … {format_value(rule.get('high'))}{scale}"
        return "Not persisted"

    @staticmethod
    def _observed_domain(observed: Any) -> str:
        if not isinstance(observed, Mapping):
            return "No observations"
        raw_range = observed.get("observed_range")
        if isinstance(raw_range, Sequence) and not isinstance(raw_range, str | bytes):
            return " … ".join(format_value(value) for value in raw_range)
        levels = observed.get("observed_levels", observed.get("levels"))
        if isinstance(levels, Sequence) and not isinstance(levels, str | bytes):
            return ", ".join(map(str, levels))
        counts = observed.get("counts")
        if isinstance(counts, Mapping):
            return ", ".join(map(str, counts))
        return "No observations"

    @staticmethod
    def _promising_region(
        response: Any,
        objective: Mapping[str, Any],
        *,
        live_detail: Mapping[str, Any] | None = None,
    ) -> str:
        response = response if isinstance(response, Mapping) else {}
        region = response.get("best_supported_region")
        if isinstance(region, Mapping):
            lower, upper = region.get("low", region.get("lower")), region.get("high", region.get("upper"))
            if lower is not None or upper is not None:
                return f"{format_value(lower)} … {format_value(upper)}"
        point = response.get("best_supported_point")
        if isinstance(point, Mapping):
            return format_value(point.get("x", point.get("category")))
        points = [value for value in response.get("points", ()) if isinstance(value, Mapping)]
        comparable = [value for value in points if isinstance(value.get("predicted_objective"), int | float)]
        if comparable:
            chosen = (max if objective.get("mode", "max") == "max" else min)(
                comparable, key=lambda value: float(value["predicted_objective"])
            )
            return format_value(chosen.get("x", chosen.get("category")))
        if live_detail:
            value = live_detail.get("best_observed_value")
            if value is not None:
                return f"Near {format_value(value)} (observed)"
            threshold = live_detail.get("possible_threshold")
            if isinstance(threshold, Mapping) and threshold.get("value") is not None:
                return f"Around {format_value(threshold.get('value'))}"
        return "Still learning"

    def _render_hpo_action_preview(self, row: int) -> None:
        if 0 <= row < len(self._hpo_actions):
            action = self._hpo_actions[row]
            self.query_one("#hpo-action-preview", Static).update(
                f"{action.get('action', action.get('decision', 'Decision'))} · "
                f"Trial {action.get('trial', action.get('candidate', '—'))}\n"
                f"{action.get('reason', action.get('why', 'No reason persisted.'))}\n"
                "Open this row to inspect all persisted controller inputs and thresholds."
            )

    def _resource_text(self) -> str:
        admission = self.study.get("admission", {})
        current = admission.get("current", {}) if isinstance(admission, Mapping) else {}
        requested = current.get("requested", {}) if isinstance(current, Mapping) else {}
        devices = current.get("devices", ()) if isinstance(current, Mapping) else ()
        lines = [
            "REQUESTED",
            f"GPUs                  {requested.get('gpus', current.get('gpu_count', 'unavailable'))}",
            f"runs_per_gpu          {requested.get('runs_per_gpu', current.get('runs_per_gpu', 'unavailable'))}",
            f"max_parallel          {requested.get('max_parallel', current.get('max_parallel', 'unavailable'))}",
            f"potential slots       {current.get('potential_slots', 'unavailable')}",
            "",
            "CURRENT ADMISSION",
            f"status                {current.get('status', 'unavailable')}",
            f"limiting resource     {current.get('limiting_resource', 'unavailable')}",
            f"reason                {current.get('reason', 'No admission evidence persisted.')}",
        ]
        for device in devices:
            if isinstance(device, Mapping):
                lines.extend(
                    (
                        "",
                        f"GPU {device.get('index', '?')}",
                        f"admitted              {device.get('active', 0)} / {device.get('limit', '?')}",
                        f"free VRAM             {format_value(device.get('free_bytes'))} bytes",
                        f"required next Run     {format_value(device.get('required_bytes'))} bytes",
                        f"status                {device.get('status', 'unknown')}",
                        f"reason                {device.get('reason', '-')}",
                    )
                )
        return "\n".join(lines)

    def _load_analysis(self, *, force: bool = False) -> None:
        if self._analysis_loading or (
            not force and time.monotonic() - self._analysis_loaded_at < 10.0
        ):
            return
        execution = self.work.get("execution_id")
        self._analysis_loading = True

        def load() -> None:
            value = None
            analysis_error: Exception | None = None
            if execution:
                try:
                    value = self.services.result_analysis(str(execution))
                except Exception as error:
                    analysis_error = error
            live_loader = getattr(self.services, "live_study_analysis", None)
            if value is None and callable(live_loader):
                try:
                    value = live_loader(self.study, job_id=self.job_id or None)
                except Exception as error:
                    analysis_error = error
            if value is None:
                reason = (
                    f"{type(analysis_error).__name__}: {analysis_error}"
                    if analysis_error is not None
                    else "No final or live Study Analysis provider is available."
                )
                self.app.call_from_thread(self._show_analysis_error, reason)
                return
            actions = None
            action_loader = getattr(self.services, "study_actions", None)
            if callable(action_loader) and self.job_id:
                try:
                    actions = action_loader(self.job_id)
                except Exception:
                    # The bounded controller tail is already a truthful fallback. A provider
                    # outage during periodic refresh must not create recurring notifications.
                    actions = None
            self.app.call_from_thread(self._apply_analysis, value, actions)

        Thread(target=load, daemon=True, name="lambdaforge-tui-study-analysis").start()

    def _show_analysis_error(self, message: str) -> None:
        self._analysis_loading = False
        self.query_one("#study-analysis-content", Static).update(
            "Analysis is not available yet.\n\n" + message
        )
        self.query_one("#hpo-loading-message", Label).update(
            "HPO parameter analysis is unavailable: " + message
        )
        self.query_one("#hpo-analysis-loading LoadingIndicator").display = False

    def _apply_analysis(
        self,
        value: Mapping[str, Any],
        actions: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self._analysis_loading = False
        self._analysis_loaded_at = time.monotonic()
        self.analysis = dict(value)
        current_controller = self.study.get("controller", {})
        current_tail = (
            current_controller.get("recent", ())
            if isinstance(current_controller, Mapping)
            else ()
        )
        if actions is not None and (actions or not current_tail):
            self._persisted_hpo_actions = [dict(action) for action in actions]
            self._populate_hpo_actions()
        self._populate_hpo_parameters()
        winner = value.get("winner", {})
        seeds = value.get("seed_analysis", {})
        surrogate = value.get("surrogate", {})
        importance = value.get("parameter_importance", {})
        findings = value.get("findings", ())
        parameter_lines = (
            [
                f"{name:24} global={float(detail.get('importance', 0)):.3f}  "
                f"reliability={detail.get('reliability', 'low')}"
                for name, detail in sorted(
                    importance.items(),
                    key=lambda item: float(item[1].get("importance", 0)),
                    reverse=True,
                )
                if isinstance(detail, Mapping)
            ]
            if isinstance(importance, Mapping)
            else []
        )
        finding_lines = [
            f"{str(item.get('severity', 'info')).upper():8} {item.get('title')} "
            f"[{item.get('reliability', 'low')}]\n  {item.get('statement')}"
            for item in findings
            if isinstance(item, Mapping)
        ]
        self.query_one("#study-analysis-content", Static).update(
            "SUMMARY\n"
            f"Winner               {(winner.get('screening_winner') or {}).get('trial', 'unavailable')}\n"
            f"Confirmation         {winner.get('confirmation_status', 'unavailable')}\n"
            f"Surrogate quality    {surrogate.get('quality', 'unavailable')}\n"
            f"CV method            {surrogate.get('validation_method', 'unavailable')}\n"
            f"Seed stability       {seeds.get('status', 'unavailable')}\n"
            f"Rank-one fraction    {format_value(seeds.get('winner_rank_one_fraction'))}\n\n"
            "PARAMETERS · predictive association, not causal effect\n"
            + ("\n".join(parameter_lines) or "Insufficient parameter evidence.")
            + "\n\nFINDINGS\n"
            + ("\n\n".join(finding_lines) or "No evidence-backed findings.")
        )
        parameters = self.query_one("#analysis-parameter-table", DataTable)
        parameters.clear(columns=True)
        parameters.add_columns("Parameter", "Global", "Top region", "Reliability", "Evidence")
        top_importance = value.get("top_region_importance", {})
        if isinstance(importance, Mapping):
            for name, detail in sorted(
                importance.items(),
                key=lambda item: float(item[1].get("importance", 0)),
                reverse=True,
            ):
                if not isinstance(detail, Mapping):
                    continue
                parameters.add_row(
                    str(name),
                    format_value(detail.get("importance")),
                    format_value(
                        top_importance.get(name, {}).get("importance")
                        if isinstance(top_importance, Mapping)
                        and isinstance(top_importance.get(name), Mapping)
                        else None
                    ),
                    str(detail.get("reliability", "low")),
                    str(detail.get("support", detail.get("observations", "-"))),
                    key=str(name),
                )
        interactions = self.query_one("#analysis-interaction-table", DataTable)
        interactions.clear(columns=True)
        interactions.add_columns(
            "Parameters", "Predictive gain", "Reliability", "Support", "Interactive"
        )
        raw_interactions = value.get("interactions", ())
        if isinstance(raw_interactions, Mapping):
            raw_interactions = raw_interactions.get("pairs", ())
        for interaction in raw_interactions if isinstance(raw_interactions, Sequence) else ():
            if isinstance(interaction, Mapping):
                pair = interaction.get("parameters", interaction.get("pair", ()))
                if not pair and interaction.get("left") is not None:
                    pair = (interaction.get("left"), interaction.get("right"))
                interactions.add_row(
                    " × ".join(map(str, pair)),
                    format_value(
                        interaction.get(
                            "predictive_gain",
                            interaction.get("gain", interaction.get("importance")),
                        )
                    ),
                    str(interaction.get("reliability", "low")),
                    str(interaction.get("support", "-")),
                    "Enter ↗",
                )
        self._populate_interaction_matrix(value.get("interactions", {}))
        coverage = value.get("coverage", {})
        self.query_one("#analysis-coverage-dashboard", CoverageDashboard).show_coverage(
            coverage if isinstance(coverage, Mapping) else {}
        )
        self._populate_seed_analysis(seeds if isinstance(seeds, Mapping) else {})
        resources = value.get("resources", {})
        scientific_pareto = value.get("pareto", {})
        self._populate_pareto(
            scientific_pareto if isinstance(scientific_pareto, Mapping) else {},
            resources if isinstance(resources, Mapping) else {},
            value.get("candidates", ()),
        )
        self._populate_findings(findings)
        if parameters.row_count:
            parameters.move_cursor(row=0)
            self._render_parameter_preview(0)

    def _populate_seed_analysis(self, seeds: Mapping[str, Any]) -> None:
        status = str(seeds.get("status", "not observed"))
        self.query_one("#analysis-seed-status", Static).update(
            "[b]EMPIRICAL STABILITY[/b]\n"
            f"{status.replace('_', ' ').upper()}\n"
            f"{str(seeds.get('reason', 'No limiting reason')).replace('_', ' ')}"
        )
        self.query_one("#analysis-seed-winner", Static).update(
            "[b]LEADING TRIAL[/b]\n"
            f"Trial {seeds.get('winner_trial', '—')}\n"
            f"Rank-one {format_value(seeds.get('winner_rank_one_fraction'))}"
        )
        self.query_one("#analysis-seed-evidence", Static).update(
            "[b]REPEATED EVIDENCE[/b]\n"
            f"{seeds.get('repeated_seed_candidate_count', 0)} / "
            f"{seeds.get('relevant_candidate_count', 0)} leading candidates\n"
            f"{seeds.get('replicates', 0)} bootstrap replicates"
        )
        model = seeds.get("model_based", {})
        model_status = (
            model.get("status", "not persisted")
            if isinstance(model, Mapping)
            else "not persisted"
        )
        self.query_one("#analysis-seed-content", Static).update(
            f"{seeds.get('interpretation', 'More repeated seeds are required for empirical stability.')}\n"
            f"Model-based uncertainty: {model_status}."
        )

    def _populate_pareto(
        self,
        scientific: Mapping[str, Any],
        resources: Mapping[str, Any],
        raw_candidates: Any,
    ) -> None:
        candidate_values = (
            [value for value in raw_candidates if isinstance(value, Mapping)]
            if isinstance(raw_candidates, Sequence)
            and not isinstance(raw_candidates, str | bytes)
            else []
        )
        candidates = {str(value.get("trial")): value for value in candidate_values}
        components = [str(value) for value in scientific.get("objective_components", ())]
        front = [str(value) for value in scientific.get("front", ())]
        scientific_table = self.query_one("#analysis-scientific-pareto", DataTable)
        scientific_table.clear(columns=True)
        if components:
            scientific_table.add_columns(
                "Trial", *(name.replace("_", " ").title() for name in components)
            )
            for trial in front:
                values = candidates.get(trial, {}).get("objective_components", {})
                values = values if isinstance(values, Mapping) else {}
                scientific_table.add_row(
                    trial,
                    *(format_value(values.get(name)) for name in components),
                )
        else:
            scientific_table.add_columns("Status", "Interpretation")
            scientific_table.add_row(
                "Not applicable",
                "The selection objective has one scalar component.",
            )

        intrinsic = resources.get("intrinsic_per_run", {})
        intrinsic = intrinsic if isinstance(intrinsic, Mapping) else {}
        pareto = intrinsic.get("pareto", resources.get("pareto", {}))
        pareto = pareto if isinstance(pareto, Mapping) else {}
        pareto_trials = {
            str(trial)
            for values in pareto.values()
            if isinstance(values, Sequence) and not isinstance(values, str | bytes)
            for trial in values
        }
        targets = intrinsic.get("targets", resources.get("targets", ()))
        targets = (
            targets
            if isinstance(targets, Sequence) and not isinstance(targets, str | bytes)
            else ()
        )
        resource_table = self.query_one("#analysis-resource-pareto", DataTable)
        resource_table.clear(columns=True)
        resource_table.add_columns(
            "Pareto", "Trial", "Objective", "Duration", "GPU time", "Peak VRAM"
        )
        for row in targets:
            if not isinstance(row, Mapping):
                continue
            trial = str(row.get("trial", "—"))
            resource_table.add_row(
                "◆" if trial in pareto_trials else "",
                trial,
                format_value(row.get("objective")),
                format_duration(row.get("duration_seconds")),
                format_duration(row.get("gpu_seconds")),
                format_value(row.get("peak_vram")),
            )
        self.query_one("#analysis-pareto-content", Static).update(
            "◆ lies on at least one objective/resource frontier. Costs are intrinsic per "
            "comparable Run; total controller spend remains separate."
        )

    def _populate_findings(self, raw: Any) -> None:
        self._analysis_findings = (
            [value for value in raw if isinstance(value, Mapping)]
            if isinstance(raw, Sequence) and not isinstance(raw, str | bytes)
            else []
        )
        table = self.query_one("#analysis-finding-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Severity", "Finding", "Reliability", "Support")
        for finding in self._analysis_findings:
            table.add_row(
                str(finding.get("severity", "info")).upper(),
                str(finding.get("title", "Finding")),
                str(finding.get("reliability", "low")).title(),
                str(finding.get("support", "—")),
            )
        self._render_finding_preview(0)

    def _render_finding_preview(self, row: int) -> None:
        panel = self.query_one("#analysis-finding-content", Static)
        if not 0 <= row < len(self._analysis_findings):
            panel.update("No evidence-backed findings.")
            return
        finding = self._analysis_findings[row]
        panel.update(
            f"[b]{finding.get('title', 'Finding')}[/b]\n"
            f"{finding.get('statement', 'No statement persisted.')}\n\n"
            f"Interpretation: {finding.get('interpretation', finding.get('caveat', '—'))}\n"
            f"Recommended next step: {finding.get('recommendation', '—')}"
        )

    def _populate_interaction_matrix(self, raw: Any) -> None:
        matrix_table = self.query_one("#analysis-interaction-matrix", DataTable)
        matrix_table.clear(columns=True)
        matrix = raw.get("matrix", {}) if isinstance(raw, Mapping) else {}
        if not isinstance(matrix, Mapping) or not matrix:
            matrix_table.add_column("Interaction evidence")
            matrix_table.add_row("No pairwise matrix is available yet.")
            return
        names = [str(value) for value in matrix]
        matrix_table.add_columns("Parameter", *(name.replace("_", " ") for name in names))
        for left in names:
            row = matrix.get(left, {})
            cells: list[Text] = []
            for right in names:
                value = row.get(right) if isinstance(row, Mapping) else None
                if not isinstance(value, int | float) or isinstance(value, bool):
                    cells.append(Text("—", style="dim"))
                    continue
                strength = max(0.0, min(1.0, float(value)))
                style = (
                    "bold bright_magenta"
                    if strength >= 0.5
                    else "bright_cyan"
                    if strength >= 0.2
                    else "dim"
                )
                cells.append(Text(f"{strength:.2f}", style=style))
            matrix_table.add_row(left.replace("_", " ").title(), *cells)

    def _render_parameter_preview(self, row: int) -> None:
        if self.analysis is None:
            return
        importance = self.analysis.get("parameter_importance", {})
        if not isinstance(importance, Mapping):
            return
        entries = sorted(
            importance.items(),
            key=lambda item: float(item[1].get("importance", 0)),
            reverse=True,
        )
        if not 0 <= row < len(entries):
            return
        name, detail = entries[row]
        responses = self.analysis.get("response_curves", {})
        response_detail = responses.get(name, {}) if isinstance(responses, Mapping) else {}
        response = response_detail.get("points", ()) if isinstance(response_detail, Mapping) else ()
        components = detail.get("reliability_components", {})
        lines = [
            f"{name} · predictive association, not causal effect",
            f"Reliability: {detail.get('reliability', 'low')}",
            f"Reliability components: {json.dumps(components, default=str)}",
            f"Best supported point: {response_detail.get('best_supported_point', 'unavailable')}",
            "",
            "RESPONSE",
        ]
        for point in response if isinstance(response, Sequence) else ():
            if isinstance(point, Mapping):
                coordinate = point.get("x", point.get("category", "-"))
                prediction = point.get("predicted_objective", point.get("prediction"))
                lines.append(
                    f"{format_value(coordinate):>12}  "
                    f"predicted={format_value(prediction)}  "
                    f"effect={format_value(point.get('effect'))}  "
                    f"support={point.get('support_count', point.get('support', '-'))}"
                )
        self.query_one("#analysis-parameter-preview", Static).update("\n".join(lines))

    def _refresh_if_active(self) -> None:
        self._refresh_study(force=False)

    def _refresh_study(self, *, force: bool) -> None:
        if self._refreshing or (
            not force
            and str(self.work.get("state"))
            not in {
                "running",
                "preparing",
                "staging",
                "queued",
                "unknown",
            }
        ):
            return
        if not self.job_id:
            return
        self._refreshing = True

        def load() -> None:
            try:
                value = self.services.study(self.job_id)
            except Exception as error:
                self.app.call_from_thread(
                    self._show_study_load_error,
                    f"{type(error).__name__}: {error}",
                )
            else:
                self.app.call_from_thread(self._apply_refresh, value)
            finally:
                self._refreshing = False

        Thread(target=load, daemon=True, name="lambdaforge-tui-study-refresh").start()

    def _show_study_load_error(self, message: str) -> None:
        self._last_refresh_error = message
        if self.study:
            self.query_one("#study-action-status", Static).update(
                "Live refresh unavailable · showing the last snapshot"
            )
            return
        # A Study can fail during validation/environment preparation, before the
        # worker has had any opportunity to create study telemetry.  That does
        # not make the durable Attempt lifecycle, submission-worker traceback or
        # cancellation controls unavailable.  Keep the Study workspace and open
        # its ordinary Work log instead of trapping the user behind a loader.
        self.query_one("#study-loading").display = False
        tabs = self.query_one("#study-tabs", TabbedContent)
        tabs.display = True
        tabs.active = "study-logs"
        self.query_one("#study-action-status", Static).update(
            "Study telemetry unavailable · Attempt log shown"
        )
        self.query_one("#study-header", Static).update(
            f"{self.work.get('name', 'Study')}  ·  "
            f"{str(self.work.get('state', 'unknown')).upper()}\n"
            "No Run telemetry was produced. The Logs tab contains the durable "
            "preparation/runtime diagnosis."
        )

    def _apply_refresh(self, value: Mapping[str, Any]) -> None:
        self.study = dict(value)
        self._last_refresh_error = None
        if not self._cancel_running:
            self.query_one("#study-action-status", Static).update("Live · updated just now")
        self.query_one("#study-loading").display = False
        self.query_one("#study-tabs").display = True
        self._render_workspace()
        self._load_study_logs()
        self._load_analysis()

    def _load_study_logs(self) -> None:
        if not self.job_id:
            return

        def load() -> None:
            try:
                report = self.services.work_logs(self.job_id, tail=2_000)
                text = str(report.get("text", "No scientific output yet."))
            except Exception as error:
                text = f"{type(error).__name__}: {error}"
            self.app.call_from_thread(self._replace_study_log, text)

        Thread(target=load, daemon=True, name="lambdaforge-tui-study-logs").start()

    def _replace_study_log(self, text: str) -> None:
        log = self.query_one("#study-log-content", RichLog)
        at_end = log.is_vertical_scroll_end
        y = log.scroll_y
        log.clear()
        log.write(text)
        if at_end:
            log.scroll_end(animate=False)
        else:
            log.scroll_to(y=y, animate=False)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id == "hpo-parameter-table":
            return
        if event.data_table.id == "hpo-action-table":
            self._render_hpo_action_preview(event.cursor_row)
            return
        if event.data_table.id == "analysis-parameter-table":
            self._render_parameter_preview(event.cursor_row)
            return
        if event.data_table.id == "analysis-finding-table":
            self._render_finding_preview(event.cursor_row)
            return
        if event.data_table.id != "trial-table":
            return
        candidates = self._visible_candidates
        if 0 <= event.cursor_row < len(candidates):
            self.selected_trial = int(candidates[event.cursor_row].get("trial", 0))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "trial-table":
            self._open_selected_trial(event.cursor_row)
        elif event.data_table.id == "hpo-parameter-table":
            self._open_hpo_parameter(event.cursor_row)
        elif event.data_table.id == "hpo-action-table":
            self._open_hpo_action(event.cursor_row)
        elif event.data_table.id == "analysis-interaction-table":
            self._export_analysis("pairwise")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "study-cancel":
            preview = {
                "study": self.work.get("name"),
                "execution": self._selector,
                "will_stop": "all active Attempts and descendant Run processes",
                "will_preserve": "logs, partial results, checkpoints and published datasets",
            }
            self.app.push_screen(ExactConfirmation("Cancel Study", preview), self._apply_cancel)
        elif event.button.id in {
            "study-overview-export",
            "study-hpo-export",
            "study-analysis-export",
        }:
            self._export_analysis("analysis")
        elif event.button.id == "study-overview-help":
            self._show_explanation(
                "Study overview",
                "Candidate objective compares only statistically comparable completed seed evidence. "
                "Run states count physical executions; pruned Runs are useful censored evidence but are "
                "not final seed scores. Click any plotted point or bar to read its exact value.",
            )
        elif event.button.id == "study-hpo-help":
            self._show_explanation(
                "Adaptive HPO",
                "Trials is the authored candidate budget. By default LambdaForge explores that budget "
                "adaptively; it saves compute by pruning weak learning curves and allocating additional "
                "seeds only near meaningful decisions. Confidence in this view describes how reliably "
                "the observed data explain one parameter—it is not the controller's joint acquisition "
                "probability and does not by itself prove causation or convergence.",
            )
        elif event.button.id == "study-analysis-help":
            self._show_explanation(
                "Study analysis",
                "Global importance measures predictive association across the observed search. Top-region "
                "importance repeats that calculation near the leading candidates. Reliability combines "
                "support, coverage and surrogate validation. Coverage distances are normalized mixed-space "
                "distances: smaller values mean reference points lie closer to observed candidates. "
                "Pairwise gain is the leave-one-candidate-out improvement over the best one-parameter model. "
                "Response and interaction surfaces always target the Study's declared selection objective; "
                "switching to an unrelated metric would describe a different model, not the HPO controller.",
            )

    @property
    def _selector(self) -> str:
        """Return the exact semantic Work identity before or after telemetry publication."""
        return str(
            self.work.get("work_id")
            or self.work.get("execution_id")
            or self.work.get("primary_job_id")
            or ""
        )

    def _apply_cancel(self, confirmed: bool | None) -> None:
        if not confirmed or self._cancel_running:
            return
        self._cancel_running = True
        self.query_one("#study-cancel", Button).disabled = True
        self.query_one("#study-action-status", Static).update(
            "Cancelling every active Attempt in this Study…"
        )

        def cancel() -> None:
            try:
                result = self.services.cancel_work(self._selector)
            except Exception as error:
                self.app.call_from_thread(self._cancel_failed, error)
            else:
                self.app.call_from_thread(self._cancel_complete, result)

        Thread(target=cancel, daemon=True, name="lambdaforge-tui-study-cancel").start()

    def _cancel_failed(self, error: Exception) -> None:
        self._cancel_running = False
        self.query_one("#study-cancel", Button).disabled = False
        self.query_one("#study-action-status", Static).update(
            f"Cancellation failed · {type(error).__name__}: {error}"
        )

    def _cancel_complete(self, result: Mapping[str, Any]) -> None:
        self._cancel_running = False
        self.work["state"] = "cancelled"
        self.query_one("#study-cancel", Button).disabled = True
        stopped = len(result.get("cancelled_jobs", ())) + len(
            result.get("reconciled_cancelled_jobs", ())
        )
        self.query_one("#study-action-status", Static).update(
            f"Cancellation complete · {stopped} active Job(s) stopped"
        )
        self._render_workspace()

    def _export_analysis(self, suffix: str) -> None:
        if self.analysis is None:
            self.notify("Analysis is still loading; try again when the tables are visible.", severity="warning")
            return
        analysis = dict(self.analysis)
        path = self._report_path(str(self.work.get("name", "study")), suffix)
        self._write_and_open_report(
            "Interactive Study report",
            lambda: write_html(analysis, path),
        )

    def action_open_selected(self) -> None:
        focused = self.focused
        if isinstance(focused, DataTable) and focused.id == "hpo-parameter-table":
            self._open_hpo_parameter(focused.cursor_row)
        elif isinstance(focused, DataTable) and focused.id == "hpo-action-table":
            self._open_hpo_action(focused.cursor_row)
        else:
            self._open_selected_trial(self.query_one("#trial-table", DataTable).cursor_row)

    def _open_hpo_parameter(self, row: int) -> None:
        if self.analysis is not None and 0 <= row < len(self._hpo_parameters):
            self.app.push_screen(
                HpoParameterWorkspace(
                    self.work,
                    self.study,
                    self.analysis,
                    self._hpo_parameters[row],
                )
            )

    def _open_hpo_action(self, row: int) -> None:
        if 0 <= row < len(self._hpo_actions):
            self.app.push_screen(
                HpoActionWorkspace(
                    self.work,
                    row + 1,
                    self._hpo_actions[row],
                )
            )

    def _open_selected_trial(self, row: int) -> None:
        candidates = self._visible_candidates
        if 0 <= row < len(candidates):
            self.selected_trial = int(candidates[row].get("trial", 0))
            self.app.push_screen(
                TrialWorkspace(
                    self.work,
                    candidates[row],
                    self.services,
                    objective=self.study.get("objective", {}),
                )
            )


class HpoParameterWorkspace(ResearchWorkspace):
    """Detailed evidence for one optimized parameter."""

    def __init__(
        self,
        work: Mapping[str, Any],
        study: Mapping[str, Any],
        analysis: Mapping[str, Any],
        parameter: str,
    ) -> None:
        self.work = dict(work)
        self.study = dict(study)
        self.analysis = dict(analysis)
        self.parameter = parameter
        super().__init__(
            f"Studies / {work.get('name', 'Study')} / HPO · {parameter.replace('_', ' ').title()}"
        )

    def compose_workspace(self) -> ComposeResult:
        yield Static(
            f"{self.parameter.replace('_', ' ').title()} · predictive evidence for "
            f"{objective_display_name(self.study.get('objective', {}))}",
            classes="workspace-header",
        )
        with VerticalScroll():
            with Horizontal(classes="workspace-actions compact-actions"):
                yield Button("? Explain", id="hpo-parameter-help", flat=True)
                yield Button("↗ Interactive HTML", id="hpo-parameter-export", flat=True)
            with Grid(classes="hpo-parameter-summary-grid"):
                yield Static(id="hpo-detail-authored", classes="metric-card")
                yield Static(id="hpo-detail-observed", classes="metric-card")
                yield Static(id="hpo-detail-promising", classes="metric-card")
                yield Static(id="hpo-detail-reliability", classes="metric-card")
            yield HpoParameterDashboard(id="hpo-parameter-dashboard")
            yield Label("Observed dispersion", classes="section-title")
            yield DataTable(id="hpo-dispersion-table", cursor_type="row", zebra_stripes=True)
            yield Label("Interactions with other parameters", classes="section-title")
            yield DataTable(id="hpo-relation-table", cursor_type="row", zebra_stripes=True)
            yield Static(id="hpo-detail-note", classes="detail-panel")

    def on_mount(self) -> None:
        space = self.analysis.get("search_space", {})
        coverage = self.analysis.get("coverage", {})
        marginal = coverage.get("marginal", {}) if isinstance(coverage, Mapping) else {}
        responses = self.analysis.get("response_curves", {})
        importance = self.analysis.get("parameter_importance", {})
        rule = space.get(self.parameter, {}) if isinstance(space, Mapping) else {}
        observed = marginal.get(self.parameter, {}) if isinstance(marginal, Mapping) else {}
        response = responses.get(self.parameter, {}) if isinstance(responses, Mapping) else {}
        detail = importance.get(self.parameter, {}) if isinstance(importance, Mapping) else {}
        live_detail = self._live_detail()
        if not response:
            response = self._live_response(live_detail)
        confidence = live_detail.get("confidence")
        reliability = live_detail.get(
            "confidence_label",
            detail.get("reliability", "low") if isinstance(detail, Mapping) else "low",
        )
        self.query_one("#hpo-detail-authored", Static).update(
            "[b]SEARCH SPACE[/b]\n"
            f"{StudyWorkspace._parameter_domain(rule)}\n"
            f"Source: {self.analysis.get('search_space_source', 'authored configuration')}"
        )
        self.query_one("#hpo-detail-observed", Static).update(
            "[b]OBSERVED COVERAGE[/b]\n" + StudyWorkspace._observed_domain(observed)
        )
        self.query_one("#hpo-detail-promising", Static).update(
            "[b]PROMISING REGION[/b]\n"
            + StudyWorkspace._promising_region(
                response,
                self.study.get("objective", {}),
                live_detail=live_detail,
            )
        )
        self.query_one("#hpo-detail-reliability", Static).update(
            "[b]RELIABILITY[/b]\n"
            + (
                f"{float(confidence):.0%} · {str(reliability).title()} · "
                if isinstance(confidence, int | float) and not isinstance(confidence, bool)
                else f"{str(reliability).title()} · "
            )
            + f"importance {format_value(detail.get('importance'))}"
        )
        self.query_one("#hpo-parameter-dashboard", HpoParameterDashboard).show_parameter(
            self.parameter, response if isinstance(response, Mapping) else {}
        )
        self._populate_dispersion()
        self._populate_interactions()
        self.query_one("#hpo-detail-note", Static).update(
            (
                "The cyan response is the fitted predictive association after accounting for the "
                "sampled context. Exact uncertainty remains in the evidence table and becomes a "
                "proper shaded band in the interactive HTML report."
                if response.get("source") != "live-observed"
                else "The cyan response summarizes current observed candidate means while final "
                "multivariate analysis is still being formed."
            )
            + " Support counts show where the Study has evidence. These views do not establish "
            "a causal effect."
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "hpo-parameter-help":
            self._show_explanation(
                f"How to read {self.parameter}",
                "Search space is what the YAML allowed. Observed coverage is what actually ran. "
                "Promising region is a predictive association conditional on the sampled context, "
                "not a causal rule. Reliability combines available support and validation evidence; "
                "low reliability means the correct next action is usually targeted sampling, not a "
                "strong conclusion. Click terminal points or bars for exact values. The HTML report "
                "adds hover labels, zoom, a real uncertainty band, heatmaps and numeric 3D surfaces.",
            )
        elif event.button.id == "hpo-parameter-export":
            path = self._report_path(
                str(self.work.get("name", "study")), f"{self.parameter}-interactive"
            )
            analysis = dict(self.analysis)
            parameter = self.parameter
            self._write_and_open_report(
                f"Interactive HPO report · {parameter}",
                lambda: write_parameter_html(analysis, parameter, path),
            )

    def _live_detail(self) -> Mapping[str, Any]:
        live = self.analysis.get("live_hpo", {})
        parameters = live.get("parameters", ()) if isinstance(live, Mapping) else ()
        return next(
            (
                value
                for value in parameters
                if isinstance(value, Mapping) and value.get("parameter") == self.parameter
            ),
            {},
        )

    @staticmethod
    def _live_response(detail: Mapping[str, Any]) -> dict[str, Any]:
        response = detail.get("response", {})
        raw_points = response.get("points", ()) if isinstance(response, Mapping) else ()
        points = []
        for point in raw_points if isinstance(raw_points, Sequence) else ():
            if not isinstance(point, Mapping):
                continue
            coordinate = point.get("parameter")
            converted = {
                "predicted_objective": point.get("objective"),
                "support_count": point.get("samples", 0),
            }
            converted["category" if coordinate is None else "x"] = (
                point.get("label") if coordinate is None else coordinate
            )
            points.append(converted)
        return {"points": points, "source": "live-observed"}

    def _populate_dispersion(self) -> None:
        grouped: dict[str, list[float]] = {}
        for candidate in self.analysis.get("candidates", ()):
            if not isinstance(candidate, Mapping):
                continue
            parameters = candidate.get("parameters", {})
            value = parameters.get(self.parameter) if isinstance(parameters, Mapping) else None
            objective = candidate.get("mean")
            if value is not None and isinstance(objective, int | float):
                grouped.setdefault(str(value), []).append(float(objective))
        table = self.query_one("#hpo-dispersion-table", DataTable)
        table.add_columns("Value", "Candidates", "Mean objective", "Standard deviation")
        if not grouped:
            response = self._live_detail().get("response", {})
            points = response.get("points", ()) if isinstance(response, Mapping) else ()
            for point in points if isinstance(points, Sequence) else ():
                if not isinstance(point, Mapping):
                    continue
                table.add_row(
                    format_value(point.get("parameter", point.get("label"))),
                    str(point.get("samples", 0)),
                    format_value(point.get("objective")),
                    "Still learning",
                )
            return
        for value, observations in sorted(grouped.items()):
            table.add_row(
                value,
                str(len(observations)),
                format_value(statistics.fmean(observations)),
                format_value(statistics.stdev(observations) if len(observations) >= 2 else None),
            )

    def _populate_interactions(self) -> None:
        raw = self.analysis.get("interactions", {})
        pairs = raw.get("pairs", ()) if isinstance(raw, Mapping) else ()
        table = self.query_one("#hpo-relation-table", DataTable)
        table.add_columns("Other parameter", "Predictive gain", "Reliability", "Support")
        rendered = 0
        for item in pairs if isinstance(pairs, Sequence) else ():
            if not isinstance(item, Mapping):
                continue
            pair = item.get("parameters", item.get("pair", ()))
            if not pair and item.get("left") is not None:
                pair = (item.get("left"), item.get("right"))
            names = [str(value) for value in pair]
            if self.parameter not in names:
                continue
            other = next((value for value in names if value != self.parameter), self.parameter)
            table.add_row(
                other.replace("_", " ").title(),
                format_value(item.get("predictive_gain", item.get("gain", item.get("importance")))),
                str(item.get("reliability", "low")).title(),
                str(item.get("support", "—")),
            )
            rendered += 1
        if rendered:
            return
        for item in self._live_detail().get("joint_relationships", ()):
            if not isinstance(item, Mapping):
                continue
            confidence = item.get("confidence")
            table.add_row(
                str(item.get("parameter", "—")).replace("_", " ").title(),
                format_value(item.get("gain")),
                (
                    f"{float(confidence):.0%} · {str(item.get('confidence_label', 'low')).title()}"
                    if isinstance(confidence, int | float) and not isinstance(confidence, bool)
                    else str(item.get("confidence_label", "low")).title()
                ),
                str(item.get("observations", "—")),
            )


class HpoActionWorkspace(ResearchWorkspace):
    """One persisted controller decision and its exact supporting values."""

    def __init__(self, work: Mapping[str, Any], number: int, action: Mapping[str, Any]) -> None:
        self.action = dict(action)
        super().__init__(f"Studies / {work.get('name', 'Study')} / HPO action {number}")

    def compose_workspace(self) -> ComposeResult:
        action = self.action.get("action", self.action.get("decision", "Decision"))
        trial = self.action.get("trial", self.action.get("candidate", "—"))
        yield Static(f"{str(action).replace('_', ' ').title()} · Trial {trial}", classes="workspace-header")
        yield VerticalScroll(
            Static(
                _structured_text(self.action, heading="PERSISTED CONTROLLER EVIDENCE", limit=240),
                classes="workspace-panel",
            )
        )


class TrialWorkspace(ResearchWorkspace):
    """One candidate's seeds, parameters, metrics, resources and decision evidence."""

    BINDINGS = [*ResearchWorkspace.BINDINGS, ("right", "open_seed", "Open seed")]

    def __init__(
        self,
        work: Mapping[str, Any],
        candidate: Mapping[str, Any],
        services: Any,
        *,
        objective: Mapping[str, Any],
    ) -> None:
        self.work = dict(work)
        self.candidate = dict(candidate)
        self.services = services
        self.objective = dict(objective)
        self.selected_run_key: str | None = None
        trial = candidate.get("trial", "?")
        super().__init__(f"Studies / {work.get('name', 'Study')} / Trial {trial}")

    def compose_workspace(self) -> ComposeResult:
        yield Static(id="trial-header", classes="workspace-header")
        with TabbedContent(initial="trial-seeds"):
            with TabPane("Seeds", id="trial-seeds"):
                yield DataTable(id="seed-table", cursor_type="row", zebra_stripes=True)
                yield Static(
                    "† partial/censored evidence; never a final seed score", classes="legend"
                )
            with TabPane("Parameters", id="trial-parameters"):
                yield DataTable(id="parameter-table", cursor_type="row", zebra_stripes=True)
            with TabPane("Metrics", id="trial-metrics"):
                yield Static(id="trial-metric-content", classes="workspace-panel")
            with TabPane("Resources", id="trial-resources"):
                yield Static(id="trial-resource-content", classes="workspace-panel")
            with TabPane("Decision", id="trial-decision"):
                yield VerticalScroll(Static(id="trial-decision-content"))

    def on_mount(self) -> None:
        self._render_workspace()

    def _render_workspace(self) -> None:
        candidate = self.candidate
        objective_name = objective_display_name(self.objective)
        self.query_one("#trial-header", Static).update(
            f"Trial {candidate.get('trial')}  ·  {str(candidate.get('state', 'unknown')).upper()}\n"
            f"{objective_name} selection {format_value(candidate.get('selection_objective'))}  "
            f"current {format_value(candidate.get('current_objective'))}  "
            f"best {format_value(candidate.get('best_objective'))}  "
            f"SE {format_value(candidate.get('selection_standard_error'))}  "
            f"Pareto {'yes' if candidate.get('pareto_optimal') else 'no'}"
        )
        runs = [value for value in candidate.get("runs", ()) if isinstance(value, Mapping)]
        table = self.query_one("#seed-table", DataTable)
        table.clear(columns=True)
        table.add_columns(
            "Seed", "Current", "Best", "Final", "Best epoch", "Latest", "State", "GPU", "Time"
        )
        for run in runs:
            state = str(run.get("state", "unknown"))
            censored = "†" if state == "pruned" or run.get("objective_censoring") else ""
            table.add_row(
                str(run.get("seed", "none")),
                f"{self._run_value(run.get('current_observed_objective'), state, 'current')}{censored}",
                f"{self._run_value(run.get('best_observed_objective', run.get('best_objective')), state, 'best')}{censored}",
                self._run_value(run.get("final_objective"), state, "final"),
                str(run.get("best_step", "-")),
                str(run.get("latest_step", "-")),
                str(run.get("state", "unknown")).upper(),
                str(run.get("gpu_index", "-")),
                format_duration(run.get("duration_seconds")),
                key=str(run.get("key", run.get("seed", "none"))),
            )
        parameters = self.query_one("#parameter-table", DataTable)
        parameters.clear(columns=True)
        parameters.add_columns("Parameter", "Value")
        for name, value in sorted(candidate.get("parameters", {}).items()):
            parameters.add_row(str(name), format_value(value))
        latest = candidate.get("latest_metrics", {})
        metric_lines = [
            f"{metric_display_name(name):32} {format_value(value)}"
            for name, value in sorted(latest.items())
        ]
        self.query_one("#trial-metric-content", Static).update(
            "LATEST AGGREGATED METRICS\n" + ("\n".join(metric_lines) or "No metrics yet.")
        )
        cost = candidate.get("cost", candidate.get("resource_cost", {}))
        self.query_one("#trial-resource-content", Static).update(
            _structured_text(cost, heading="RESOURCE EVIDENCE")
        )
        self.query_one("#trial-decision-content", Static).update(
            "CANDIDATE DECISION\n"
            f"State                 {candidate.get('state', 'unknown')}\n"
            f"Feasibility           {candidate.get('feasibility', 'unavailable')}\n"
            f"Confirmation          {candidate.get('confirmation_status', 'unavailable')}\n"
            f"Partial/censored      {bool(candidate.get('partially_censored'))}\n\n"
            "Parameters and evidence shown here are persisted controller facts, not inferred UI state."
        )

    @staticmethod
    def _run_value(value: Any, state: str, kind: str) -> str:
        if value is not None:
            return format_value(value)
        if state == "pruned":
            return "not final · pruned" if kind == "final" else "not observed"
        if state in {"scheduled", "queued", "preparing", "running"}:
            return "not reported yet"
        if state in {"failed", "timeout", "cancelled"}:
            return "not produced"
        return "not recorded"

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id != "seed-table":
            return
        runs = [value for value in self.candidate.get("runs", ()) if isinstance(value, Mapping)]
        if 0 <= event.cursor_row < len(runs):
            self.selected_run_key = str(runs[event.cursor_row].get("key", ""))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "seed-table":
            self._open_seed(event.cursor_row)

    def action_open_seed(self) -> None:
        self._open_seed(self.query_one("#seed-table", DataTable).cursor_row)

    def _open_seed(self, row: int) -> None:
        runs = [value for value in self.candidate.get("runs", ()) if isinstance(value, Mapping)]
        if 0 <= row < len(runs):
            self.selected_run_key = str(runs[row].get("key", ""))
            self.app.push_screen(
                SeedWorkspace(
                    self.work,
                    self.candidate,
                    runs[row],
                    self.services,
                    objective=self.objective,
                )
            )


class SeedWorkspace(ResearchWorkspace):
    """Live bounded curves, epochs, resources and isolated logs for one Study Run."""

    BINDINGS = [
        *ResearchWorkspace.BINDINGS,
        ("up", "previous_epoch", "Previous epoch"),
        ("down", "next_epoch", "Next epoch"),
        ("right", "open_epoch", "Epoch detail"),
        ("n", "next_chart_page", "Next curves"),
        ("p", "previous_chart_page", "Previous curves"),
    ]

    def __init__(
        self,
        work: Mapping[str, Any],
        candidate: Mapping[str, Any],
        run: Mapping[str, Any],
        services: Any,
        *,
        objective: Mapping[str, Any],
    ) -> None:
        self.work = dict(work)
        self.candidate = dict(candidate)
        self.run = dict(run)
        self.services = services
        self.objective = dict(objective)
        self.detail: dict[str, Any] | None = None
        self.selected_epoch: int | None = (
            int(run["latest_step"]) if isinstance(run.get("latest_step"), int) else None
        )
        self._loading = False
        self.chart_page = 0
        trial, seed = candidate.get("trial", "?"), run.get("seed", "none")
        super().__init__(f"Studies / {work.get('name', 'Study')} / Trial {trial} / Seed {seed}")

    @property
    def active(self) -> bool:
        state = str((self.detail or self.run).get("state", "unknown"))
        return state in {"running", "scheduled", "retrying", "queued", "preparing"}

    def compose_workspace(self) -> ComposeResult:
        yield Static(id="seed-header", classes="workspace-header")
        with Vertical(id="seed-loading", classes="workspace-loading"):
            yield LoadingIndicator()
            yield Label(
                "Loading persisted Run telemetry and learning curves from the cluster…",
                id="seed-loading-message",
            )
        with TabbedContent(initial="seed-curves", id="seed-tabs"):
            with TabPane("Curves", id="seed-curves"):
                with VerticalScroll(id="curve-scroll"):
                    with Horizontal(classes="curve-controls"):
                        yield Button(
                            "‹ Previous",
                            id="curve-previous",
                            classes="curve-page-button",
                            flat=True,
                        )
                        yield Select(
                            (("Page 1 of 1", 0),),
                            value=0,
                            allow_blank=False,
                            compact=True,
                            id="curve-page-select",
                        )
                        yield Button(
                            "Next ›",
                            id="curve-next",
                            classes="curve-page-button",
                            flat=True,
                        )
                        yield Button(
                            "↗ Interactive HTML",
                            id="curve-export",
                            classes="curve-page-button",
                            flat=True,
                        )
                        yield Button(
                            "? Explain",
                            id="curve-help",
                            classes="curve-page-button",
                            flat=True,
                        )
                    yield Static(id="curve-content", classes="curve-page-summary")
                    yield MetricDashboard(id="seed-metric-dashboard")
            with TabPane("Epochs", id="seed-epochs"):
                yield DataTable(id="epoch-table", cursor_type="row", zebra_stripes=True)
            with TabPane("Metrics", id="seed-metrics"):
                yield DataTable(id="metric-table", cursor_type="row", zebra_stripes=True)
            with TabPane("Resources", id="seed-resources"):
                yield Static(id="seed-resource-content", classes="workspace-panel")
            with TabPane("Logs", id="seed-logs"):
                yield RichLog(id="seed-log-content", wrap=False, auto_scroll=True, highlight=False)
            with TabPane("Metadata", id="seed-metadata"):
                yield VerticalScroll(Static(id="seed-metadata-content"))

    def on_mount(self) -> None:
        self._render_header(self.run)
        self.query_one("#seed-loading").display = True
        self.query_one("#seed-tabs").display = False
        self._load()
        self.set_interval(2.0, self._refresh_if_active)

    def _load(self) -> None:
        if self._loading:
            return
        self._loading = True
        job_id = str(self.work.get("primary_job_id", ""))
        run_key = str(self.run.get("key", ""))

        def load() -> None:
            try:
                detail = self.services.study_run(job_id, run_key, tail=2_000, curve_points=300)
            except Exception as error:
                self.app.call_from_thread(
                    self._show_load_error,
                    f"{type(error).__name__}: {error}",
                )
            else:
                self.app.call_from_thread(self._apply_detail, detail)
            finally:
                self._loading = False

        Thread(target=load, daemon=True, name="lambdaforge-tui-seed-detail").start()

    def _show_load_error(self, message: str) -> None:
        self.query_one("#seed-loading LoadingIndicator").display = False
        self.query_one("#seed-loading-message", Label).update(
            "Run telemetry could not be loaded. This is not an empty Run.\n" + message
        )

    def _refresh_if_active(self) -> None:
        if self.active:
            self._load()

    def _apply_detail(self, detail: Mapping[str, Any]) -> None:
        self.detail = dict(detail)
        self.query_one("#seed-loading").display = False
        self.query_one("#seed-tabs").display = True
        self._render_header(detail)
        curves = detail.get("curves", {})
        curves = curves if isinstance(curves, Mapping) else {}
        rows = epoch_rows(curves)
        available_steps = [step for step, _values in rows]
        if self.selected_epoch not in available_steps:
            self.selected_epoch = available_steps[-1] if available_steps else None
        self._render_curves(curves, detail)
        self._populate_epochs(rows)
        latest = detail.get("latest_metrics", {})
        chart_filter = detail.get("chart_filter", {})
        aliases = chart_filter.get("display_names", {}) if isinstance(chart_filter, Mapping) else {}
        aliases = aliases if isinstance(aliases, Mapping) else {}
        metric_table = self.query_one("#metric-table", DataTable)
        metric_table.clear(columns=True)
        metric_table.add_columns("Metric", "Current", "Direction")
        for name, value in sorted(latest.items()):
            metric_table.add_row(
                metric_display_name(name, aliases), format_value(value), "not declared"
            )
        resources = detail.get("resources", {})
        usage = resources.get("usage", resources) if isinstance(resources, Mapping) else {}
        self.query_one("#seed-resource-content", Static).update(
            "OBSERVED RESOURCE USE\n"
            f"GPU                    {detail.get('gpu_index', 'unavailable')}\n"
            f"GPU allocated/live     {format_value(usage.get('gpu_mem_mb'))} MiB\n"
            f"GPU allocator reserved {format_value(usage.get('gpu_reserved_mb'))} MiB\n"
            f"Peak VRAM              {format_value(usage.get('gpu_peak_reserved_mb', usage.get('peak_vram')))} MiB\n"
            f"RAM                    {format_value(usage.get('ram_mb', usage.get('peak_ram')))} MiB\n"
            f"CPU                    {format_value(usage.get('cpu_percent'))} %\n"
            f"Elapsed                {format_duration(detail.get('duration_seconds'))}\n\n"
            "Reserved GPU memory is allocator cache, not live scientific tensor allocation."
        )
        log = self.query_one("#seed-log-content", RichLog)
        at_end = log.is_vertical_scroll_end
        previous_y = log.scroll_y
        log.clear()
        log.write(str(detail.get("log", "No scientific output yet.")))
        failure = detail.get("failure")
        if isinstance(failure, Mapping):
            failure_text = (
                "\nSCIENTIFIC FAILURE\n"
                f"{failure.get('type', 'Exception')}: {failure.get('message', '')}\n"
                f"Phase: {failure.get('phase', 'Work.run')}\n"
                f"Result: {failure.get('result_path', 'unavailable')}"
            )
            traceback = str(failure.get("traceback", ""))
            if traceback and traceback not in str(detail.get("log", "")):
                failure_text += "\n\nTRACEBACK\n" + traceback
            log.write(failure_text)
        if at_end:
            log.scroll_end(animate=False)
        else:
            log.scroll_to(y=previous_y, animate=False)
        self.query_one("#seed-metadata-content", Static).update(
            json.dumps(
                {
                    "trial": detail.get("trial"),
                    "seed": detail.get("seed"),
                    "state": detail.get("state"),
                    "objective_status": detail.get("objective_status"),
                    "failure": failure,
                    "paths": detail.get("paths"),
                },
                indent=2,
                default=str,
            )
        )

    def _render_curves(self, curves: Mapping[str, Any], detail: Mapping[str, Any]) -> None:
        chart_filter = detail.get("chart_filter", {})
        chart_filter = chart_filter if isinstance(chart_filter, Mapping) else {}
        names = visible_metrics(curves, chart_filter)
        aliases = chart_filter.get("display_names", {})
        aliases = aliases if isinstance(aliases, Mapping) else {}
        page_size = 4
        page_count = max(1, (len(names) + page_size - 1) // page_size)
        self.chart_page = min(self.chart_page, page_count - 1)
        shown = names[self.chart_page * page_size : (self.chart_page + 1) * page_size]
        page_select = self.query_one("#curve-page-select", Select)
        page_select.set_options(
            tuple((f"Page {index + 1} of {page_count}", index) for index in range(page_count))
        )
        page_select.value = self.chart_page
        self.query_one("#curve-previous", Button).disabled = page_count <= 1
        self.query_one("#curve-next", Button).disabled = page_count <= 1
        best_step = detail.get("best_step") if isinstance(detail.get("best_step"), int) else None
        self.query_one("#curve-content", Static).update(
            f"CURVES {self.chart_page + 1}/{page_count} · "
            f"showing {', '.join(metric_display_name(name, aliases) for name in shown)} · "
            "use the page selector or n/p · "
            "green = best objective epoch · red = selected epoch"
            if shown
            else "No scalar learning curves are available yet."
        )
        self.query_one("#seed-metric-dashboard", MetricDashboard).show_curves(
            curves,
            shown,
            selected_step=self.selected_epoch,
            best_step=best_step,
            display_names=aliases,
        )

    def action_next_chart_page(self) -> None:
        self._change_chart_page(1)

    def action_previous_chart_page(self) -> None:
        self._change_chart_page(-1)

    def _change_chart_page(self, increment: int) -> None:
        if self.detail is None:
            return
        curves = self.detail.get("curves", {})
        if not isinstance(curves, Mapping):
            return
        chart_filter = self.detail.get("chart_filter", {})
        names = visible_metrics(curves, chart_filter if isinstance(chart_filter, Mapping) else {})
        page_count = max(1, (len(names) + 3) // 4)
        self.chart_page = (self.chart_page + increment) % page_count
        self._render_curves(curves, self.detail)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "curve-previous":
            self.action_previous_chart_page()
        elif event.button.id == "curve-next":
            self.action_next_chart_page()
        elif event.button.id == "curve-help":
            self._show_explanation(
                "Learning curves",
                "Each curve is a scalar emitted by the Work. Green marks the epoch selected by "
                "the objective; red marks the epoch selected in the table. Changing page resets "
                "both axes to the new data automatically. Click a point for its exact epoch and "
                "value, or open the HTML view for hover labels and zoom.",
            )
        elif event.button.id == "curve-export":
            self._export_curves()

    def _export_curves(self) -> None:
        if self.detail is None:
            self.notify("Run telemetry is still loading.", severity="warning")
            return
        curves = self.detail.get("curves", {})
        if not isinstance(curves, Mapping):
            self.notify("No scalar curves are available for export.", severity="warning")
            return
        chart_filter = self.detail.get("chart_filter", {})
        chart_filter = chart_filter if isinstance(chart_filter, Mapping) else {}
        aliases = chart_filter.get("display_names", {})
        aliases = aliases if isinstance(aliases, Mapping) else {}
        names = visible_metrics(curves, chart_filter)
        trial = self.candidate.get("trial", "trial")
        seed = self.run.get("seed", "seed")
        path = self._report_path(
            str(self.work.get("name", "study")), f"trial-{trial}-seed-{seed}-curves"
        )
        self._write_and_open_report(
            "Interactive learning curves",
            lambda: write_metric_html(curves, names, path, display_names=aliases),
        )

    def on_select_changed(self, event: Select.Changed) -> None:
        if (
            event.select.id == "curve-page-select"
            and isinstance(event.value, int)
            and event.value != self.chart_page
            and self.detail is not None
        ):
            self.chart_page = event.value
            curves = self.detail.get("curves", {})
            if isinstance(curves, Mapping):
                self._render_curves(curves, self.detail)

    def _render_header(self, detail: Mapping[str, Any]) -> None:
        state = str(detail.get("state", self.run.get("state", "unknown")))
        censored = "  † CENSORED" if state == "pruned" else ""
        objective_name = objective_display_name(self.objective)
        self.query_one("#seed-header", Static).update(
            f"Seed {detail.get('seed', self.run.get('seed', 'none'))}  ·  "
            f"{state.upper()}{censored}\n"
            f"{objective_name}: current {TrialWorkspace._run_value(detail.get('current_observed_objective'), state, 'current')}   "
            f"best {TrialWorkspace._run_value(detail.get('best_observed_objective', detail.get('best_objective')), state, 'best')}   "
            f"final {TrialWorkspace._run_value(detail.get('final_objective'), state, 'final')}   "
            f"best epoch {detail.get('best_step', '-')}   latest {detail.get('current_step', detail.get('latest_step', '-'))}   "
            f"GPU {detail.get('gpu_index', '-')}   elapsed {format_duration(detail.get('duration_seconds'))}"
        )

    def _populate_epochs(self, rows: Sequence[tuple[int, Mapping[str, float]]]) -> None:
        table = self.query_one("#epoch-table", DataTable)
        table.clear(columns=True)
        metric = str(self.objective.get("metric", ""))
        names = sorted({name for _step, values in rows for name in values})
        if metric in names:
            names.remove(metric)
            names.insert(0, metric)
        chart_filter = (self.detail or {}).get("chart_filter", {})
        aliases = chart_filter.get("display_names", {}) if isinstance(chart_filter, Mapping) else {}
        aliases = aliases if isinstance(aliases, Mapping) else {}
        table.add_columns(
            "Epoch", *(metric_display_name(name, aliases) for name in names), "Marker"
        )
        best_step = (self.detail or {}).get("best_step")
        for step, values in rows:
            markers = []
            if step == best_step:
                markers.append("★ best")
            if step == self.selected_epoch:
                markers.append("◆ selected")
            table.add_row(
                str(step),
                *(format_value(values.get(name)) for name in names),
                " · ".join(markers),
                key=str(step),
            )
        if self.selected_epoch is not None:
            for row, (step, _values) in enumerate(rows):
                if step == self.selected_epoch:
                    table.move_cursor(row=row)
                    break

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id != "epoch-table" or self.detail is None:
            return
        rows = epoch_rows(self.detail.get("curves", {}))
        if 0 <= event.cursor_row < len(rows):
            self.selected_epoch = rows[event.cursor_row][0]
            curves = self.detail.get("curves", {})
            if isinstance(curves, Mapping):
                self._render_curves(curves, self.detail)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "epoch-table":
            self.action_open_epoch()

    def action_previous_epoch(self) -> None:
        table = self.query_one("#epoch-table", DataTable)
        table.action_cursor_up()

    def action_next_epoch(self) -> None:
        table = self.query_one("#epoch-table", DataTable)
        table.action_cursor_down()

    def action_open_epoch(self) -> None:
        if self.detail is None or self.selected_epoch is None:
            return
        values = dict(
            next(
                (
                    metrics
                    for step, metrics in epoch_rows(self.detail.get("curves", {}))
                    if step == self.selected_epoch
                ),
                {},
            )
        )
        self.app.push_screen(
            EpochWorkspace(
                self.breadcrumb,
                self.selected_epoch,
                values,
                best=self.selected_epoch == self.detail.get("best_step"),
                objective_status=self.detail.get("objective_status", {}),
                display_names=(
                    self.detail.get("chart_filter", {}).get("display_names", {})
                    if isinstance(self.detail.get("chart_filter"), Mapping)
                    else {}
                ),
            )
        )


class EpochWorkspace(ResearchWorkspace):
    """All same-step scalar evidence for one selected epoch."""

    def __init__(
        self,
        parent: str,
        epoch: int,
        values: Mapping[str, float],
        *,
        best: bool,
        objective_status: Mapping[str, Any],
        display_names: Mapping[str, Any] | None = None,
    ) -> None:
        self.epoch = epoch
        self.values = dict(values)
        self.best = best
        self.objective_status = dict(objective_status)
        self.display_names = dict(display_names or {})
        super().__init__(f"{parent} / Epoch {epoch}")

    def compose_workspace(self) -> ComposeResult:
        marker = " · BEST OBJECTIVE" if self.best else ""
        yield Label(f"Epoch {self.epoch}{marker}", classes="workspace-header")
        table: DataTable[Any] = DataTable(
            id="epoch-detail-table", cursor_type="row", zebra_stripes=True
        )
        table.add_columns("Metric", "Value")
        for name, value in sorted(self.values.items()):
            table.add_row(metric_display_name(name, self.display_names), format_value(value))
        yield table
        yield Static(
            "Objective status: "
            + str(self.objective_status.get("status", "unknown"))
            + (
                "\nMissing same-step components: "
                + ", ".join(map(str, self.objective_status.get("missing_components", ())))
                if self.objective_status.get("missing_components")
                else ""
            ),
            classes="detail-panel",
        )


class CredentialDialog(ModalScreen[str | None]):
    """Collect a password without ever echoing or persisting it in widget state."""

    def __init__(self, cluster: str) -> None:
        super().__init__()
        self.cluster = cluster

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-card"):
            yield Label(f"Store password for {self.cluster}", classes="modal-title")
            yield Static(
                "The secret is stored in the OS keyring; cluster YAML stores only a reference."
            )
            yield Input(password=True, id="credential-secret", placeholder="Password")
            with Horizontal(classes="modal-actions"):
                yield Button("Cancel", id="credential-cancel")
                yield Button("Store securely", id="credential-store", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "credential-cancel":
            self.dismiss(None)
        else:
            secret = self.query_one("#credential-secret", Input).value
            self.dismiss(secret or None)


class ExactConfirmation(ModalScreen[bool]):
    """Confirm an exact mutation through a readable, already-previewed scope."""

    def __init__(self, title: str, preview: Mapping[str, Any]) -> None:
        super().__init__()
        self.title_text = title
        self.preview = dict(preview)

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-card wide-modal"):
            yield Label(self.title_text, classes="modal-title")
            with VerticalScroll(id="confirmation-scroll"):
                yield Static(
                    confirmation_text(self.preview),
                    id="exact-preview",
                    markup=False,
                )
            with Horizontal(classes="modal-actions"):
                yield Button("Cancel", id="exact-cancel")
                yield Button("Confirm and apply", id="exact-apply", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "exact-apply")


class OperationResizeHandle(Static):
    """Mouse resize handle for a workspace operation console."""

    def __init__(self, target: str) -> None:
        super().__init__("↕ drag to resize output", id="cluster-operation-resize")
        self.target = target
        self._dragging = False
        self._start_y = 0.0
        self._start_height = 0

    def on_mouse_down(self, event: MouseDown) -> None:
        if event.button != 1:
            return
        target = self.screen.query_one(self.target)
        self._dragging = True
        self._start_y = float(event.screen_y)
        self._start_height = target.size.height
        self.capture_mouse()
        event.stop()

    def on_mouse_move(self, event: MouseMove) -> None:
        if not self._dragging:
            return
        target = self.screen.query_one(self.target)
        # Preserve a useful upper viewport plus Header/navigation/Footer even when
        # the console is dragged aggressively on a short terminal.
        maximum = max(6, self.screen.size.height - 22)
        height = self._start_height + int(self._start_y - float(event.screen_y))
        target.styles.height = max(6, min(maximum, height))
        event.stop()

    def on_mouse_up(self, event: MouseUp) -> None:
        if self._dragging and event.button == 1:
            self._dragging = False
            self.release_mouse()
            event.stop()


class ClusterWorkspace(ResearchWorkspace):
    """Operational cluster workspace backed by real Python services."""

    def __init__(self, name: str, services: Any) -> None:
        self.cluster_name = name
        self.services = services
        self.detail: dict[str, Any] | None = None
        self._refreshing_resources = False
        self._operation_running = False
        self._operation_label = "Idle"
        self._operation_started = 0.0
        self._provider_lock = Lock()
        super().__init__(f"Clusters / {name}")

    def compose_workspace(self) -> ComposeResult:
        # The upper workspace owns its own scroll viewport.  Growing the operation
        # console therefore reduces this viewport instead of clipping its tabs and
        # actions beyond reach.
        with VerticalScroll(id="cluster-workspace-main"):
            yield Static(
                f"{self.cluster_name} · loading",
                id="cluster-workspace-header",
                classes="workspace-header",
            )
            yield Static("CLUSTER ACTIONS", classes="section-title workspace-action-title")
            with Horizontal(classes="workspace-actions"):
                yield Button("Edit", id="cluster-edit", classes="cluster-action", flat=True)
                yield Button(
                    "Doctor",
                    id="cluster-doctor",
                    classes="cluster-action",
                    variant="primary",
                    flat=True,
                )
                yield Button(
                    "Plan bootstrap",
                    id="cluster-bootstrap-plan",
                    classes="cluster-action",
                    flat=True,
                )
                yield Button(
                    "Bootstrap…",
                    id="cluster-bootstrap-apply",
                    classes="cluster-action",
                    flat=True,
                )
            with TabbedContent(initial="cluster-overview", id="cluster-detail-tabs"):
                with TabPane("Overview", id="cluster-overview"):
                    with Grid(classes="cluster-overview-grid"):
                        yield Static(
                            "STATE\nLoading…", id="cluster-state-card", classes="metric-card"
                        )
                        yield Static(
                            "CONNECTION\nLoading…",
                            id="cluster-connection-card",
                            classes="metric-card",
                        )
                        yield Static(
                            "CAPACITY\nLoading…",
                            id="cluster-capacity-card",
                            classes="metric-card",
                        )
                        yield Static(
                            "STORAGE\nLoading…",
                            id="cluster-storage-card",
                            classes="metric-card",
                        )
                    yield ResourceDashboard(id="cluster-live-dashboard")
                with TabPane("Connection", id="cluster-connection"):
                    yield Static(id="cluster-connection-content", classes="workspace-panel")
                with TabPane("Resources", id="cluster-resources"):
                    yield VerticalScroll(Static(id="cluster-resource-content"))
                with TabPane("Environment", id="cluster-environment"):
                    yield Static(id="cluster-environment-content", classes="workspace-panel")
                with TabPane("GPU policy", id="cluster-gpu"):
                    yield Static(id="cluster-gpu-content", classes="workspace-panel")
                with TabPane("Storage", id="cluster-storage"):
                    yield VerticalScroll(Static(id="cluster-storage-content"))
                with TabPane("Credentials", id="cluster-credentials"):
                    yield Static(id="cluster-credential-content", classes="workspace-panel")
                    with Horizontal(classes="workspace-actions nested-actions"):
                        yield Button("Set credentials", id="cluster-credential-set", flat=True)
                        yield Button(
                            "Delete credentials",
                            id="cluster-credential-delete",
                            variant="warning",
                            flat=True,
                        )
                with TabPane("Jobs", id="cluster-jobs"):
                    yield Static(
                        "Use Work for semantic executions; low-level Jobs remain available through Ctrl+P."
                    )
                with TabPane("Danger zone", id="cluster-danger-zone"):
                    yield Static(
                        "Removing a profile never removes remote files, jobs, datasets or keyring secrets.",
                        classes="workspace-panel",
                    )
                    with Horizontal(classes="workspace-actions nested-actions"):
                        yield Button(
                            "Remove cluster", id="cluster-remove", variant="error", flat=True
                        )
        with Horizontal(id="cluster-operation-toolbar"):
            yield Label("OPERATION OUTPUT", classes="section-title")
            yield Static("Idle", id="cluster-operation-status")
            yield Select(
                (("Compact", 8), ("Comfortable", 14), ("Large", 24)),
                value=14,
                allow_blank=False,
                compact=True,
                id="cluster-operation-size",
            )
            yield Button("Clear", id="cluster-operation-clear", flat=True, compact=True)
        yield OperationResizeHandle("#cluster-operation-log")
        yield RichLog(id="cluster-operation-log", wrap=True, auto_scroll=True, highlight=True)

    def on_mount(self) -> None:
        self._set_operation_height(14)
        self._run_operation(
            "Inspect cluster",
            lambda: self.services.cluster_detail(self.cluster_name),
            self._apply_detail,
        )
        self.set_interval(2.0, self._refresh_resource_snapshot)
        self.set_interval(1.0, self._refresh_operation_status)

    def _apply_detail(self, detail: Mapping[str, Any]) -> None:
        self.detail = dict(detail)
        profile = detail.get("profile", {})
        resources = detail.get("resources", {})
        storage = detail.get("storage", {})
        self.query_one("#cluster-workspace-header", Static).update(
            f"{self.cluster_name} · "
            f"{'ONLINE' if resources.get('online') else 'UNREACHABLE'} · "
            f"{profile.get('scheduler', 'unknown')}"
        )
        observed = resources.get("observed", {}) if isinstance(resources, Mapping) else {}
        gpus = observed.get("gpus", ()) if isinstance(observed, Mapping) else ()
        self.query_one("#cluster-state-card", Static).update(
            "STATE\n"
            f"{'● ONLINE' if resources.get('online') else '○ OFFLINE'}\n"
            f"{profile.get('scheduler', 'unknown')} scheduler"
        )
        self.query_one("#cluster-connection-card", Static).update(
            "CONNECTION\n"
            f"{profile.get('user') or 'current user'}@{profile.get('host', 'local')}\n"
            f"{profile.get('transport', 'local')} · {detail.get('authentication_status', 'unknown')}"
        )
        self.query_one("#cluster-capacity-card", Static).update(
            "CAPACITY\n"
            f"{observed.get('cpu_total', '?')} CPU · {self._bytes(observed.get('ram_total_bytes'))} RAM\n"
            f"{len(gpus) if isinstance(gpus, Sequence) else 0} GPU"
        )
        self.query_one("#cluster-storage-card", Static).update(
            "OWNED STORAGE\n"
            f"{self._bytes(storage.get('total_bytes', storage.get('used_bytes')))} tracked\n"
            f"{storage.get('status', 'available')}"
        )
        self.query_one("#cluster-live-dashboard", ResourceDashboard).show_cluster(
            self.cluster_name,
            observed if isinstance(observed, Mapping) else {},
            resources.get("personal", {}) if isinstance(resources, Mapping) else {},
            sample_id=str(resources.get("observed_at_utc") or "") or None,
        )
        self.query_one("#cluster-connection-content", Static).update(
            f"Transport            {profile.get('transport')}\n"
            f"Host                 {profile.get('host')}\n"
            f"User                 {profile.get('user')}\n"
            f"Port                 {profile.get('port')}\n"
            "Connection policy\n" + _structured_text(profile.get("connection", {}))
        )
        self.query_one("#cluster-resource-content", Static).update(self._resource_text(resources))
        self.query_one("#cluster-environment-content", Static).update(
            _structured_text(
                profile.get("python", profile.get("environment", {})),
                heading="ENVIRONMENT POLICY",
            )
        )
        self.query_one("#cluster-gpu-content", Static).update(
            "GPU assignment is inherited from the scheduler/site grant. LambdaForge never "
            "overwrites CUDA_VISIBLE_DEVICES.\n\n" + _structured_text(profile.get("gpu_access", {}))
        )
        self.query_one("#cluster-storage-content", Static).update(
            _structured_text(storage, heading="OWNED STORAGE")
        )
        self.query_one("#cluster-credential-content", Static).update(
            f"Status: {detail.get('authentication_status')}\n"
            f"Reference: {profile.get('auth', {}).get('credential', 'interactive')}\n\n"
            "Passwords are never displayed or stored in YAML."
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        selected = event.button.id
        if selected == "cluster-operation-clear":
            self.query_one("#cluster-operation-log", RichLog).clear()
            return
        if self._operation_running and selected and selected.startswith("cluster-"):
            self.notify(
                f"{self._operation_label} is still running. Its transport is kept exclusive until completion.",
                severity="warning",
            )
            return
        if selected == "cluster-edit":
            editor = getattr(self.app, "edit_cluster", None)
            if callable(editor):
                editor(self.cluster_name, self._edited)
        elif selected == "cluster-doctor":
            self._run_operation("Doctor", lambda: self.services.doctor(self.cluster_name))
        elif selected == "cluster-bootstrap-plan":
            self._run_operation(
                "Bootstrap preview",
                lambda progress: self.services.bootstrap(
                    self.cluster_name, project=Path.cwd(), dry_run=True, progress=progress
                ),
                accepts_progress=True,
            )
        elif selected == "cluster-bootstrap-apply":
            self.app.push_screen(
                ExactConfirmation(
                    f"Bootstrap {self.cluster_name}",
                    {
                        "cluster": self.cluster_name,
                        "project": str(Path.cwd()),
                        "will_change": [
                            "managed user-space runtime/environment state on this cluster",
                            "the active managed-environment pointer after verification",
                            "unreferenced superseded managed environments",
                        ],
                        "will_preserve": [
                            "system Python, CUDA drivers and shell startup files",
                            "environments referenced by active Jobs",
                            "scientific datasets, Work results and project files",
                        ],
                        "recommended_first_step": "Use Plan bootstrap for a read-only environment plan.",
                    },
                ),
                self._apply_bootstrap,
            )
        elif selected == "cluster-credential-set":
            self.app.push_screen(CredentialDialog(self.cluster_name), self._store_credential)
        elif selected == "cluster-credential-delete":
            self.app.push_screen(
                ExactConfirmation(
                    f"Delete stored credential for {self.cluster_name}",
                    {
                        "will_remove": ["the referenced OS-keyring secret"],
                        "will_preserve": [
                            "cluster profile",
                            "remote files, jobs and environments",
                        ],
                        "effect": "future password authentication becomes interactive",
                    },
                ),
                self._apply_credential_delete,
            )
        elif selected == "cluster-remove":
            self._run_operation(
                "Remove preview",
                lambda: self.services.remove_cluster(self.cluster_name, apply=False),
                self._confirm_remove,
            )

    def _edited(self, saved: bool | None) -> None:
        if saved:
            self.notify("Cluster updated. Reopen it to use the refreshed service catalog.")

    def _apply_bootstrap(self, confirmed: bool | None) -> None:
        if confirmed:
            self._run_operation(
                "Bootstrap",
                lambda progress: self.services.bootstrap(
                    self.cluster_name, project=Path.cwd(), dry_run=False, progress=progress
                ),
                accepts_progress=True,
            )

    def _confirm_remove(self, preview: Mapping[str, Any]) -> None:
        self.app.push_screen(
            ExactConfirmation(f"Remove cluster {self.cluster_name}", preview),
            self._apply_remove,
        )

    def _apply_remove(self, confirmed: bool | None) -> None:
        if confirmed:
            self._run_operation(
                "Remove cluster",
                lambda: self.services.remove_cluster(self.cluster_name, apply=True),
            )

    def _store_credential(self, secret: str | None) -> None:
        if secret is None:
            return
        self._run_operation(
            "Store credential",
            lambda: self.services.set_cluster_credential(self.cluster_name, secret),
            self._reload,
        )

    def _apply_credential_delete(self, confirmed: bool | None) -> None:
        if confirmed:
            self._run_operation(
                "Delete credential",
                lambda: self.services.delete_cluster_credential(self.cluster_name),
                self._reload,
            )

    def _reload(self, _value: Mapping[str, Any]) -> None:
        self._run_operation(
            "Refresh",
            lambda: self.services.cluster_detail(self.cluster_name),
            self._apply_detail,
        )

    def _refresh_resource_snapshot(self) -> None:
        if self._refreshing_resources or self._operation_running:
            return
        self._refreshing_resources = True

        def load() -> None:
            if not self._provider_lock.acquire(blocking=False):
                self._refreshing_resources = False
                return
            try:
                value = self.services.cluster_resources(self.cluster_name)
            except Exception:
                pass
            else:
                self.app.call_from_thread(self._apply_resources, value)
            finally:
                self._provider_lock.release()
                self._refreshing_resources = False

        Thread(
            target=load,
            daemon=True,
            name="lambdaforge-tui-cluster-resources",
        ).start()

    def _apply_resources(self, resources: Mapping[str, Any]) -> None:
        observed = resources.get("observed", {})
        observed = observed if isinstance(observed, Mapping) else {}
        self.query_one("#cluster-live-dashboard", ResourceDashboard).show_cluster(
            self.cluster_name,
            observed,
            resources.get("personal", {}) if isinstance(resources.get("personal"), Mapping) else {},
            sample_id=str(resources.get("observed_at_utc") or "") or None,
        )
        self.query_one("#cluster-resource-content", Static).update(self._resource_text(resources))

    @classmethod
    def _resource_text(cls, resources: Mapping[str, Any]) -> str:
        observed = resources.get("observed", {})
        observed = observed if isinstance(observed, Mapping) else {}
        personal = resources.get("personal", {})
        personal = personal if isinstance(personal, Mapping) else {}
        lines = [
            "LIVE CAPACITY",
            f"CPU                  {observed.get('cpu_load', 'n/a')}% of {observed.get('cpu_total', '?')} cores",
            f"RAM                  {cls._bytes(observed.get('ram_available_bytes'))} available / {cls._bytes(observed.get('ram_total_bytes'))}",
            f"Observed at          {resources.get('observed_at_utc', 'unavailable')}",
            "",
            "YOUR LAMBDAFORGE ACTIVITY",
            f"Active jobs          {personal.get('active_jobs', 0)}",
            f"Scope                {personal.get('scope', 'unavailable')}",
            f"Note                 {personal.get('note', 'No personal observation recorded.')}",
            "",
            "GPU DEVICES",
        ]
        gpus = observed.get("gpus", ())
        for index, gpu in enumerate(gpus if isinstance(gpus, Sequence) else ()):
            if isinstance(gpu, Mapping):
                lines.append(
                    f"GPU {gpu.get('index', index)}  ·  {gpu.get('name', 'device')}  ·  "
                    f"utilization {gpu.get('utilization_percent', 'n/a')}%  ·  "
                    f"memory {cls._bytes(gpu.get('memory_used_bytes'))} / {cls._bytes(gpu.get('memory_total_bytes'))}"
                )
        if len(lines) == 11:
            lines.append("No GPU telemetry reported.")
        return "\n".join(lines)

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

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "cluster-operation-size" and isinstance(event.value, int):
            self._set_operation_height(event.value)

    def on_resize(self, event: Resize) -> None:
        """Keep both panes reachable after the terminal itself changes size."""
        del event
        try:
            current = self.query_one("#cluster-operation-log", RichLog).size.height
        except Exception:
            return
        self._set_operation_height(current or 14)

    def _set_operation_height(self, requested: int) -> None:
        maximum = max(6, self.size.height - 22)
        self.query_one("#cluster-operation-log", RichLog).styles.height = max(
            6, min(maximum, requested)
        )

    def _refresh_operation_status(self) -> None:
        if not self._operation_running:
            return
        elapsed = int(time.monotonic() - self._operation_started)
        self.query_one("#cluster-operation-status", Static).update(
            f"{self._operation_label} · working · {elapsed}s"
        )

    def _operation_progress(self, message: str) -> None:
        elapsed = int(time.monotonic() - self._operation_started)
        self.query_one("#cluster-operation-log", RichLog).write(f"[{elapsed:>4}s] {message}")
        self.query_one("#cluster-operation-status", Static).update(
            f"{self._operation_label} · {message}"
        )

    def _finish_operation(self, label: str, *, failed: bool) -> None:
        elapsed = int(time.monotonic() - self._operation_started)
        self._operation_running = False
        self.query_one("#cluster-operation-status", Static).update(
            f"{'Failed' if failed else 'Completed'} · {label} · {elapsed}s"
        )
        for button in self.query(".cluster-action"):
            if isinstance(button, Button):
                button.disabled = False

    def _run_operation(
        self,
        label: str,
        action: Any,
        callback: Any = None,
        *,
        accepts_progress: bool = False,
    ) -> None:
        if self._operation_running:
            self.notify(f"{self._operation_label} is already running.", severity="warning")
            return
        self._operation_running = True
        self._operation_label = label
        self._operation_started = time.monotonic()
        for button in self.query(".cluster-action"):
            if isinstance(button, Button):
                button.disabled = True
        log = self.query_one("#cluster-operation-log", RichLog)
        log.write(f"▶ {label} started. The console remains responsive; progress appears here.")

        def run() -> None:
            try:
                def progress(message: Any) -> None:
                    self.app.call_from_thread(self._operation_progress, str(message))

                if self._provider_lock.locked():
                    progress("Waiting for the current bounded resource probe to finish.")
                with self._provider_lock:
                    value = action(progress) if accepts_progress else action()
            except Exception as error:
                self.app.call_from_thread(log.write, f"{label}: {type(error).__name__}: {error}")
                self.app.call_from_thread(self._finish_operation, label, failed=True)
                return
            self.app.call_from_thread(
                log.write,
                f"✓ {label} completed\n{_structured_text(value, limit=20)}",
            )
            if callback is not None:
                self.app.call_from_thread(callback, value)
            self.app.call_from_thread(self._finish_operation, label, failed=False)

        Thread(
            target=run, daemon=True, name=f"lambdaforge-tui-{label.lower().replace(' ', '-')}"
        ).start()


class WorkWorkspace(ResearchWorkspace):
    """Semantic Work summary with Attempts, logs, resources and Study handoff."""

    def __init__(self, work: Mapping[str, Any], services: Any) -> None:
        self.work = dict(work)
        self.services = services
        self._log_loading = False
        self._log_text = ""
        self._log_initialized = False
        self._log_updated_at: float | None = None
        super().__init__(f"Work / {work.get('name', 'Work')}")

    def compose_workspace(self) -> ComposeResult:
        yield Static(
            f"{self.work.get('name')} · {str(self.work.get('state', 'unknown')).upper()} · "
            f"{self.work.get('cluster', 'local')}",
            classes="workspace-header",
        )
        with Horizontal(classes="workspace-actions"):
            yield Button("Cancel", id="work-cancel", variant="warning")
            yield Button("Retry latest Attempt", id="work-retry")
            yield Button("Delete", id="work-delete", variant="error")
        if isinstance(self.work.get("study"), Mapping):
            yield Button("Open Study", id="work-open-study", variant="primary")
        with TabbedContent(initial="work-summary", id="work-tabs"):
            with TabPane("Summary", id="work-summary"):
                yield Static(
                    f"State       {self.work.get('state')}\n"
                    f"Progress    {self.work.get('progress')}\n"
                    f"Attempts    {len(self.work.get('attempt_history', ()))}\n"
                    f"Cluster     {self.work.get('cluster')}",
                    classes="workspace-panel",
                )
            with TabPane("Attempts", id="work-attempts"):
                yield DataTable(id="attempt-table", cursor_type="row", zebra_stripes=True)
            with TabPane("Logs", id="work-logs"):
                yield Static(
                    "Waiting for the first log snapshot…",
                    id="work-log-status",
                    classes="freshness-line",
                )
                yield RichLog(id="work-log-content", wrap=False, auto_scroll=True)
            with TabPane("Resources", id="work-resources"):
                yield Static(
                    json.dumps(self.work.get("resources", {}), indent=2), classes="workspace-panel"
                )
            with TabPane("Outputs", id="work-outputs"):
                yield Static(
                    "Managed output evidence is available in the terminal result envelope."
                )

    def on_mount(self) -> None:
        table = self.query_one("#attempt-table", DataTable)
        table.add_columns("Attempt", "State", "Cluster", "Started", "Duration")
        for index, attempt in enumerate(self.work.get("attempt_history", ()), 1):
            if isinstance(attempt, Mapping):
                table.add_row(
                    str(index),
                    str(attempt.get("state", "unknown")),
                    str(attempt.get("cluster", self.work.get("cluster", "local"))),
                    str(attempt.get("created_at_utc", attempt.get("started_at_utc", "-"))),
                    format_duration(attempt.get("duration_seconds")),
                )
        if self.work.get("primary_job_id"):
            self._refresh_logs()
            self.set_interval(2.0, self._refresh_logs)
        else:
            self.query_one("#work-log-status", Static).update(
                "No persisted Job is associated with this Work."
            )

    def _refresh_logs(self) -> None:
        """Poll bounded logs while preserving the last readable snapshot."""
        job_id = self.work.get("primary_job_id")
        if not job_id or self._log_loading or self.app.screen is not self:
            return
        self._log_loading = True
        status = self.query_one("#work-log-status", Static)
        if self._log_updated_at is None:
            status.update("Loading Work logs…")
        else:
            age = max(0, int(time.monotonic() - self._log_updated_at))
            status.update(f"Live logs · last update {age}s ago · refreshing…")

        def load() -> None:
            try:
                report = self.services.work_logs(str(job_id))
                text = str(report.get("text", ""))
            except Exception as error:
                self.app.call_from_thread(self._log_failed, error)
            else:
                self.app.call_from_thread(self._logs_loaded, text)

        Thread(target=load, daemon=True, name="lambdaforge-tui-work-logs").start()

    def _logs_loaded(self, text: str) -> None:
        log = self.query_one("#work-log-content", RichLog)
        previous = self._log_text
        if not self._log_initialized or text != previous:
            at_end = log.is_vertical_scroll_end
            scroll_y = log.scroll_y
            if previous and text.startswith(previous):
                addition = text[len(previous) :].lstrip("\n")
                if addition:
                    log.write(addition)
            else:
                log.clear()
                log.write(text or "No Work output has been emitted yet.")
            if at_end:
                log.scroll_end(animate=False)
            else:
                log.scroll_to(y=scroll_y, animate=False)
            self._log_text = text
            self._log_initialized = True
        self._log_loading = False
        self._log_updated_at = time.monotonic()
        self.query_one("#work-log-status", Static).update("Live logs · updated just now")

    def _log_failed(self, error: Exception) -> None:
        self._log_loading = False
        suffix = " · showing the previous snapshot" if self._log_updated_at is not None else ""
        self.query_one("#work-log-status", Static).update(
            f"Log refresh failed · {type(error).__name__}: {error}{suffix}"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "work-open-study":
            self.app.push_screen(StudyWorkspace(self.work, self.services))
        elif event.button.id == "work-cancel":
            preview = {
                "work": self.work.get("name"),
                "execution": self._selector,
                "will_stop": "all active Attempts and their descendant Run processes",
                "will_preserve": "logs, results and published datasets",
            }
            self.app.push_screen(ExactConfirmation("Cancel Work", preview), self._apply_cancel)
        elif event.button.id == "work-retry":
            job_id = self.work.get("primary_job_id")
            if job_id:
                self._operation("Retry", lambda: self.services.retry_job(str(job_id)))
        elif event.button.id == "work-delete":
            self._operation(
                "Delete preview",
                lambda: self.services.delete_work(self._selector, apply=False),
                self._confirm_delete,
            )

    def _apply_cancel(self, confirmed: bool | None) -> None:
        if confirmed:
            self._operation("Cancel", lambda: self.services.cancel_work(self._selector))

    def _confirm_delete(self, preview: Mapping[str, Any]) -> None:
        self.app.push_screen(
            ExactConfirmation("Delete Work history and owned Attempt state", preview),
            self._apply_delete,
        )

    def _apply_delete(self, confirmed: bool | None) -> None:
        if confirmed:
            self._operation(
                "Delete", lambda: self.services.delete_work(self._selector, apply=True)
            )

    @property
    def _selector(self) -> str:
        """Use an exact persisted identity so equal display names remain independent."""
        return str(
            self.work.get("work_id")
            or self.work.get("execution_id")
            or self.work.get("primary_job_id")
            or self.work.get("name", "")
        )

    def _operation(self, label: str, action: Any, callback: Any = None) -> None:
        def run() -> None:
            try:
                value = action()
            except Exception as error:
                self.app.call_from_thread(
                    self.notify,
                    f"{type(error).__name__}: {error}",
                    title=f"{label} failed",
                    severity="error",
                )
                return
            self.app.call_from_thread(
                self.notify,
                _structured_text(value, heading="COMPLETED", limit=12),
                title=label,
            )
            if callback is not None:
                self.app.call_from_thread(callback, value)

        Thread(target=run, daemon=True, name=f"lambdaforge-work-{label.lower()}").start()


class DatasetWorkspace(ResearchWorkspace):
    """DatasetVersion summary with discoverable lifecycle sections."""

    def __init__(self, dataset: Mapping[str, Any], services: Any) -> None:
        self.dataset = dict(dataset)
        self.services = services
        self._members: list[Mapping[str, Any]] = []
        self._members_loading = False
        self._members_loaded = False
        name = f"{dataset.get('name')}@{dataset.get('version')}"
        super().__init__(f"Datasets / {name}")

    def compose_workspace(self) -> ComposeResult:
        name = f"{self.dataset.get('name')}@{self.dataset.get('version')}"
        yield Static(f"{name} · immutable DatasetVersion", classes="workspace-header")
        with Horizontal(id="dataset-toolbar", classes="workspace-actions"):
            yield Static(
                "Ready · destructive actions require an exact preview and confirmation.",
                id="dataset-operation-status",
                classes="status-line",
            )
            yield Button("Delete DatasetVersion…", id="dataset-delete", variant="error")
        with TabbedContent(initial="dataset-summary", id="dataset-tabs"):
            with TabPane("Summary", id="dataset-summary"):
                with Grid(classes="dataset-summary-grid"):
                    yield Static(
                        f"IDENTITY\n{self.selector}\nimmutable content",
                        classes="metric-card",
                    )
                    yield Static(
                        f"SCALE\n{format_value(self.dataset.get('sample_count'))} members\n"
                        f"{self._bytes(self._dataset_size())}",
                        classes="metric-card",
                    )
                    yield Static(
                        f"LOCATIONS\n{len(self._sequence(self.dataset.get('placements')))} managed\n"
                        + (self._location_names() or "not materialized"),
                        classes="metric-card",
                    )
                    yield Static(
                        f"STRUCTURE\n{len(self._mapping(self.dataset.get('partitions')))} partitions\n"
                        f"{len(self._sequence(self.dataset.get('lineage')))} lineage inputs",
                        classes="metric-card",
                    )
                yield Static(
                    f"Content ID  {self.dataset.get('content_id', self.dataset.get('dataset_id', 'unavailable'))}\n"
                    f"Created     {self.dataset.get('created_at_utc', 'unavailable')}\n\n"
                    "Open a tab for bounded evidence; member and integrity reads never mutate the dataset.",
                    classes="workspace-panel",
                )
                yield Static("SPLITS AND PRIMARY TARGET", classes="section-title dataset-split-title")
                yield DataTable(id="dataset-split-table", zebra_stripes=True)
                yield Static(
                    "Loading exact per-split target counts from the logical member index…",
                    id="dataset-split-note",
                    classes="legend",
                )
            with TabPane("Members", id="dataset-members"):
                with Vertical(id="dataset-members-loading", classes="inline-loading"):
                    yield LoadingIndicator()
                    yield Label("Loading the first bounded page of logical members…")
                yield DataTable(id="dataset-member-table", cursor_type="row", zebra_stripes=True)
                yield Static(
                    "The first 200 logical records load automatically when this tab is opened.",
                    id="dataset-member-content",
                    classes="detail-panel",
                )
            with TabPane("Partitions", id="dataset-partitions"):
                yield VerticalScroll(
                    Static(
                        _structured_text(self.dataset.get("partitions", {}), heading="PARTITIONS")
                    )
                )
            with TabPane("Locations", id="dataset-locations"):
                yield VerticalScroll(
                    Static(
                        _structured_text(
                            self.dataset.get("placements", ()), heading="MANAGED LOCATIONS"
                        )
                    )
                )
            with TabPane("Lineage", id="dataset-lineage"):
                yield VerticalScroll(
                    Static(
                        _structured_text(
                            self.dataset.get("lineage", ()), heading="SCIENTIFIC LINEAGE"
                        )
                    )
                )
            with TabPane("Stats", id="dataset-stats"):
                with Vertical(classes="dataset-action-state", id="dataset-stats-action"):
                    yield Static(
                        "Physical statistics walk the selected managed placement and may be slow for large datasets."
                    )
                    yield Button("Compute physical statistics", id="dataset-show-stats", variant="primary")
                yield VerticalScroll(Static(id="dataset-stat-content"))
            with TabPane("Integrity", id="dataset-integrity"):
                with Vertical(classes="dataset-action-state", id="dataset-verify-action"):
                    yield Static(
                        "Integrity verification reads manifests and checksums. Run it explicitly when required."
                    )
                    yield Button("Verify integrity", id="dataset-verify", variant="primary")
                yield VerticalScroll(
                    Static("Not checked in this console session.", id="dataset-integrity-content")
                )

    @property
    def selector(self) -> str:
        return f"{self.dataset.get('name')}@{self.dataset.get('version')}"

    def _location_names(self) -> str:
        return ", ".join(
            str(item.get("cluster", "unknown"))
            for item in self._sequence(self.dataset.get("placements"))
            if isinstance(item, Mapping)
        )

    @staticmethod
    def _mapping(value: Any) -> Mapping[str, Any]:
        return value if isinstance(value, Mapping) else {}

    @staticmethod
    def _sequence(value: Any) -> Sequence[Any]:
        return value if isinstance(value, Sequence) and not isinstance(value, str | bytes) else ()

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

    def on_button_pressed(self, event: Button.Pressed) -> None:
        target: str | None = None
        action: Any = None
        if event.button.id == "dataset-show-stats":
            target = "#dataset-stat-content"
            action = partial(self.services.dataset_stats, self.selector)
            self.query_one("#dataset-stats-action").display = False
            self.query_one(target, Static).update("Computing exact physical statistics…")
        elif event.button.id == "dataset-verify":
            target = "#dataset-integrity-content"
            action = partial(self.services.dataset_verify, self.selector)
            self.query_one("#dataset-verify-action").display = False
            self.query_one(target, Static).update("Verifying manifests, identity and checksums…")
        elif event.button.id == "dataset-delete":
            self._preview_delete()
            return
        if target is not None and action is not None:
            self._load_into(target, action)

    def on_mount(self) -> None:
        self.query_one("#dataset-member-table", DataTable).display = False
        self._apply_logical_summary(self.dataset, exact=False)
        self._load_logical_summary()

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        if event.tabbed_content.id == "dataset-tabs" and event.pane.id == "dataset-members":
            self._load_members(partial(self.services.dataset_members, self.selector))

    def _load_logical_summary(self) -> None:
        def load() -> None:
            try:
                value = self.services.dataset_summary(self.selector)
            except Exception as error:
                self.app.call_from_thread(
                    self.query_one("#dataset-split-note", Static).update,
                    "Exact class counts could not be read from a managed placement. "
                    f"Cached split totals remain visible. {type(error).__name__}: {error}",
                )
                return
            self.app.call_from_thread(self._apply_logical_summary, value, exact=True)

        Thread(target=load, daemon=True, name="lambdaforge-dataset-summary").start()

    def _apply_logical_summary(self, value: Mapping[str, Any], *, exact: bool) -> None:
        table = self.query_one("#dataset-split-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Split", "Members", "Positive", "Negative", "Other labels")
        partitions = self._mapping(value.get("partitions"))
        splits = self._mapping(partitions.get("split")) or self._mapping(value.get("splits"))
        cross = self._mapping(self._mapping(value.get("partition_targets")).get("split"))
        primary_target = self._primary_target(cross)
        for split, count in self._ordered_splits(splits):
            distribution = self._mapping(
                self._mapping(cross.get(split)).get(primary_target)
                if primary_target is not None
                else None
            )
            positive, negative, other = self._binary_counts(distribution)
            table.add_row(
                split,
                str(count),
                str(positive) if positive is not None else "—",
                str(negative) if negative is not None else "—",
                self._distribution_text(other),
            )
        if not splits:
            table.add_row("No split partition", str(value.get("member_count", self.dataset.get("sample_count", 0))), "—", "—", "—")
        target_text = metric_display_name(primary_target) if primary_target else "no binary target recorded"
        self.query_one("#dataset-split-note", Static).update(
            f"{'Exact logical-index counts' if exact else 'Cached manifest counts'} · primary target: {target_text}. "
            "Dash means the dataset does not declare that binary class; it is not interpreted as zero."
        )

    @classmethod
    def _primary_target(cls, cross: Mapping[str, Any]) -> str | None:
        names = {
            str(name)
            for split in cross.values()
            if isinstance(split, Mapping)
            for name in split
        }
        return next((name for name in ("label", "target", "class") if name in names), None) or (
            sorted(names)[0] if names else None
        )

    @staticmethod
    def _ordered_splits(splits: Mapping[str, Any]) -> list[tuple[str, Any]]:
        priority = {"train": 0, "validation": 1, "val": 1, "test": 2}
        return sorted(
            ((str(name), count) for name, count in splits.items()),
            key=lambda item: (priority.get(item[0].lower(), 9), item[0]),
        )

    @staticmethod
    def _binary_counts(distribution: Mapping[str, Any]) -> tuple[int | None, int | None, dict[str, int]]:
        if not distribution:
            return None, None, {}
        positive = 0
        negative = 0
        has_positive = False
        has_negative = False
        other: dict[str, int] = {}
        for raw, count in distribution.items():
            label = str(raw).strip('"').lower()
            amount = int(count)
            if label in {"1", "true", "positive", "pos", "yes"}:
                positive += amount
                has_positive = True
            elif label in {"0", "-1", "false", "negative", "neg", "no"}:
                negative += amount
                has_negative = True
            else:
                other[str(raw)] = amount
        return positive if has_positive else None, negative if has_negative else None, other

    @staticmethod
    def _distribution_text(distribution: Mapping[str, int]) -> str:
        return ", ".join(f"{name}: {count}" for name, count in distribution.items()) or "—"

    def _load_members(self, action: Any) -> None:
        if self._members_loading or self._members_loaded:
            return
        self._members_loading = True

        def load() -> None:
            try:
                value = action()
            except Exception as error:
                self.app.call_from_thread(
                    self._members_failed,
                    error,
                )
                return
            self.app.call_from_thread(self._apply_members, value)

        Thread(target=load, daemon=True, name="lambdaforge-dataset-members").start()

    def _members_failed(self, error: Exception) -> None:
        self._members_loading = False
        self.query_one("#dataset-members-loading").display = False
        self.query_one("#dataset-member-content", Static).update(
            f"Members could not be read: {type(error).__name__}: {error}"
        )

    def _apply_members(self, value: Mapping[str, Any]) -> None:
        self._members_loading = False
        self._members_loaded = True
        self.query_one("#dataset-members-loading").display = False
        members = [
            item for item in self._sequence(value.get("members")) if isinstance(item, Mapping)
        ]
        self._members = members
        table = self.query_one("#dataset-member-table", DataTable)
        table.display = True
        table.clear(columns=True)
        table.add_columns("Member", "Partitions", "Targets", "Assets")
        for member in members:
            table.add_row(
                str(member.get("id", "-")),
                self._compact_mapping(member.get("partitions")),
                self._compact_mapping(member.get("targets")),
                str(len(self._mapping(member.get("assets")))),
                key=str(member.get("id", "-")),
            )
        self.query_one("#dataset-member-content", Static).update(
            f"Showing {len(members)} bounded members from offset {value.get('offset', 0)}. "
            "Select a row to inspect its metadata and logical assets."
        )
        if members:
            table.move_cursor(row=0)
            self._show_member(0)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id == "dataset-member-table":
            self._show_member(event.cursor_row)

    def _show_member(self, row: int) -> None:
        if 0 <= row < len(self._members):
            self.query_one("#dataset-member-content", Static).update(
                _structured_text(self._members[row], heading="SELECTED MEMBER", limit=36)
            )

    def _load_into(self, target: str, action: Any) -> None:
        def load() -> None:
            try:
                value = action()
                text = self._dataset_operation_text(target, value)
            except Exception as error:
                text = f"{type(error).__name__}: {error}"
                self.app.call_from_thread(self._dataset_operation_failed, target, text)
                return
            self.app.call_from_thread(self.query_one(target, Static).update, text)

        Thread(target=load, daemon=True, name="lambdaforge-dataset-detail").start()

    def _dataset_operation_failed(self, target: str, message: str) -> None:
        self.query_one(target, Static).update(message)
        action = "#dataset-stats-action" if target == "#dataset-stat-content" else "#dataset-verify-action"
        self.query_one(action).display = True

    def _preview_delete(self) -> None:
        button = self.query_one("#dataset-delete", Button)
        if button.disabled:
            return
        button.disabled = True
        self._delete_status("Preparing exact deletion preview…")

        def load() -> None:
            try:
                preview = self.services.delete_dataset(self.selector, apply=False)
            except Exception as error:
                self.app.call_from_thread(self._delete_failed, error)
                return
            self.app.call_from_thread(self._confirm_delete, preview)

        Thread(target=load, daemon=True, name="lambdaforge-dataset-delete-preview").start()

    def _confirm_delete(self, preview: Mapping[str, Any]) -> None:
        self.query_one("#dataset-delete", Button).disabled = False
        if not preview.get("safe"):
            self._delete_status("Deletion blocked · the exact preview is not safe.", error=True)
            self.notify(
                "Dataset deletion is not safe; inspect the exact placement reasons.",
                severity="error",
            )
            return
        self._delete_status("Preview ready · confirmation required.")
        self.app.push_screen(
            ExactConfirmation(f"Delete {self.selector} and every listed placement", preview),
            self._apply_delete,
        )

    def _apply_delete(self, confirmed: bool | None) -> None:
        if not confirmed:
            self.query_one("#dataset-delete", Button).disabled = False
            self._delete_status("Deletion cancelled · no changes made.")
            return
        self.query_one("#dataset-delete", Button).disabled = True
        self._delete_status("Deleting managed placements and registry entries…")

        def apply() -> None:
            try:
                result = self.services.delete_dataset(self.selector, apply=True)
            except Exception as error:
                self.app.call_from_thread(self._delete_failed, error)
                return
            self.app.call_from_thread(self._delete_complete, result)

        Thread(target=apply, daemon=True, name="lambdaforge-dataset-delete-apply").start()

    def _delete_failed(self, error: Exception) -> None:
        self.query_one("#dataset-delete", Button).disabled = False
        self._delete_status(f"Delete failed · {type(error).__name__}: {error}", error=True)
        self.notify(f"{type(error).__name__}: {error}", title="Delete failed", severity="error")

    def _delete_complete(self, _result: Mapping[str, Any]) -> None:
        self._delete_status("Deleted · managed storage and registry are consistent.")
        self.notify(f"{self.selector} was deleted from managed storage and registries.")
        self.app.pop_screen()
        try:
            screen = self.app.query_one("#datasets")
        except Exception:
            return
        reload_screen = getattr(screen, "reload", None)
        if callable(reload_screen):
            reload_screen()

    def _delete_status(self, message: str, *, error: bool = False) -> None:
        status = self.query_one("#dataset-operation-status", Static)
        status.update(message)
        status.set_class(error, "operation-error")

    @classmethod
    def _dataset_operation_text(cls, target: str, value: Any) -> str:
        if not isinstance(value, Mapping):
            return _structured_text(value)
        if target == "#dataset-stat-content":
            return (
                "PHYSICAL SCALE\n"
                f"Members              {format_value(value.get('member_count', value.get('sample_count')))}\n"
                f"Files                {format_value(value.get('file_count'))}\n"
                f"Stored size          {cls._bytes(value.get('size_bytes'))}\n"
                f"Format               {format_value(value.get('format'))}\n\n"
                "LOGICAL ORGANIZATION\n"
                + _structured_text(
                    {
                        "splits": value.get("splits", {}),
                        "partitions": value.get("partitions", {}),
                        "index_summary": value.get("index_summary", {}),
                    },
                    limit=40,
                )
            )
        if target == "#dataset-integrity-content":
            errors = value.get("errors", ())
            errors = errors if isinstance(errors, Sequence) and not isinstance(errors, str) else ()
            return (
                f"{'✓ VERIFIED' if value.get('valid') else '✗ INTEGRITY ISSUE'}\n"
                f"Dataset              {value.get('dataset', value.get('dataset_id', 'unavailable'))}\n"
                f"Placement            {value.get('cluster', 'local')} · {value.get('placement_state', 'observed')}\n"
                f"Checked files        {format_value(value.get('file_count'))}\n"
                f"Checked size         {cls._bytes(value.get('size_bytes'))}\n\n"
                "ERRORS\n" + ("\n".join(f"• {error}" for error in errors) or "None")
            )
        return _structured_text(value)

    def _dataset_size(self) -> int | None:
        direct = self.dataset.get("size_bytes")
        if isinstance(direct, int):
            return direct
        for placement in self._sequence(self.dataset.get("placements")):
            if isinstance(placement, Mapping) and isinstance(placement.get("size_bytes"), int):
                return int(placement["size_bytes"])
        return None

    @classmethod
    def _compact_mapping(cls, value: Any) -> str:
        mapping = cls._mapping(value)
        if not mapping:
            return "—"
        return " · ".join(f"{key}={format_value(item)}" for key, item in mapping.items())


class ResultWorkspace(ResearchWorkspace):
    """Completed Execution summary and persisted Study Analysis."""

    def __init__(self, result: Mapping[str, Any], services: Any) -> None:
        self.result = dict(result)
        self.services = services
        super().__init__(f"Results / {result.get('name', 'Execution')}")

    def compose_workspace(self) -> ComposeResult:
        yield Static(
            f"{self.result.get('name')} · {str(self.result.get('status', 'unknown')).upper()}",
            classes="workspace-header",
        )
        with Horizontal(classes="workspace-actions"):
            yield Button("Analyze / refresh", id="result-analyze", variant="primary")
            yield Button("Export HTML report", id="result-report")
            yield Button("Delete", id="result-delete", variant="error")
        with TabbedContent(initial="result-summary"):
            with TabPane("Summary", id="result-summary"):
                yield Static(
                    _structured_text(self.result.get("summary", {}), heading="RESULT SUMMARY"),
                    classes="workspace-panel",
                )
            with TabPane("Runs", id="result-runs"):
                yield Static(f"{len(self.result.get('runs', ()))} persisted Runs")
            with TabPane("Metrics", id="result-metrics"):
                yield Static("Metric summaries are retained in the execution envelope.")
            with TabPane("Analysis", id="result-analysis"):
                yield VerticalScroll(
                    Static("Loading persisted Study Analysis…", id="result-analysis-content")
                )
            with TabPane("Artifacts", id="result-artifacts"):
                yield VerticalScroll(
                    Static(
                        _structured_text(
                            self.result.get("artifacts", ()), heading="MANAGED ARTIFACTS"
                        )
                    )
                )
            with TabPane("Logs", id="result-logs"):
                yield Static("Logs remain linked through the originating Work Attempts.")
            with TabPane("Metadata", id="result-metadata"):
                yield VerticalScroll(
                    Static(_structured_text(self.result, heading="PERSISTED METADATA"))
                )

    def on_mount(self) -> None:
        selector = self.result.get("execution_id")
        if not selector:
            return

        def load() -> None:
            try:
                analysis = self.services.result_analysis(str(selector))
                winner = analysis.get("winner", {})
                seeds = analysis.get("seed_analysis", {})
                text = (
                    f"Winner: {(winner.get('screening_winner') or {}).get('trial', 'unavailable')}\n"
                    f"Confirmation: {winner.get('confirmation_status', 'unavailable')}\n"
                    f"Empirical seed stability: {seeds.get('status', 'unavailable')}\n\n"
                    "PARAMETERS\n"
                    + "\n".join(
                        f"{name}: {detail.get('importance', 0):.3f} [{detail.get('reliability', 'low')}]"
                        for name, detail in analysis.get("parameter_importance", {}).items()
                    )
                    + "\n\nFINDINGS\n"
                    + "\n\n".join(
                        f"{item.get('title')}\n{item.get('statement')}"
                        for item in analysis.get("findings", ())
                    )
                )
            except Exception as error:
                text = f"Analysis unavailable: {type(error).__name__}: {error}"
            self.app.call_from_thread(
                self.query_one("#result-analysis-content", Static).update, text
            )

        Thread(target=load, daemon=True, name="lambdaforge-tui-result-analysis").start()

    @property
    def selector(self) -> str:
        return str(self.result.get("execution_id", self.result.get("name", "")))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "result-analyze":
            self._result_operation(
                "Analyze", lambda: self.services.analyze(self.selector, recompute=True)
            )
        elif event.button.id == "result-report":
            destination = Path.cwd() / f"{self.selector}-analysis.html"
            self._result_operation(
                "Report", lambda: {"path": str(self.services.report(self.selector, destination))}
            )
        elif event.button.id == "result-delete":
            self._result_operation(
                "Delete preview",
                lambda: self.services.delete_result(self.selector, apply=False),
                self._confirm_result_delete,
            )

    def _confirm_result_delete(self, preview: Mapping[str, Any]) -> None:
        self.app.push_screen(
            ExactConfirmation(f"Delete Result {self.selector}", preview),
            self._apply_result_delete,
        )

    def _apply_result_delete(self, confirmed: bool | None) -> None:
        if confirmed:
            self._result_operation(
                "Delete", lambda: self.services.delete_result(self.selector, apply=True)
            )

    def _result_operation(self, label: str, action: Any, callback: Any = None) -> None:
        def run() -> None:
            try:
                value = action()
            except Exception as error:
                self.app.call_from_thread(
                    self.notify,
                    f"{type(error).__name__}: {error}",
                    title=f"{label} failed",
                    severity="error",
                )
                return
            self.app.call_from_thread(
                self.notify,
                _structured_text(value, heading="COMPLETED", limit=12),
                title=label,
            )
            if callback is not None:
                self.app.call_from_thread(callback, value)

        Thread(target=run, daemon=True, name=f"lambdaforge-result-{label.lower()}").start()


__all__ = [
    "ClusterWorkspace",
    "CredentialDialog",
    "DatasetWorkspace",
    "EpochWorkspace",
    "ExactConfirmation",
    "HpoActionWorkspace",
    "HpoParameterWorkspace",
    "ResultWorkspace",
    "SeedWorkspace",
    "StudyWorkspace",
    "TrialWorkspace",
    "WorkWorkspace",
]
