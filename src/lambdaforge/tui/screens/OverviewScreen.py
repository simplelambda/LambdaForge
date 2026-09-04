"""Visual, domain-separated Research Console overview."""

# Human-facing terminal copy is intentionally kept as complete strings.
# ruff: noqa: E501

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Grid, Vertical, VerticalScroll
from textual.widgets import DataTable, Label, LoadingIndicator, Static

from lambdaforge.tui.screens.Base import DataScreen, EntitySelected
from lambdaforge.tui.viewmodels import (
    entity_key,
    format_duration,
    format_value,
    objective_display_name,
)
from lambdaforge.tui.widgets import ResourceDashboard


class OverviewScreen(DataScreen):
    """Show Clusters, ordinary Work and Studies as distinct research concepts."""

    TITLE = "Overview · research at a glance"
    BINDINGS = [
        ("right", "open_selected", "Open details"),
        ("c", "focus_clusters", "Clusters"),
        ("w", "focus_work", "Work"),
        ("s", "focus_studies", "Studies"),
    ]

    def __init__(self, loader: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(loader, *args, **kwargs)
        self._selected_panel = "overview-clusters"
        self._selected_key: str | None = None
        self._rendering_snapshot = False

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label(self.TITLE, classes="screen-title"),
            Static("Loading…", id="screen-status", classes="status-line"),
            LoadingIndicator(id="screen-loading"),
            Grid(
                Vertical(
                    Label("CLUSTERS", classes="overview-section-title"),
                    DataTable(id="overview-clusters", cursor_type="row", zebra_stripes=True),
                    classes="overview-card",
                ),
                Vertical(
                    Label("WORK", classes="overview-section-title"),
                    DataTable(id="overview-work", cursor_type="row", zebra_stripes=True),
                    classes="overview-card",
                ),
                Vertical(
                    Label("STUDIES", classes="overview-section-title"),
                    DataTable(id="overview-studies", cursor_type="row", zebra_stripes=True),
                    classes="overview-card",
                ),
                id="overview-grid",
            ),
            VerticalScroll(
                Static(
                    "Select a Cluster, Work or Study to see a visual summary.",
                    id="overview-detail",
                ),
                ResourceDashboard(id="overview-resource-dashboard"),
                id="overview-detail-scroll",
            ),
            Static(
                "Waiting for the first update",
                id="screen-freshness",
                classes="freshness-line",
            ),
            Static(
                "Tab/c/w/s switch panel  ·  Enter/right open details  ·  r refresh  ·  "
                "q or Exit closes the console",
                id="overview-controls",
                classes="legend",
            ),
        )

    def on_mount(self) -> None:
        super().on_mount()
        self.set_interval(2.0, self._refresh_visible)

    def _refresh_visible(self) -> None:
        # Root screens remain mounted (and technically ``display``-visible) below a
        # pushed entity workspace.  Do not let the overview's multi-cluster poll
        # compete with an interactive bootstrap/doctor/delete operation there.
        if self.display and self.app.screen is self.app.screen_stack[0]:
            self.reload()

    def _loaded(self, value: Any) -> None:
        panel = self._selected_panel
        key = self._selected_key
        self._accept_snapshot(value, status="Live snapshot · refreshed automatically")
        self._record_resources(value)
        self._rendering_snapshot = True
        try:
            self._populate_clusters(value)
            self._populate_work(value)
            self._populate_studies(value)
            selected = self.query_one(f"#{panel}", DataTable)
            self._restore_selection(selected, key)
        finally:
            self._rendering_snapshot = False
        self._remember_selection(selected)
        self._update_preview(selected)

    def populate(self, table: DataTable[Any], value: Any) -> None:
        """DataScreen compatibility; Overview owns three explicit tables instead."""

    @staticmethod
    def _items(value: Any) -> list[Mapping[str, Any]]:
        if not isinstance(value, Mapping):
            return []
        work = value.get("work", {})
        if not isinstance(work, Mapping):
            return []
        return [item for item in work.get("items", ()) if isinstance(item, Mapping)]

    @classmethod
    def _ordinary_work(cls, value: Any) -> list[Mapping[str, Any]]:
        return [item for item in cls._items(value) if not item.get("study_expected")]

    @classmethod
    def _studies(cls, value: Any) -> list[Mapping[str, Any]]:
        return [
            item
            for item in cls._items(value)
            if item.get("study_expected") or isinstance(item.get("study"), Mapping)
        ]

    @staticmethod
    def _clusters(value: Any) -> list[Mapping[str, Any]]:
        if not isinstance(value, Mapping):
            return []
        return [item for item in value.get("clusters", ()) if isinstance(item, Mapping)]

    def _populate_clusters(self, value: Any) -> None:
        table = self.query_one("#overview-clusters", DataTable)
        table.clear(columns=True)
        table.add_columns("Cluster", "State", "CPU", "RAM", "GPU")
        for item in self._clusters(value):
            observed = self._mapping(item.get("observed"))
            table.add_row(
                str(item.get("cluster", "-")),
                Text("● ONLINE", style="bold green")
                if item.get("online")
                else Text("○ OFFLINE", style="bold red"),
                self._percent_text(observed.get("cpu_load")),
                self._percent_text(self._ram_percent(observed)),
                self._percent_text(self._gpu_percent(observed)),
                key=str(item.get("cluster", "-")),
            )

    def _populate_work(self, value: Any) -> None:
        table = self.query_one("#overview-work", DataTable)
        table.clear(columns=True)
        table.add_columns("Work", "Target", "State", "Progress")
        for index, item in enumerate(self._ordinary_work(value)):
            table.add_row(
                str(item.get("name", "-")),
                str(item.get("cluster", "local")),
                self._state_cell(item.get("state")),
                self._progress_text(item.get("progress")),
                key=entity_key(item, kind="work", fallback_index=index),
            )

    def _populate_studies(self, value: Any) -> None:
        table = self.query_one("#overview-studies", DataTable)
        table.clear(columns=True)
        table.add_columns("Study", "State", "Objective", "Progress")
        for index, item in enumerate(self._studies(value)):
            study = self._mapping(item.get("study"))
            objective = self._mapping(study.get("objective"))
            table.add_row(
                str(item.get("name", "-")),
                self._state_cell(item.get("state")),
                objective_display_name(objective) if objective else "Telemetry unavailable",
                self._study_progress(study, item.get("progress")),
                key=entity_key(item, kind="study", fallback_index=index),
            )

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id and event.data_table.id.startswith("overview-"):
            event.stop()
            if self._rendering_snapshot or event.data_table is not self.app.focused:
                return
            self._remember_selection(event.data_table)
            self._update_preview(event.data_table)

    def on_descendant_focus(self, event: Any) -> None:
        """Change the preview with panel focus, even when a table has only one row."""
        control = getattr(event, "control", None)
        if isinstance(control, DataTable) and control.id and control.id.startswith("overview-"):
            if self._rendering_snapshot:
                return
            self._remember_selection(control)
            self._update_preview(control)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id and event.data_table.id.startswith("overview-"):
            event.stop()
            self._open(event.data_table)

    def action_open_selected(self) -> None:
        table = self._focused_table()
        if table is not None:
            self._open(table)

    def action_focus_clusters(self) -> None:
        self.query_one("#overview-clusters", DataTable).focus()

    def action_focus_work(self) -> None:
        self.query_one("#overview-work", DataTable).focus()

    def action_focus_studies(self) -> None:
        self.query_one("#overview-studies", DataTable).focus()

    def _focused_table(self) -> DataTable[Any] | None:
        focused = self.app.focused
        if isinstance(focused, DataTable) and focused.id and focused.id.startswith("overview-"):
            return focused
        return None

    def _selection(self, table: DataTable[Any]) -> tuple[str, Mapping[str, Any]] | None:
        row = table.cursor_row
        values, kind = self._table_values(table)
        return (kind, values[row]) if 0 <= row < len(values) else None

    def _open(self, table: DataTable[Any]) -> None:
        selected = self._selection(table)
        if selected is None:
            return
        kind, item = selected
        self.post_message(
            EntitySelected(kind, str(item.get("cluster", "")) if kind == "cluster" else item)
        )

    def _remember_selection(self, table: DataTable[Any]) -> None:
        self._selected_panel = str(table.id or "overview-clusters")
        selected = self._selection(table)
        self._selected_key = self._entity_key(*selected) if selected is not None else None

    def _restore_selection(self, table: DataTable[Any], key: str | None) -> None:
        values, kind = self._table_values(table)
        row = 0
        if key is not None:
            row = next(
                (index for index, item in enumerate(values) if self._entity_key(kind, item) == key),
                0,
            )
        if values:
            table.move_cursor(row=row)

    def _table_values(self, table: DataTable[Any]) -> tuple[list[Mapping[str, Any]], str]:
        if table.id == "overview-clusters":
            return self._clusters(self.last_success), "cluster"
        if table.id == "overview-studies":
            return self._studies(self.last_success), "study"
        return self._ordinary_work(self.last_success), "work"

    @staticmethod
    def _entity_key(kind: str, item: Mapping[str, Any]) -> str:
        return entity_key(item, kind=kind)

    def _update_preview(self, table: DataTable[Any]) -> None:
        selected = self._selection(table)
        if selected is None:
            self.query_one("#overview-detail", Static).update(
                "This panel has no persisted entries yet."
            )
            self.query_one("#overview-resource-dashboard", ResourceDashboard).hide_dashboard()
            return
        kind, item = selected
        text = (
            self._cluster_detail(item)
            if kind == "cluster"
            else self._study_detail(item)
            if kind == "study"
            else self._work_detail(item)
        )
        self.query_one("#overview-detail", Static).update(text)
        dashboard = self.query_one("#overview-resource-dashboard", ResourceDashboard)
        if kind == "cluster":
            dashboard.show_cluster(
                str(item.get("cluster", "-")),
                self._mapping(item.get("observed")),
                self._mapping(item.get("personal")),
                ingest=False,
                sample_id=str(item.get("observed_at_utc") or "") or None,
            )
        else:
            dashboard.hide_dashboard()

    def _record_resources(self, value: Any) -> None:
        dashboard = self.query_one("#overview-resource-dashboard", ResourceDashboard)
        for cluster in self._clusters(value):
            name = str(cluster.get("cluster", "-"))
            observed = self._mapping(cluster.get("observed"))
            dashboard.ingest(
                name,
                observed,
                self._mapping(cluster.get("personal")),
                sample_id=str(cluster.get("observed_at_utc") or "") or None,
            )

    def _cluster_detail(self, cluster: Mapping[str, Any]) -> str:
        name = str(cluster.get("cluster", "-"))
        observed = self._mapping(cluster.get("observed"))
        personal = self._mapping(cluster.get("personal"))
        requested = self._mapping(personal.get("requested"))
        gpus = self._sequence(observed.get("gpus"))
        works = [
            item for item in self._ordinary_work(self.last_success) if item.get("cluster") == name
        ]
        studies = [item for item in self._studies(self.last_success) if item.get("cluster") == name]
        lines = [
            f"{name}  ·  {'ONLINE' if cluster.get('online') else 'OFFLINE'}  ·  {cluster.get('scheduler', 'unknown')} scheduler",
            f"Capacity  {observed.get('cpu_total', '?')} CPU cores  ·  {self._bytes(observed.get('ram_total_bytes'))} RAM  ·  {len(gpus)} GPU",
            f"Research  {len(works)} Work  ·  {len(studies)} Studies  ·  {personal.get('active_jobs', 0)} active LambdaForge Jobs",
            f"Mine requested  {requested.get('cpu_cores', 0)} CPU  ·  {self._bytes(requested.get('ram_bytes'))} RAM  ·  {requested.get('gpu_count', 0)} GPU",
            "Resource plots  CPU · RAM · GPU memory; cyan is total and magenta is your observed LambdaForge share.",
        ]
        if not self._mapping(personal.get("observed")).get("job_count"):
            lines.append("Personal live usage is unavailable; requested allocations remain exact.")
        if cluster.get("error"):
            lines.append(f"Attention  {cluster['error']}")
        return "\n".join(lines)

    def _work_detail(self, work: Mapping[str, Any]) -> str:
        progress = self._mapping(work.get("progress"))
        attempts = self._sequence(work.get("attempt_history"))
        lines = [
            f"{work.get('name', 'Work')}  ·  {str(work.get('state', 'unknown')).upper()}",
            f"Target       {work.get('cluster', 'local')}",
            f"Progress     {self._progress_bar(progress)}  {self._progress_text(progress)}",
            f"Attempts     {len(attempts)}",
            f"Created      {work.get('created_at_utc', 'unavailable')}",
            f"Last update  {work.get('updated_at_utc', 'unavailable')}",
            "",
            "ATTEMPT HISTORY",
        ]
        for attempt in attempts[-5:]:
            item = self._mapping(attempt)
            lines.append(
                f"  {item.get('label', 'Attempt')}  {str(item.get('state', 'unknown')).upper():10}  {item.get('cluster', 'local')}"
            )
        return "\n".join(lines)

    def _study_detail(self, work: Mapping[str, Any]) -> str:
        study = self._mapping(work.get("study"))
        if not study:
            return (
                f"{work.get('name', 'Study')}  ·  {str(work.get('state', 'unknown')).upper()}\n"
                "Study identity is preserved, but detailed telemetry is not currently reachable.\n"
                "Enter opens the Study workspace; Logs retain the terminal failure."
            )
        counts = self._mapping(study.get("counts"))
        objective = self._mapping(study.get("objective"))
        candidates = [self._mapping(item) for item in self._sequence(study.get("candidates"))]
        comparable = [
            item for item in candidates if self._number(item.get("selection_objective")) is not None
        ]
        reverse = str(objective.get("mode", "max")) == "max"
        leader = (
            sorted(
                comparable, key=lambda item: float(item["selection_objective"]), reverse=reverse
            )[0]
            if comparable
            else None
        )
        total = int(counts.get("planned_runs", study.get("planned_runs", 0)) or 0)
        completed = int(counts.get("completed_runs", 0) or 0) + int(
            counts.get("pruned_runs", 0) or 0
        )
        lines = [
            f"{work.get('name', 'Study')}  ·  {str(work.get('state', 'unknown')).upper()}  ·  {study.get('strategy', 'search')}",
            f"Objective     {objective_display_name(objective)} ({objective.get('mode', '-')})",
            f"Run progress  {self._bar(completed, total)}  {completed}/{total or '?'} terminal",
            f"Candidates    {counts.get('candidates', len(candidates))} total  ·  {counts.get('active_runs', 0)} active  ·  {counts.get('queued_runs', 0)} waiting  ·  {counts.get('pruned_runs', 0)} pruned",
            f"GPU time      {format_duration(self._mapping(study.get('cost')).get('gpu_seconds'))}",
            f"Current lead  Trial {leader.get('trial')} · {objective_display_name(objective)}={format_value(leader.get('selection_objective'))}"
            if leader is not None
            else "Current lead  not comparable yet",
            "",
            "Enter opens Trials, seeds, HPO diagnostics, analysis, resources and logs.",
        ]
        admission = self._mapping(study.get("admission")).get("current")
        if isinstance(admission, Mapping) and admission.get("status") == "probe-unavailable":
            lines.append(f"Attention  GPU observation paused: {admission.get('reason')}")
        return "\n".join(lines)

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

    @classmethod
    def _ram_percent(cls, observed: Mapping[str, Any]) -> float | None:
        total = cls._number(observed.get("ram_total_bytes"))
        available = cls._number(observed.get("ram_available_bytes"))
        return 100 * (total - available) / total if total and available is not None else None

    @classmethod
    def _gpu_percent(cls, observed: Mapping[str, Any]) -> float | None:
        values = [
            cls._number(cls._mapping(item).get("utilization_percent"))
            for item in cls._sequence(observed.get("gpus"))
        ]
        finite = [value for value in values if value is not None]
        return sum(finite) / len(finite) if finite else None

    @staticmethod
    def _percent_text(value: Any) -> str:
        return f"{float(value):.0f}%" if isinstance(value, int | float) else "n/a"

    @staticmethod
    def _state_cell(value: Any) -> Text:
        state = str(value or "unknown").upper()
        style = (
            "bold green"
            if state in {"SUCCEEDED", "COMPLETED", "FINISHED"}
            else "bold cyan"
            if state in {"RUNNING", "PREPARING", "STAGING", "QUEUED"}
            else "bold red"
            if state in {"FAILED", "CANCELLED"}
            else "bold yellow"
        )
        return Text(state, style=style)

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

    @classmethod
    def _progress_text(cls, value: Any) -> str:
        progress = cls._mapping(value)
        completed, total = progress.get("completed"), progress.get("total")
        unit = str(progress.get("unit") or "items")
        if isinstance(completed, int) and isinstance(total, int) and total > 0:
            return f"{completed}/{total} {unit} · {100 * completed / total:.0f}%"
        if isinstance(completed, int):
            return f"{completed} {unit} completed"
        return "No progress reported"

    @classmethod
    def _study_progress(cls, study: Mapping[str, Any], fallback: Any) -> str:
        counts = cls._mapping(study.get("counts"))
        completed = int(counts.get("completed_runs", 0) or 0)
        pruned = int(counts.get("pruned_runs", 0) or 0)
        active = int(counts.get("active_runs", 0) or 0)
        waiting = int(counts.get("queued_runs", 0) or 0)
        return (
            f"{completed} done · {active} live · {waiting} wait · {pruned} pruned"
            if counts
            else cls._progress_text(fallback)
        )

    @classmethod
    def _progress_bar(cls, progress: Mapping[str, Any]) -> str:
        completed, total = progress.get("completed"), progress.get("total")
        return (
            cls._bar(completed, total)
            if isinstance(completed, int) and isinstance(total, int) and total > 0
            else "░░░░░░░░░░░░"
        )

    @staticmethod
    def _bar(completed: int, total: int, width: int = 16) -> str:
        ratio = min(1.0, max(0.0, completed / total)) if total > 0 else 0.0
        filled = round(width * ratio)
        return "█" * filled + "░" * (width - filled)

    def detail(self, row: int) -> str:
        return "Overview uses the highlighted row in each domain panel."

    def open_row(self, row: int) -> None:
        self.action_open_selected()


__all__ = ["OverviewScreen"]
