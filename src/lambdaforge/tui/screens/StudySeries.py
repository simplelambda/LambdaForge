"""Research navigation over immutable Study revisions and their real native Jobs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from threading import Thread
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import Button, DataTable, LoadingIndicator, Static, TabbedContent, TabPane

from lambdaforge.controlplane.SecretRedactor import SecretRedactor
from lambdaforge.tui.screens.Workspace import ResearchWorkspace
from lambdaforge.tui.viewmodels import (
    format_duration,
    format_value,
    metric_display_name,
    structured_text,
)
from lambdaforge.tui.widgets import StudyOverviewDashboard


class SeriesReply(Message):
    def __init__(self, phase: str, value: Any = None, error: BaseException | None = None):
        super().__init__()
        self.phase, self.value, self.error = phase, value, error


class StudySeriesWorkspace(ResearchWorkspace):
    """One research family, with cumulative evidence and explicit physical provenance."""

    DEFAULT_CSS = """
    StudySeriesWorkspace #series-actions { height: 3; }
    StudySeriesWorkspace #series-actions Button { width: auto; margin-right: 1; }
    StudySeriesWorkspace #series-status { height: auto; min-height: 2; }
    StudySeriesWorkspace #series-summary { height: auto; padding: 1; }
    StudySeriesWorkspace #series-loading { height: 3; }
    StudySeriesWorkspace DataTable { height: 1fr; }
    StudySeriesWorkspace #series-plots { height: 20; }
    StudySeriesWorkspace #series-selection { height: auto; max-height: 15; }
    """

    def __init__(self, work: Mapping[str, Any], services: Any):
        super().__init__(f"Studies / {work.get('name', 'Study series')}")
        self.work, self.services = dict(work), services
        self.series_id = str(work["study_series_id"])
        self.value: dict[str, Any] | None = None
        self.busy = False

    def compose_workspace(self) -> ComposeResult:
        with Horizontal(id="series-actions"):
            yield Button("Refresh", id="series-refresh", flat=True)
            yield Button("Extend Study…", id="series-extend", variant="primary", flat=True)
            yield Button("Analysis HTML…", id="series-report", flat=True)
            yield Button("Export series…", id="series-export", flat=True)
        yield Static("Loading cumulative evidence…", id="series-status", markup=False)
        yield LoadingIndicator(id="series-loading")
        with TabbedContent(id="series-tabs"):
            with TabPane("Overview"):
                with VerticalScroll():
                    yield Static(id="series-summary", markup=False)
                    yield StudyOverviewDashboard(id="series-plots")
            with TabPane("Runs · all revisions", id="series-runs-pane"):
                yield DataTable(id="series-runs", cursor_type="row")
                yield Static(
                    "Select a Run to inspect its unchanged native provenance.",
                    id="series-selection",
                    markup=False,
                )
            with TabPane("Revisions & Jobs"):
                yield DataTable(id="series-revisions", cursor_type="row")
                yield Button("Open selected native Job…", id="series-open-job", flat=True)
            with TabPane("History"):
                yield DataTable(id="series-history", cursor_type="row")

    def on_mount(self) -> None:
        self._load()
        self.set_interval(5, self._poll)

    def _poll(self) -> None:
        if self.app.screen is self and self.value and self.value["counts"]["pending"]:
            self._load()

    def _start(self, phase: str, operation: Callable[[], Any]) -> None:
        if self.busy:
            return
        self.busy = True
        for button in self.query("#series-actions Button"):
            button.disabled = True
        self.query_one("#series-loading").display = self.value is None
        self.query_one("#series-status", Static).update(
            {
                "load": "Updating cumulative evidence…",
                "report": "Generating cumulative HTML…",
                "job": "Opening the native Job…",
                "export": "Exporting all revisions…",
            }[phase]
        )

        def run() -> None:
            try:
                value = operation()
            except (Exception, SystemExit) as error:
                self.post_message(SeriesReply(phase, error=error))
            else:
                self.post_message(SeriesReply(phase, value=value))

        Thread(target=run, daemon=True, name="lf-series-" + phase).start()

    def _load(self) -> None:
        live = self.query_one("#series-tabs", TabbedContent).active == "series-runs-pane"
        self._start(
            "load",
            lambda: {
                "overview": self.services.study_series_overview(
                    self.series_id, refresh=True, include_live=live
                ),
                "history": self.services.study_series_history(self.series_id),
            },
        )

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        if event.pane.id == "series-runs-pane" and self.value is not None:
            self._load()

    def on_series_reply(self, message: SeriesReply) -> None:
        message.stop()
        if not self.is_mounted:
            return
        self.busy = False
        for button in self.query("#series-actions Button"):
            button.disabled = False
        self.query_one("#series-loading").display = False
        status = self.query_one("#series-status", Static)
        if message.error:
            status.update(
                f"Could not update · {type(message.error).__name__}: "
                f"{SecretRedactor.redact(message.error)[:2000]}"
            )
            return
        if message.phase == "report":
            status.update(f"Cumulative report saved · {message.value}")
            return
        if message.phase == "export":
            status.update(f"Series export saved · {message.value['path']}")
            return
        if message.phase == "job":
            from lambdaforge.tui.screens.Workspace import StudyWorkspace

            self.app.push_screen(StudyWorkspace(message.value, self.services))
            return
        value = message.value["overview"]
        counts = value["counts"]
        status.update(f"{value['state']} · revision {len(value['revisions']) - 1} · updated now")
        if self.value == value:
            return
        positions = {
            table.id: (table.cursor_row, table.scroll_x, table.scroll_y)
            for table in self.query(DataTable)
        }
        self.value = value
        self.query_one("#series-summary", Static).update(
            f"{value['name']} · {counts['completed']}/{counts['required']} verified Runs\n"
            f"{counts['candidates']} candidates · {counts['pending']} pending · "
            f"{value['physical_cost']['spent_attempts']} spent Attempts · "
            f"{format_duration(value['physical_cost']['elapsed_seconds'])} historical duration\n"
            + ("Different code/runtime revisions · inference is separated in the HTML Code tabs.\n"
               if value.get("mixed_code") else "")
            +
            "Exploratory post-hoc extension. Original metrics, checkpoints and IDs are unchanged."
        )
        self.query_one("#series-plots", StudyOverviewDashboard).show_study(value)
        runs = self.query_one("#series-runs", DataTable)
        selected, x, y = runs.cursor_row, runs.scroll_x, runs.scroll_y
        runs.clear(columns=True)
        metric_names = sorted({key for row in value["runs"] for key in row["metrics"]})
        runs.add_columns(
            "Candidate",
            "Revision",
            "Seed",
            "State",
            *(metric_display_name(key) for key in metric_names),
            "Execution",
            "Run",
            "Attempt",
        )
        for row in value["runs"]:
            runs.add_row(
                str(row["presentation_trial"]),
                str(row["origin_revision"]),
                str(row["seed"]),
                row["status"],
                *(format_value(row["metrics"].get(key)) for key in metric_names),
                str(row.get("execution_id", "pending")),
                str(row.get("run_id", "pending")),
                str(row.get("attempt_id", "pending")),
                key=row["cell_id"],
            )
        if runs.row_count:
            runs.move_cursor(row=min(selected, runs.row_count - 1))
        self.call_after_refresh(runs.scroll_to, x=x, y=y, animate=False)
        revisions = self.query_one("#series-revisions", DataTable)
        revisions.clear(columns=True)
        revisions.add_columns(
            "Revision", "Code policy", "Native state", "Job", "Execution", "New cells"
        )
        origin = value["revisions"][0]
        revisions.add_row(
            "0 · origin",
            "original",
            "succeeded",
            str(origin.get("job_id") or "direct local"),
            origin["execution_id"],
            str(len(origin["evidence"])),
        )
        for op in value["operations"]:
            revisions.add_row(
                str(op["revision"]),
                value["revisions"][op["revision"]].get("code_compatibility", {}).get(
                    "kind", "verified"
                ),
                op["native_state"],
                str(op.get("latest_job_id") or "not submitted"),
                str(op.get("execution_id") or "not published"),
                str(len(value["revisions"][op["revision"]]["new_cells"])),
            )
        history = self.query_one("#series-history", DataTable)
        history.clear(columns=True)
        history.add_columns("When", "Revision", "Action", "State", "Job")
        for event in message.value["history"]["history"]:
            history.add_row(
                event["at_utc"],
                str(event["revision"]),
                event["action"],
                event["state"],
                str(event.get("job_id") or "—"),
            )

        def restore() -> None:
            for table in self.query(DataTable):
                row, x, y = positions.get(table.id, (0, 0, 0))
                if table.row_count:
                    table.move_cursor(row=min(row, table.row_count - 1))
                table.scroll_to(x=x, y=y, animate=False)

        self.call_after_refresh(restore)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if (
            event.data_table.id == "series-runs"
            and self.value
            and event.cursor_row < len(self.value["runs"])
        ):
            self.query_one("#series-selection", Static).update(
                structured_text(self.value["runs"][event.cursor_row], limit=25)
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "series-refresh":
            self._load()
        elif event.button.id == "series-extend":
            from lambdaforge.tui.screens.StudyExtension import StudyExtensionDialog

            self.app.push_screen(
                StudyExtensionDialog(self.work, self.services),
                lambda result: self._load() if result else None,
            )
        elif event.button.id in {"series-report", "series-export"}:
            from lambdaforge.tui.screens.Workspace import ExportDirectoryPicker

            export = event.button.id == "series-export"

            def selected(path: Path | None) -> None:
                if path:
                    self._start(
                        "export" if export else "report",
                        (lambda: self.services.study_series_export(self.series_id, path))
                        if export
                        else lambda: self.services.study_series_report(
                            self.series_id, path / (self.series_id + ".html")
                        ),
                    )

            self.app.push_screen(ExportDirectoryPicker(Path.cwd()), selected)
        elif event.button.id == "series-open-job" and self.value:
            row = self.query_one("#series-revisions", DataTable).cursor_row
            job_id = (
                self.value["revisions"][0].get("job_id")
                if row == 0
                else self.value["operations"][row - 1].get("latest_job_id")
            )
            if job_id:
                self._start("job", lambda: self.services.study_series_native_work(job_id))
            else:
                self.query_one("#series-status", Static).update(
                    "Direct local Execution · no provider Job. Its original evidence is in Results."
                )
