"""Metadata-only execution queue view; submission and cancellation stay native services."""

from __future__ import annotations

from threading import Thread
from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Label, Static

if TYPE_CHECKING:
    from lambdaforge.tui.services import ConsoleServices


class ExecutionQueueDialog(ModalScreen[None]):
    """On-demand queue metadata; ordinary entity views retain execution/log controls."""

    BINDINGS = [("escape", "back", "Back to Run Work")]

    def __init__(self, services: ConsoleServices, cluster: str) -> None:
        self.services, self.cluster = services, cluster
        self._busy = False
        super().__init__()

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-card wide-modal", id="execution-queue-dialog"):
            yield Label(f"Sequential execution queue · {self.cluster}", classes="modal-title")
            yield Static("Loading queue metadata…", id="execution-queue-status", markup=False)
            table: DataTable[Any] = DataTable(
                id="execution-queue-table", cursor_type="row", zebra_stripes=True
            )
            table.add_columns(
                "Order", "Work / Study", "Queue phase", "Last recorded Job state", "Job"
            )
            yield table
            yield Static(
                "Close this view safely. Cancel Jobs from their Work/Study controls. "
                "Preparing still needs this PC; queued/running Jobs are supervised on the cluster. "
                "Job states are the last recorded observation; Refresh reads queue metadata only."
            )
            with Horizontal(classes="modal-actions"):
                yield Button("Refresh", id="execution-queue-refresh")
                yield Button("Back to Run Work", id="execution-queue-close", variant="primary")

    def on_mount(self) -> None:
        self.call_after_refresh(self._refresh)

    def action_back(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "execution-queue-close":
            self.dismiss(None)
        elif event.button.id == "execution-queue-refresh":
            self._refresh()

    def _refresh(self) -> None:
        if self._busy:
            return
        self._busy = True
        self.query_one("#execution-queue-refresh", Button).disabled = True
        self.query_one("#execution-queue-status", Static).update("Refreshing queue metadata…")

        def read() -> None:
            try:
                result = self.services.execution_queue(self.cluster)
                error = None
            except Exception as failure:
                result, error = None, f"{type(failure).__name__}: {failure}"
            if self.is_mounted:
                self.app.call_from_thread(self._ready, result, error)

        Thread(target=read, daemon=True, name="lambdaforge-queue-view").start()

    def _ready(self, result: dict[str, Any] | None, error: str | None) -> None:
        if not self.is_mounted:
            return
        self._busy = False
        self.query_one("#execution-queue-refresh", Button).disabled = False
        status = self.query_one("#execution-queue-status", Static)
        if error:
            status.update(Text("Queue observation failed · " + error, style="red"))
            return
        table = self.query_one("#execution-queue-table", DataTable)
        table.clear()
        rows = (result or {}).get("entries", ())
        for index, row in enumerate(rows, 1):
            phase = {
                "pending": "Preparing · controller needed",
                "dispatched": "On cluster · FIFO admission",
                "external": "Earlier Job",
            }.get(row["phase"], row["phase"])
            table.add_row(
                str(index),
                str(row["name"]),
                str(phase),
                str(row["last_known_state"]),
                row["job_id"],
                key=row["job_id"],
            )
        status.update(
            f"{len(rows)} unsettled ticket(s) · FIFO · project/cluster scope"
            if rows
            else "No pending tickets in this project/cluster."
        )
