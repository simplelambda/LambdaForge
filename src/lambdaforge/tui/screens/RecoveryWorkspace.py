"""Saved recovery audit and persistent action errors over native Console services."""

# Human-facing explanations are deliberately complete strings.
# ruff: noqa: E501

from __future__ import annotations

from collections.abc import Mapping
from threading import Thread
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, DataTable, Label, Static

from lambdaforge.controlplane.SecretRedactor import SecretRedactor
from lambdaforge.tui.viewmodels import confirmation_text


class RecoveryAuditLoaded(Message):
    """Deliver a saved native audit to the mounted screen's event loop."""

    def __init__(self, audit: Mapping[str, Any]) -> None:
        super().__init__()
        self.audit = audit


class RecoveryAuditFailed(Message):
    """Deliver an audit read failure without calling an unmounted screen."""

    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error


class RecoveryWorkspace(Screen[None]):
    """Explicit metadata-only drill-down: no provider poll or hidden artifact retrieval."""

    BINDINGS = [("escape", "back", "Back")]
    DEFAULT_CSS = """
    RecoveryWorkspace { layout: vertical; padding: 1 2; }
    RecoveryWorkspace Horizontal { height: 3; min-height: 3; }
    RecoveryWorkspace Button { width: auto; margin-right: 1; }
    RecoveryWorkspace Static { height: auto; }
    #recovery-jobs { height: 8; min-height: 4; }
    #recovery-events { height: 1fr; min-height: 6; }
    #recovery-event-detail { height: 9; min-height: 3; }
    """

    def __init__(self, services: Any, job_id: str) -> None:
        super().__init__()
        self.services = services
        self.job_id = job_id
        self.audit_offset = 0
        self.next_offset: int | None = None
        self._busy = False
        self._jobs: list[Mapping[str, Any]] = []
        self._events: list[Mapping[str, Any]] = []

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Button("← Back", id="recovery-back", flat=True)
            yield Label("Recovery history · native saved evidence")
        yield Static("Loading saved recovery audit…", id="recovery-status", markup=False)
        yield DataTable(id="recovery-jobs", cursor_type="row", zebra_stripes=True)
        with Horizontal():
            yield Button("Refresh saved audit", id="recovery-refresh")
            yield Button("Newer events", id="recovery-newer", disabled=True)
            yield Button("Older events", id="recovery-older", disabled=True)
        yield DataTable(id="recovery-events", cursor_type="row", zebra_stripes=True)
        with VerticalScroll(id="recovery-event-detail"):
            yield Static(
                "Select an event for its evidence and diagnostic command.",
                id="recovery-detail",
                markup=False,
            )

    def on_mount(self) -> None:
        self.call_after_refresh(self._load)

    def action_back(self) -> None:
        self.app.pop_screen()

    def _load(self) -> None:
        if self._busy:
            return
        self._busy = True
        job_id, offset = self.job_id, self.audit_offset
        self.query_one("#recovery-status", Static).update(f"Reading saved audit for {job_id}…")
        for button in self.query(Button):
            if button.id != "recovery-back":
                button.disabled = True

        def read() -> None:
            try:
                audit = self.services.recovery_history(job_id, offset=offset, limit=100)
            except Exception as error:
                self.post_message(RecoveryAuditFailed(error))
            else:
                self.post_message(RecoveryAuditLoaded(audit))

        Thread(target=read, daemon=True, name="lambdaforge-recovery-history").start()

    def on_recovery_audit_loaded(self, event: RecoveryAuditLoaded) -> None:
        self._ready(event.audit)

    def on_recovery_audit_failed(self, event: RecoveryAuditFailed) -> None:
        self._failed(event.error)

    def _failed(self, error: Exception) -> None:
        if not self.is_mounted:
            return
        self._busy = False
        self.query_one("#recovery-refresh", Button).disabled = False
        self.query_one("#recovery-status", Static).update(
            f"Audit read failed · {SecretRedactor.redact(error)[:1000]} · previous evidence retained."
        )

    def _ready(self, audit: Mapping[str, Any]) -> None:
        if not self.is_mounted:
            return
        self._jobs = list(audit["jobs"])
        jobs = self.query_one("#recovery-jobs", DataTable)
        jobs.clear(columns=True)
        jobs.add_columns("Job", "State", "Recovery phase", "Request", "Operational health")
        for job in self._jobs:
            recovery = job.get("recovery") or {}
            jobs.add_row(
                str(job["job_id"]),
                str(job["state"]),
                str(recovery.get("phase", "origin")),
                str(recovery.get("request_id", "—")),
                str(job.get("operational", {}).get("health", "unverified")),
                key=str(job["job_id"]),
            )
        page = audit["events"]
        self._events = list(page["items"])
        events = self.query_one("#recovery-events", DataTable)
        events.clear(columns=True)
        events.add_columns("Time (UTC)", "Phase", "State", "Request", "Evidence")
        for index, event in enumerate(self._events):
            events.add_row(
                str(event.get("timestamp_utc", "—")),
                str(event.get("phase") or event.get("source") or "lifecycle"),
                str(event.get("state", "unknown")),
                str(event.get("recovery_request_id", "—")),
                str(event.get("message", "")),
                key=str(index),
            )
        self.next_offset = page["next_offset"]
        self._busy = False
        self.query_one("#recovery-refresh", Button).disabled = False
        self.query_one("#recovery-newer", Button).disabled = self.audit_offset == 0
        self.query_one("#recovery-older", Button).disabled = self.next_offset is None
        self.query_one("#recovery-status", Static).update(
            f"{self.job_id} · {len(self._events)} of {page['total']} saved events · newest first. "
            "Select a linked Job with Enter to inspect its events. Provider not queried; acceptance is not scientific progress."
        )
        if audit.get("missing_dependencies"):
            self.query_one("#recovery-detail", Static).update(
                "Some referenced owner Jobs are not in the local catalog; missing references are not reconstructed."
            )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "recovery-jobs" and not self._busy:
            selected = str(event.row_key.value)
            if selected != self.job_id:
                self.job_id, self.audit_offset = selected, 0
                self._load()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id == "recovery-events" and event.cursor_row < len(self._events):
            value = self._events[event.cursor_row]
            self.query_one("#recovery-detail", Static).update(
                f"{value.get('phase') or value.get('source')} · {value.get('timestamp_utc')}\n"
                f"{value.get('message', '')}\n"
                + confirmation_text(value.get("details", {}))
                + f"\nInspect saved history: lf jobs recovery {self.job_id} --json\n"
                + f"Inspect logs (may need provider access): lf jobs logs {self.job_id} --tail 300"
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "recovery-back":
            self.action_back()
        elif not self._busy:
            if event.button.id == "recovery-newer":
                self.audit_offset = max(0, self.audit_offset - 100)
            elif event.button.id == "recovery-older" and self.next_offset is not None:
                self.audit_offset = self.next_offset
            else:
                self.audit_offset = 0
            self._load()


class RecoveryFailureDialog(ModalScreen[str | None]):
    """Critical action failure stays visible until acknowledged; no polling toast."""

    BINDINGS = [("escape", "close", "Return")]

    def __init__(self, error: Exception, *, job_id: str, phase: str, has_plan: bool) -> None:
        super().__init__()
        self.error = error
        self.job_id = str(getattr(error, "recovery_job_id", job_id))
        self.phase = phase
        self.has_plan = has_plan

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-card wide-modal"):
            yield Label("Study recovery did not complete", classes="modal-title")
            with VerticalScroll():
                request = getattr(
                    self.error, "recovery_request_id", "No apply request; preview only"
                )
                yield Static(
                    f"Phase: {self.phase}\nJob: {self.job_id}\nRequest: {request}\n\n"
                    f"{type(self.error).__name__}: {SecretRedactor.redact(self.error)[:2000]}\n\n"
                    "No new execution admission has been confirmed by this action. Inspect the saved "
                    "history and recheck ownership before repeating recovery. Scientific evidence is not reset.\n"
                    f"Saved audit: lf jobs recovery {self.job_id} --json",
                    id="recovery-failure-detail",
                    markup=False,
                )
                if getattr(self.error, "recovery_audit_error", None):
                    yield Static(
                        f"Audit persistence also failed: {getattr(self.error, 'recovery_audit_error', '')}",
                        markup=False,
                    )
            with Horizontal(classes="modal-actions"):
                yield Button("Return", id="recovery-failure-close")
                yield Button("Logs", id="recovery-failure-logs")
                yield Button("Saved history", id="recovery-failure-history")
                yield Button("Back to plan", id="recovery-failure-plan", disabled=not self.has_plan)

    def action_close(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(
            {
                "recovery-failure-logs": "logs",
                "recovery-failure-history": "history",
                "recovery-failure-plan": "plan",
            }.get(event.button.id or "")
        )
