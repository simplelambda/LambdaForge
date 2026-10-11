"""Compact operational workbench using the same native observer as automation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from threading import Thread
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Input,
    Label,
    Static,
    TabbedContent,
    TabPane,
)


class OperationalWorkspace(Screen[None]):
    """Inspect compact health and configure the independent native project observer."""

    BINDINGS = [("escape", "back", "Back")]
    DEFAULT_CSS = """
    OperationalWorkspace { layout: vertical; padding: 1 2; }
    OperationalWorkspace > Horizontal { height: 3; min-height: 3; }
    OperationalWorkspace Static { height: auto; margin-bottom: 1; }
    OperationalWorkspace Label { height: auto; }
    OperationalWorkspace Button { width: auto; margin-right: 1; }
    OperationalWorkspace TabbedContent { height: 1fr; min-height: 10; }
    OperationalWorkspace DataTable { height: 1fr; }
    OperationalWorkspace VerticalScroll { height: 1fr; }
    OperationalWorkspace Input { height: 3; }
    """

    def __init__(self, services: Any) -> None:
        super().__init__()
        self.services = services
        self._busy = False
        self._loaded_settings = False

    def compose(self) -> ComposeResult:
        with Horizontal(classes="workspace-actions"):
            yield Button("← Back", id="operations-back", flat=True)
            yield Label("Operational health · project monitor and incidents")
        yield Static("Loading saved observations…", id="operations-status", markup=False)
        with Horizontal(classes="workspace-actions"):
            yield Button("Refresh providers", id="operations-refresh")
            yield Button("Start monitor", id="operations-start")
            yield Button("Stop monitor only", id="operations-stop")
        yield Static("", id="operations-monitor", markup=False, classes="detail-panel")
        with TabbedContent():
            with TabPane("Health"):
                yield DataTable(id="operations-health", cursor_type="row", zebra_stripes=True)
            with TabPane("Incidents"):
                yield DataTable(id="operations-incidents", cursor_type="row", zebra_stripes=True)
                yield Static(
                    "Select an incident to inspect its evidence and next action.",
                    id="operations-detail",
                    markup=False,
                    classes="detail-panel",
                )
            with TabPane("Monitor settings"):
                with VerticalScroll():
                    yield Static(
                        "Operational policy does not change scientific YAML. Progress expectations "
                        "are disabled by default: a Work without metrics is not presumed dead. "
                        "Detached monitoring survives terminal closure; reboot/logout persistence "
                        "requires an OS service (lf monitor install).",
                        markup=False,
                    )
                    for identifier, label, value in (
                        ("interval", "Polling interval, seconds (minimum 10)", "60"),
                        ("freshness", "Provider freshness, seconds", "180"),
                        ("heartbeat", "Heartbeat expectation, seconds", "45"),
                        ("grace", "Startup/heartbeat grace, seconds", "120"),
                        ("progress", "Progress expectation, seconds (blank disables)", ""),
                        (
                            "unreachable",
                            "Consecutive failed probes before unreachable (integer)",
                            "2",
                        ),
                    ):
                        yield Label(label)
                        yield Input(value=value, id=f"operations-{identifier}")
                    yield Checkbox("Enable email alerts (optional)", id="operations-smtp-enabled")
                    for identifier, label in (
                        ("host", "SMTP host"),
                        ("port", "SMTP port (default 587)"),
                        (
                            "cooldown_seconds",
                            "Email retry cooldown, seconds (minimum 30; default 900)",
                        ),
                        ("sender", "Sender email"),
                        ("recipients", "Recipient emails, comma-separated"),
                        ("username", "SMTP username (optional)"),
                        (
                            "credential",
                            "Credential reference: env:NAME or keyring:KEY — never a password",
                        ),
                    ):
                        yield Label(label)
                        yield Input(id=f"operations-smtp-{identifier}")
                    yield Checkbox("Implicit TLS (otherwise STARTTLS)", id="operations-smtp-ssl")
                    with Horizontal(classes="workspace-actions"):
                        yield Button("Save settings", id="operations-save", variant="primary")
                        yield Button("Send test email", id="operations-test")

    def on_mount(self) -> None:
        self.call_after_refresh(
            self._request, lambda: self.services.operational_snapshot(), "Reading saved health…"
        )

    def action_back(self) -> None:
        self.app.pop_screen()

    def _request(self, operation: Callable[[], Any], message: str) -> None:
        if self._busy:
            return
        self._busy = True
        self.query_one("#operations-status", Static).update(message)
        for button in self.query(Button):
            if button.id != "operations-back":
                button.disabled = True
        app = self.app

        def execute() -> None:
            try:
                outcome = operation()
                result = self.services.operational_snapshot()
            except Exception as error:
                if self.is_mounted:
                    app.call_from_thread(self._failed, error)
            else:
                if self.is_mounted:
                    app.call_from_thread(self._ready, result, outcome)

        Thread(target=execute, daemon=True, name="lambdaforge-operational-workbench").start()

    def _finished(self) -> None:
        self._busy = False
        for button in self.query(Button):
            button.disabled = False

    def _failed(self, error: Exception) -> None:
        if not self.is_mounted:
            return
        self._finished()
        from lambdaforge.controlplane.SecretRedactor import SecretRedactor

        self.query_one("#operations-status", Static).update(
            f"Operation failed · {type(error).__name__}: {SecretRedactor.redact(error)[:1000]}"
        )

    def _ready(self, snapshot: Mapping[str, Any], outcome: Any = None) -> None:
        if not self.is_mounted:
            return
        self._finished()
        self.query_one("#operations-status", Static).update(
            "Test email accepted by SMTP · scientific Jobs unchanged."
            if isinstance(outcome, Mapping) and outcome.get("accepted_by_smtp")
            else "Operation complete · saved snapshot; provider age is shown per Job."
        )
        monitor = snapshot["monitor"]
        self.query_one("#operations-monitor", Static).update(
            f"Monitor: {monitor['health']} · process: {monitor['process_lifetime']} · "
            f"heartbeat age: {monitor.get('heartbeat_age_seconds')}s\n"
            "Stopping this monitor does not cancel scientific Jobs. Inspect: lf monitor diagnose"
        )
        health = self.query_one("#operations-health", DataTable)
        health.clear(columns=True)
        health.add_columns("Job", "Cluster", "Logical state", "Health", "Last confirmed", "Age (s)")
        for job in snapshot["jobs"]:
            facts = job["operational"]
            age = facts["observation_age_seconds"]
            health.add_row(
                job["job_id"],
                job["cluster"],
                job["state"],
                facts["health"],
                facts.get("last_confirmed_state") or "unverified",
                str(round(age)) if age is not None else "unknown",
                key=job["job_id"],
            )
        self._incidents = list(snapshot["incidents"])
        incidents = self.query_one("#operations-incidents", DataTable)
        incidents.clear(columns=True)
        incidents.add_columns("Job", "Severity", "Status", "Incident", "Last observed")
        for incident in self._incidents:
            incidents.add_row(
                incident["job_id"],
                incident["severity"],
                incident["status"],
                incident["code"],
                incident["last_seen_at_utc"],
                key=incident["incident_id"],
            )
        if not self._loaded_settings:
            settings = snapshot["settings"]
            policy = settings.get("observation", {})
            for identifier, field, default in (
                ("interval", "interval_seconds", 60),
                ("freshness", "freshness_seconds", 180),
                ("heartbeat", "heartbeat_seconds", 45),
                ("grace", "grace_seconds", 120),
                ("progress", "progress_seconds", ""),
                ("unreachable", "unreachable_failures", 2),
            ):
                value = (
                    settings.get(field, default)
                    if identifier == "interval"
                    else policy.get(field, default)
                )
                self.query_one(f"#operations-{identifier}", Input).value = (
                    "" if value is None else str(value)
                )
            smtp = settings.get("smtp", {})
            for field in (
                "host",
                "port",
                "sender",
                "recipients",
                "username",
                "credential",
                "cooldown_seconds",
            ):
                value = smtp.get(field, "")
                self.query_one(f"#operations-smtp-{field}", Input).value = (
                    ", ".join(value) if isinstance(value, list) else str(value)
                )
            self.query_one("#operations-smtp-enabled", Checkbox).value = bool(smtp.get("enabled"))
            self.query_one("#operations-smtp-ssl", Checkbox).value = smtp.get("tls") == "ssl"
            self._loaded_settings = True

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id == "operations-incidents" and event.cursor_row < len(
            self._incidents
        ):
            incident = self._incidents[event.cursor_row]
            self.query_one("#operations-detail", Static).update(
                f"{incident['summary']}\nFirst seen: {incident['first_seen_at_utc']} · "
                f"occurrences: {incident['occurrences']}\n{incident['recommendation']}\n"
                f"Inspect: {incident['diagnostic_command']}\n"
                f"Notification: {incident.get('notification', {}).get('status', 'disabled')}"
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        identifier = event.button.id
        if identifier == "operations-back":
            self.action_back()
        elif identifier == "operations-refresh":
            self._request(
                lambda: self.services.operational_snapshot(refresh=True), "Checking providers…"
            )
        elif identifier in {"operations-start", "operations-stop", "operations-test"}:
            operation = {
                "operations-start": "start",
                "operations-stop": "stop",
                "operations-test": "test-alert",
            }[identifier]
            self._request(lambda: self.services.monitor_action(operation), f"Monitor {operation}…")
        elif identifier == "operations-save":
            try:

                def value(field: str) -> str:
                    return self.query_one(f"#operations-{field}", Input).value.strip()

                progress = value("progress")
                settings = {
                    "interval_seconds": float(value("interval")),
                    "observation": {
                        "freshness_seconds": float(value("freshness")),
                        "heartbeat_seconds": float(value("heartbeat")),
                        "grace_seconds": float(value("grace")),
                        "unreachable_failures": int(value("unreachable")),
                        "progress_seconds": float(progress) if progress else None,
                    },
                    "smtp": {
                        "enabled": self.query_one("#operations-smtp-enabled", Checkbox).value,
                        "tls": "ssl"
                        if self.query_one("#operations-smtp-ssl", Checkbox).value
                        else "starttls",
                        **{
                            field: value(f"smtp-{field}")
                            for field in ("host", "sender", "username", "credential")
                        },
                        "port": int(value("smtp-port") or "587"),
                        "cooldown_seconds": float(value("smtp-cooldown_seconds") or "900"),
                        "recipients": [
                            item.strip()
                            for item in value("smtp-recipients").split(",")
                            if item.strip()
                        ],
                    },
                }
            except Exception as error:
                self._failed(error)
                return
            self._request(
                lambda: self.services.monitor_action("config", settings),
                "Saving validated monitor settings…",
            )
