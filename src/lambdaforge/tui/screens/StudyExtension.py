"""Nonblocking, explicitly reviewed Study extension dialog over native domain services."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from threading import Thread
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    LoadingIndicator,
    Static,
    TabbedContent,
    TabPane,
)

from lambdaforge.controlplane.SecretRedactor import SecretRedactor
from lambdaforge.scientific_format import format_parameter_value


class ExtensionReply(Message):
    def __init__(self, phase: str, value: Any = None, error: BaseException | None = None) -> None:
        super().__init__()
        self.phase, self.value, self.error = phase, value, error


class StudyExtensionDialog(ModalScreen[dict[str, Any] | None]):
    """Choose the original edited YAML, review the exact delta, then submit once."""

    BINDINGS = [("escape", "cancel", "Back")]
    DEFAULT_CSS = """
    StudyExtensionDialog { align: center middle; background: rgba(0, 0, 0, 0.65); }
    #extension-dialog {
        width: 96; max-width: 95%; height: 88%; padding: 1 2;
        background: $surface; border: round $accent;
    }
    #extension-body { height: 1fr; }
    #extension-body Static { height: auto; margin-bottom: 1; }
    #extension-loading { height: 2; }
    #extension-path { height: 3; }
    #extension-controls { height: 3; }
    #extension-status { height: auto; min-height: 2; }
    #extension-controls Button { width: auto; min-width: 12; margin-right: 1; }
    #extension-evidence { height: 15; }
    #extension-evidence DataTable { height: 1fr; }
    """

    def __init__(self, work: Mapping[str, Any], services: Any) -> None:
        super().__init__()
        self.work, self.services = dict(work), services
        self.context: dict[str, Any] | None = None
        self.plan: dict[str, Any] | None = None
        self.busy: str | None = None
        self._reviewed_path: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="extension-dialog"):
            yield Label("Extend Study · keep completed Runs", classes="modal-title")
            with VerticalScroll(id="extension-body"):
                yield Static(
                    "Edit the original YAML outside LF, adding sweep values OR seeds. "
                    "Choose that same file here. Review preserves completed evidence; "
                    "Extend submits only new combinations. This is not Resume.",
                    markup=False,
                )
                yield Input(
                    placeholder="Original YAML (filled from the exact Study)", id="extension-path"
                )
                yield Button("Browse YAML…", id="extension-browse")
                yield LoadingIndicator(id="extension-loading")
                yield Static(
                    "No extension has been reviewed or submitted.",
                    id="extension-plan",
                    markup=False,
                )
                with TabbedContent(id="extension-evidence"):
                    with TabPane("Retained Runs"):
                        yield DataTable(id="extension-retained", cursor_type="row")
                    with TabPane("New Runs"):
                        yield DataTable(id="extension-new", cursor_type="row")
            yield Static("Resolving the exact Study…", id="extension-status", markup=False)
            with Horizontal(id="extension-controls"):
                yield Button("Back", id="extension-cancel")
                yield Button(
                    "Review changes", id="extension-review", variant="primary", disabled=True
                )
                yield Button("Extend Study", id="extension-apply", variant="success", disabled=True)

    def on_mount(self) -> None:
        self.query_one("#extension-evidence").display = False
        self._start("context", lambda: self.services.extension_context(self.work))

    def _start(self, phase: str, operation: Callable[[], Any]) -> None:
        if self.busy is not None:
            return
        self.busy = phase
        self.query_one("#extension-loading").display = True
        self.query_one("#extension-review", Button).disabled = True
        self.query_one("#extension-apply", Button).disabled = True
        self.query_one("#extension-path", Input).disabled = True
        self.query_one("#extension-browse", Button).disabled = True
        self.query_one("#extension-status", Static).update(
            {
                "context": "Resolving the exact Study…",
                "preview": "Checking identities and new combinations… nothing submitted.",
                "apply": "Submitting the reviewed delta… do not submit a duplicate.",
            }[phase]
        )

        def execute() -> None:
            try:
                value = operation()
            except (Exception, SystemExit) as error:
                self.post_message(ExtensionReply(phase, error=error))
            else:
                self.post_message(ExtensionReply(phase, value=value))

        Thread(target=execute, daemon=True, name="lambdaforge-study-extension-" + phase).start()

    def on_extension_reply(self, message: ExtensionReply) -> None:
        message.stop()
        if not self.is_mounted:
            return
        self.busy = None
        self.query_one("#extension-loading").display = False
        self.query_one("#extension-path", Input).disabled = False
        self.query_one("#extension-browse", Button).disabled = False
        self.query_one("#extension-review", Button).disabled = self.context is None
        if message.error is not None:
            self.plan = None
            self.query_one("#extension-status", Static).update(
                f"Extension blocked · {type(message.error).__name__}: "
                f"{SecretRedactor.redact(message.error)[:2000]}"
            )
            return
        if message.phase == "context":
            self.context = message.value
            self.query_one("#extension-path", Input).value = self.context["config"]
            self.query_one("#extension-review", Button).disabled = False
            self.query_one("#extension-status", Static).update(
                "Select the edited original YAML and review its changes."
            )
        elif message.phase == "preview":
            self.query_one("#extension-evidence").display = True
            self.plan = message.value
            counts = self.plan["comparison"]["counts"]
            changes = self.plan["comparison"]
            added = "\n".join(
                f"{name}: add {format_parameter_value(values)}"
                for name, values in changes["added_values"].items()
            )
            if changes["added_seeds"]:
                added += "\nAdd seeds: " + format_parameter_value(changes["added_seeds"])
            if not added and counts["new"]:
                added = "New exact combinations · inspect the New Runs table below."
            target = self.plan.get("target") or (self.context or {}).get("target", "—")
            attempts = self.plan.get("budget", {}).get("historical_attempts", "—")
            self.query_one("#extension-plan", Static).update(
                f"KEEP {counts['reused']} completed Runs\n"
                f"EXECUTE {counts['new']} new Runs · {counts['total']} total cells\n\n"
                f"Target: {target} · "
                f"revision {self.plan.get('expected_revision', 0) + 1}\n"
                f"Historical Attempts: {attempts}\n"
                f"Code: {self.plan.get('code_compatibility', {}).get('kind', 'verified')}\n"
                f"{self.plan.get('code_compatibility', {}).get('warning', '')}\n"
                f"{added or 'No new evidence requested.'}\n\n"
                "Original metrics, checkpoints and artifacts remain owned by their original "
                "Runs. The delta receives its own native Job. Post-hoc extension is exploratory."
            )
            for identifier, rows in (
                ("extension-retained", self.plan.get("reused_evidence", [])),
                ("extension-new", changes.get("new_cells", [])),
            ):
                table = self.query_one("#" + identifier, DataTable)
                table.clear(columns=True)
                table.add_columns("Seed", "Parameters", "Origin Run", "Attempt")
                for row in rows:
                    table.add_row(
                        str(row["seed"]),
                        format_parameter_value(row["parameters"]),
                        str(row.get("run_id", "new")),
                        str(row.get("attempt_id", "new")),
                        key=row["cell_id"],
                    )
            supported = bool(self.plan["apply_supported"]) and counts["new"] > 0
            self.query_one("#extension-apply", Button).disabled = not supported
            self.query_one("#extension-apply", Button).label = (
                "Extend · current code"
                if self.plan.get("code_compatibility", {}).get("kind") == "revisioned-current"
                else "Extend Study"
            )
            self.query_one("#extension-status", Static).update(
                "Reviewed · confirmation required; no computation started."
                if supported
                else "Nothing to submit, or this placement is not supported."
            )
        else:
            self.dismiss(message.value)

    def on_input_changed(self, event: Input.Changed) -> None:
        if (
            event.input.id == "extension-path"
            and self.plan is not None
            and event.value.strip() != self._reviewed_path
        ):
            self.plan = None
            self.query_one("#extension-apply", Button).disabled = True
            self.query_one("#extension-status", Static).update(
                "YAML selection changed · review again before extending."
            )

    def action_cancel(self) -> None:
        if self.busy == "apply":
            self.query_one("#extension-status", Static).update(
                "Native submission is in progress; waiting for its ownership receipt."
            )
            return
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "extension-cancel":
            self.action_cancel()
        elif self.busy is not None:
            return
        elif event.button.id == "extension-browse":
            from lambdaforge.tui.App import YamlFilePicker

            self.app.push_screen(YamlFilePicker(Path.cwd()), self._picked)
        elif event.button.id == "extension-review" and self.context is not None:
            context = dict(self.context)
            path = self.query_one("#extension-path", Input).value.strip()
            self.plan = None
            self._reviewed_path = path
            self._start("preview", lambda: self.services.extension_preview(context, path))
        elif (
            event.button.id == "extension-apply"
            and self.plan is not None
            and self.context is not None
        ):
            context, plan = dict(self.context), dict(self.plan)
            path = self.query_one("#extension-path", Input).value.strip()
            if path != self._reviewed_path:
                self.plan = None
                return
            self._start(
                "apply",
                lambda: self.services.extend_study(context, path, plan_hash=plan["plan_hash"]),
            )

    def _picked(self, path: Path | None) -> None:
        if path is not None:
            self.query_one("#extension-path", Input).value = str(path)
