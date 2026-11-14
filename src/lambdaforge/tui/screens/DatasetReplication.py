"""Choose native replication endpoints; no dataset or transport logic in widgets."""

from collections.abc import Sequence

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static


class ReplicationPassword(ModalScreen[str | None]):
    """Explicit one-use credential delivery, never a saved cluster credential."""

    DEFAULT_CSS = """
    ReplicationPassword { align: center middle; }
    ReplicationPassword > Vertical { width: 78; max-width: 95%; height: auto;
        padding: 1 2; border: round $warning; background: $surface; }
    ReplicationPassword Horizontal { height: 3; align-horizontal: right; }
    """
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, source: str, destination: str) -> None:
        super().__init__()
        self.source, self.destination = source, destination

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"One-use destination password · {self.destination}", classes="modal-title")
            yield Static(
                f"Authorize {self.source} → {self.destination}. "
                "The password is sent over encrypted "
                f"SSH to {self.source}, kept only in memory and used to log in to the destination. "
                "It is not saved. Host fingerprints must already be trusted. After Job "
                "acknowledgement the copy needs no laptop. This only checks access; copying "
                "still requires the next confirmation."
            )
            yield Input(
                password=True, placeholder="Destination account password", id="replication-password"
            )
            yield Static("", id="replication-password-status")
            with Horizontal():
                yield Button("Cancel", id="password-cancel")
                yield Button("Check direct access", id="password-accept", variant="primary")

    def action_cancel(self) -> None:
        self.query_one(Input).value = ""
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "password-cancel":
            self.action_cancel()
        elif event.button.id == "password-accept":
            self._accept_password()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._accept_password()

    def _accept_password(self) -> None:
        field = self.query_one(Input)
        secret, field.value = field.value, ""
        if secret:
            self.dismiss(secret)
        else:
            self.query_one("#replication-password-status", Static).update(
                "Enter a destination password, or Cancel."
            )


class DatasetReplication(ModalScreen[tuple[str, str, str] | None]):
    """Endpoint selection followed by the workspace's exact preview/confirmation."""

    DEFAULT_CSS = """
    DatasetReplication { align: center middle; }
    DatasetReplication > Vertical { width: 76; max-width: 95%; height: 90%; max-height: 34;
        padding: 1 2; border: round $primary; background: $surface; }
    DatasetReplication .replication-fields { height: 1fr; }
    DatasetReplication Select { margin-bottom: 1; }
    DatasetReplication .modal-actions { height: 3; align-horizontal: right; }
    """
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, selector: str, sources: Sequence[str], targets: Sequence[str]) -> None:
        super().__init__()
        self.selector, self.sources, self.targets = selector, sources, targets

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"Replicate {self.selector}", classes="modal-title")
            with VerticalScroll(classes="replication-fields"):
                yield Static(
                    "Copy exact published bytes with compression. Preview checks both endpoints "
                    "and SSH push/pull without starting the copy."
                )
                yield Label("Source placement")
                yield Select(
                    [(name, name) for name in self.sources],
                    value=self.sources[0],
                    allow_blank=False,
                    id="replicate-source",
                )
                yield Label("Destination cluster")
                yield Select(
                    [(name, name) for name in self.targets],
                    prompt="Choose destination",
                    id="replicate-destination",
                )
                yield Label("Transfer route · execution host is chosen from this policy")
                yield Select(
                    [
                        ("Direct only · no laptop required", "direct"),
                        ("Direct · ask destination password, never save it", "password"),
                        ("Automatic · relay needs explicit confirmation", "auto"),
                        ("Relay via this PC · must stay powered/connected", "relay"),
                    ],
                    value="direct" if self.sources[0] != "local" else "auto",
                    allow_blank=False,
                    id="replicate-route",
                )
                yield Static("", id="replicate-route-help", markup=False)
                yield Static("", id="replicate-choice-status")
            with Horizontal(classes="modal-actions"):
                yield Button("Cancel", id="replicate-cancel")
                yield Button("Preview replication", id="replicate-preview", variant="primary")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "replicate-route":
            return
        self.query_one("#replicate-route-help", Static).update(
            {
                "direct": "Runs on source or destination, surviving laptop shutdown. "
                "Requires authorized SSH between clusters. If neither direction works, "
                "preview explains the failure; no relay is started.",
                "password": "Enter the destination password once. Source-host Job uses it "
                "only in memory; no password file, key forwarding or laptop relay.",
                "auto": "Tries direct push then pull. If neither works, preview proposes "
                "relay via this PC; you can cancel instead of accepting it.",
                "relay": "Bytes pass through this PC. Closing LF is safe, but suspending, "
                "shutting down or disconnecting the PC can interrupt the copy.",
            }.get(str(event.value), "")
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "replicate-cancel":
            self.dismiss(None)
        elif event.button.id == "replicate-preview":
            source = self.query_one("#replicate-source", Select).value
            target = self.query_one("#replicate-destination", Select).value
            if target == Select.BLANK or source == target:
                self.query_one("#replicate-choice-status", Static).update(
                    "Choose a different destination cluster."
                )
            else:
                route = self.query_one("#replicate-route", Select).value
                if route in {"direct", "password"} and "local" in {source, target}:
                    self.query_one("#replicate-choice-status", Static).update(
                        "A local endpoint requires this PC. Choose Automatic or Relay, "
                        "or select two remote clusters for Direct only."
                    )
                    return
                self.dismiss((str(source), str(target), str(route)))
