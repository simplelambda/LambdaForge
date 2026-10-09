"""Choose native replication endpoints; no dataset or transport logic in widgets."""

from collections.abc import Sequence

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Select, Static


class DatasetReplication(ModalScreen[tuple[str, str] | None]):
    """Endpoint selection followed by the workspace's exact preview/confirmation."""

    DEFAULT_CSS = """
    DatasetReplication { align: center middle; }
    DatasetReplication > Vertical { width: 76; height: auto; padding: 1 2;
        border: round $primary; background: $surface; }
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
            yield Static(
                "Copy exact published bytes, not a reconstruction. A read-only preview "
                "checks the destination before confirmation. Transfer is compressed; "
                "direct site SSH is preferred, with an authenticated streaming fallback."
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
            yield Static("", id="replicate-choice-status")
            with Horizontal(classes="modal-actions"):
                yield Button("Cancel", id="replicate-cancel")
                yield Button("Preview replication", id="replicate-preview", variant="primary")

    def action_cancel(self) -> None:
        self.dismiss(None)

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
                self.dismiss((str(source), str(target)))
