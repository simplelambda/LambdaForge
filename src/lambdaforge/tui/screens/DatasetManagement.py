"""Explicit identity/placement choice; all mutations remain native service operations."""

from collections.abc import Sequence

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Select, Static


class DatasetManagement(ModalScreen[tuple[str, str] | None]):
    """Choose an exact dataset placement and preview its native management operation."""

    DEFAULT_CSS = """
    DatasetManagement { align: center middle; }
    DatasetManagement > Vertical { width: 76; max-width: 95%; height: 90%; max-height: 32;
        padding: 1 2; border: round $primary; background: $surface; }
    DatasetManagement .management-fields { height: 1fr; }
    DatasetManagement Select { margin-bottom: 1; }
    DatasetManagement .modal-actions { height: 3; align-horizontal: right; }
    """
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, selector: str, identity: str, targets: Sequence[str]) -> None:
        super().__init__()
        self.selector, self.identity, self.targets = selector, identity, targets

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"Manage copies · {self.selector}", classes="modal-title")
            with VerticalScroll(classes="management-fields"):
                yield Static(f"Selected exact content\n{self.identity}", markup=False)
                yield Static(
                    "Choose the cluster registering this identity. Other identities and old "
                    "Run evidence remain untouched. Every action has a read-only preview "
                    "and requires confirmation."
                )
                yield Label("Target cluster")
                yield Select(
                    [(name, name) for name in self.targets],
                    value=self.targets[0],
                    allow_blank=False,
                    id="manage-cluster",
                )
                yield Label("Operation")
                yield Select(
                    [
                        ("Keep this identity as project reference", "adopt"),
                        ("Remove registration · keep files", "remove"),
                        ("Delete this managed copy · delete files", "delete"),
                    ],
                    value="adopt",
                    allow_blank=False,
                    id="manage-operation",
                )
                yield Static(
                    "Keep as reference verifies every checksum and selects future resolution; "
                    "it does not erase other copies or certify scientific equivalence.\n"
                    "Remove registration archives the index, keeping bytes (no disk space freed).\n"
                    "Delete copy also removes exact owned files; active consumers block removal."
                )
            with Horizontal(classes="modal-actions"):
                yield Button("Cancel", id="manage-cancel")
                yield Button("Preview action", id="manage-preview", variant="primary")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "manage-cancel":
            self.dismiss(None)
        elif event.button.id == "manage-preview":
            self.dismiss(
                (
                    str(self.query_one("#manage-operation", Select).value),
                    str(self.query_one("#manage-cluster", Select).value),
                )
            )
