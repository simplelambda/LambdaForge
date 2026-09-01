"""Full-screen Textual Research Console launched by bare ``lf``."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from threading import Thread
from typing import Any, cast

from textual.app import App, ComposeResult
from textual.command import Hit, Hits, Provider
from textual.containers import Horizontal, Vertical
from textual.events import Resize
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Label, Select, Static

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.GpuAccessPolicy import GpuAccessPolicy
from lambdaforge.tui.models import CONSOLE_ACTIONS, ConsoleAction
from lambdaforge.tui.screens.Base import DataScreen
from lambdaforge.tui.screens.ClusterScreen import ClusterScreen
from lambdaforge.tui.screens.DatasetScreen import DatasetScreen
from lambdaforge.tui.screens.OverviewScreen import OverviewScreen
from lambdaforge.tui.screens.ResultsScreen import ResultsScreen
from lambdaforge.tui.screens.StudyScreen import StudyScreen
from lambdaforge.tui.screens.WorkScreen import WorkScreen
from lambdaforge.tui.services import ConsoleServices


class ActionProvider(Provider):
    """Expose the explicit CLI/TUI parity inventory through fuzzy search."""

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for action in CONSOLE_ACTIONS:
            label = f"{action.name} · {action.operation}"
            score = matcher.match(label)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(label),
                    partial(cast(Any, self.app).open_console_action, action),
                    help=f"Open {action.screen}; direct domain service, never a CLI subprocess.",
                )


class ConfirmationDialog(ModalScreen[bool]):
    """Visible confirmation boundary for destructive service actions."""

    def __init__(self, title: str, impact: str) -> None:
        super().__init__()
        self.title_text = title
        self.impact = impact

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog", classes="modal-card"):
            yield Label(self.title_text, classes="modal-title")
            yield Static(self.impact)
            with Horizontal(classes="modal-actions"):
                yield Button("Cancel", id="cancel", variant="default")
                yield Button("Confirm", id="confirm", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm")


class ContextHelp(ModalScreen[None]):
    """Explain common navigation and semantic colour without leaving the console."""

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-card"):
            yield Label("Contextual help", classes="modal-title")
            yield Static(
                "Enter opens the selected item. Esc goes back. Ctrl+P searches every action.\n\n"
                "Blue means active, green healthy, yellow incomplete/censored, red failure or "
                "destructive. Interactive actions call Python domain services directly."
            )
            yield Button("Close", id="close")

    def on_button_pressed(self) -> None:
        self.dismiss(None)


class ClusterEditor(ModalScreen[bool]):
    """Human-first cluster setup with contextual explanations and safe defaults."""

    HELP_TEXT = {
        "cluster-name": "A short stable name used by --on and dataset placement selectors.",
        "cluster-host": "SSH hostname or IP. LambdaForge never stores a password in this profile.",
        "cluster-user": "Remote account name. Authentication remains in OpenSSH/keyring.",
        "cluster-workspace": "Absolute remote root for owned jobs, cache and managed environments.",
        "cluster-project": (
            "Optional remote mirror of this project for large inputs and published outputs."
        ),
        "cluster-scheduler": (
            "Direct runs a detached supervisor after SSH. SLURM submits through the site scheduler."
        ),
    }

    def __init__(self, services: ConsoleServices) -> None:
        super().__init__()
        self.services = services

    def compose(self) -> ComposeResult:
        with Vertical(id="cluster-editor", classes="modal-card wide-modal"):
            yield Label("Add cluster", classes="modal-title")
            yield Static("Common settings", classes="section-title")
            yield Input(placeholder="Cluster name", id="cluster-name")
            yield Input(placeholder="Host or IP", id="cluster-host")
            yield Input(placeholder="Remote user", id="cluster-user")
            yield Select(
                (("Direct after SSH login", "local"), ("Through SLURM", "slurm")),
                value="local",
                id="cluster-scheduler",
            )
            yield Input(placeholder="/home/user/.lambdaforge", id="cluster-workspace")
            yield Input(placeholder="Optional /home/user/project mirror", id="cluster-project")
            yield Static(self.HELP_TEXT["cluster-name"], id="cluster-help", classes="help-panel")
            yield Static(
                "Passwords are never written to YAML. Advanced SSH, CUDA, roots, wheelhouse "
                "and GPU command/claim settings can be edited after save from the cluster "
                "detail panel.",
                classes="muted",
            )
            with Horizontal(classes="modal-actions"):
                yield Button("Exit without saving", id="cancel")
                yield Button("Save profile", id="save", variant="primary")

    def on_descendant_focus(self, event: Any) -> None:
        selected = getattr(event.control, "id", None)
        if selected in self.HELP_TEXT:
            self.query_one("#cluster-help", Static).update(self.HELP_TEXT[selected])

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(False)
            return
        name = self.query_one("#cluster-name", Input).value.strip()
        host = self.query_one("#cluster-host", Input).value.strip()
        user = self.query_one("#cluster-user", Input).value.strip() or None
        workspace = self.query_one("#cluster-workspace", Input).value.strip()
        project = self.query_one("#cluster-project", Input).value.strip() or None
        scheduler = str(self.query_one("#cluster-scheduler", Select).value)
        profile = ClusterProfile(
            name,
            transport="ssh",
            scheduler=scheduler,
            host=host,
            user=user,
            workspace=workspace,
            environment="managed",
            project_root=project,
            gpu_access=GpuAccessPolicy(mode="scheduler" if scheduler == "slurm" else "auto"),
        )
        ClusterCatalog.add(ClusterCatalog.user_path(), profile)
        self.dismiss(True)


class LambdaForgeApp(App[None]):
    """One responsive shell for Work, Studies, Clusters, Datasets and Results."""

    CSS_PATH = "theme.tcss"
    TITLE = "LambdaForge Research Console"
    COMMANDS = {ActionProvider}
    BINDINGS = [
        ("ctrl+p", "command_palette", "Commands"),
        ("question_mark", "help", "Help"),
        ("escape", "go_back", "Back"),
    ]

    def __init__(self, services: ConsoleServices | None = None) -> None:
        super().__init__()
        self.services = services or ConsoleServices()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="console-body"):
            with Vertical(id="sidebar"):
                yield Label("LambdaForge", id="brand")
                for label, target in (
                    ("Overview", "overview"),
                    ("Work", "work"),
                    ("Studies", "studies"),
                    ("Clusters", "clusters"),
                    ("Datasets", "datasets"),
                    ("Results", "results"),
                ):
                    yield Button(label, id=f"nav-{target}", classes="nav-button", flat=True)
            with Vertical(id="content"):
                yield OverviewScreen(self.services.overview_snapshot, id="overview")
                yield WorkScreen(self.services.overview_snapshot, id="work")
                yield StudyScreen(self.services.overview_snapshot, id="studies")
                yield ClusterScreen(self.services.cluster_rows, id="clusters")
                yield DatasetScreen(self.services.dataset_rows, id="datasets")
                yield ResultsScreen(self.services.result_rows, id="results")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id and event.button.id.startswith("nav-"):
            self.show_screen(event.button.id.removeprefix("nav-"))

    def on_mount(self) -> None:
        self.show_screen("overview")

    def on_resize(self, event: Resize) -> None:
        self.screen.set_class(event.size.width < 90, "narrow")

    def show_screen(self, name: str) -> None:
        for screen in self.query(DataScreen):
            screen.display = screen.id == name

    def open_console_action(self, action: ConsoleAction) -> None:
        self.show_screen(action.screen)
        if action.family == "cluster" and action.operation == "add":
            self.push_screen(ClusterEditor(self.services), self._cluster_saved)
            return
        if action.family == "result" and action.operation in {"analyze", "report"}:
            results = self.query_one("#results", ResultsScreen)
            selector = results.selected_selector()
            if selector is None:
                self.notify("Select one completed Execution first.", severity="warning")
                return
            self._run_result_action(action.operation, selector)
            return
        if action.destructive:
            self.push_screen(
                ConfirmationDialog(
                    f"Confirm {action.operation}",
                    "The exact selected target and preview will be shown before the domain "
                    "service applies this operation.",
                )
            )
            return
        self.notify(
            f"{action.name}: select a target in {action.screen.title()} to continue.",
            title="Action ready",
        )

    def _run_result_action(self, operation: str, selector: str) -> None:
        """Run potentially expensive analysis/report work without blocking Textual."""
        def execute() -> None:
            try:
                if operation == "analyze":
                    value = self.services.analyze(selector)
                    status = value.get("source", {}).get("status", "unknown")
                    message = f"Analysis ready ({status}) for {selector}."
                else:
                    destination = Path.cwd() / f"{selector}-analysis.html"
                    path = self.services.report(selector, destination)
                    message = f"Offline report written to {path}."
            except Exception as error:
                self.call_from_thread(
                    self.notify,
                    f"{type(error).__name__}: {error}",
                    title=f"Result {operation} failed",
                    severity="error",
                )
                return
            self.call_from_thread(self.notify, message, title=f"Result {operation}")
            self.call_from_thread(self.query_one("#results", ResultsScreen).reload)

        Thread(
            target=execute,
            daemon=True,
            name=f"lambdaforge-tui-result-{operation}",
        ).start()

    def _cluster_saved(self, saved: bool | None) -> None:
        if saved:
            self.services = ConsoleServices()
            clusters = self.query_one("#clusters", ClusterScreen)
            clusters.loader = self.services.cluster_rows
            clusters.reload()
            self.notify(
                "Cluster profile saved. Test/doctor/bootstrap remain explicit next actions."
            )

    def action_help(self) -> None:
        self.push_screen(ContextHelp())

    def action_go_back(self) -> None:
        self.show_screen("overview")


def run_console() -> int:
    """Run the full-screen application and return a console-script exit code."""
    LambdaForgeApp().run()
    return 0


__all__ = ["ClusterEditor", "ConfirmationDialog", "LambdaForgeApp", "run_console"]
