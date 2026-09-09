"""Full-screen Textual Research Console launched by bare ``lf``."""

from __future__ import annotations

import re
import shlex
import webbrowser
from collections.abc import Iterable, Mapping
from functools import partial
from pathlib import Path
from threading import Thread
from typing import Any, cast

import yaml
from rich.text import Text
from textual.app import App, ComposeResult
from textual.command import Hit, Hits, Provider
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Resize
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    DirectoryTree,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from lambdaforge.analysis.Report import write_resource_html
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.tui.models import PALETTE_ACTIONS, ConsoleAction
from lambdaforge.tui.screens.Base import DataScreen, EntitySelected
from lambdaforge.tui.screens.ClusterScreen import ClusterScreen
from lambdaforge.tui.screens.DatasetScreen import DatasetScreen
from lambdaforge.tui.screens.OverviewScreen import OverviewScreen
from lambdaforge.tui.screens.ResultsScreen import ResultsScreen
from lambdaforge.tui.screens.StudyScreen import StudyScreen
from lambdaforge.tui.screens.WorkScreen import WorkScreen
from lambdaforge.tui.screens.Workspace import (
    ClusterWorkspace,
    DatasetWorkspace,
    ResultWorkspace,
    StudyWorkspace,
    WorkWorkspace,
)
from lambdaforge.tui.services import ConsoleServices
from lambdaforge.tui.viewmodels import structured_text
from lambdaforge.tui.widgets import ResourceDashboard


class ActionProvider(Provider):
    """Expose the explicit CLI/TUI parity inventory through fuzzy search."""

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for action in PALETTE_ACTIONS:
            label = f"{action.name} · {action.operation}"
            score = matcher.match(label)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(label),
                    partial(cast(Any, self.app).open_console_action, action),
                    help=f"Open {action.screen}; direct domain service, never a CLI subprocess.",
                )


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
    """Edit every durable cluster-profile field without exposing credential values."""

    BINDINGS = [("escape", "cancel", "Exit without saving")]

    HELP_TEXT = {
        "cluster-name": "A short stable name used by --on and dataset placement selectors.",
        "cluster-host": "SSH hostname or IP. LambdaForge never stores a password in this profile.",
        "cluster-user": "Remote account name. Authentication remains in OpenSSH/keyring.",
        "cluster-port": "SSH TCP port, normally 22.",
        "cluster-transport": (
            "SSH reaches another machine; local is intended only for an explicit local profile."
        ),
        "cluster-workspace": "Absolute remote root for owned jobs, cache and managed environments.",
        "cluster-project": (
            "Optional remote mirror of this project for large inputs and published outputs."
        ),
        "cluster-data-environment": (
            "Logical data-placement identity. It defaults to the profile name and may be shared "
            "only when that placement is intentionally the same."
        ),
        "cluster-project-module": "Optional default import module for project-owned Work classes.",
        "cluster-scheduler": (
            "Direct runs a detached supervisor after SSH. SLURM submits through the site scheduler."
        ),
        "cluster-auth-mode": (
            "OpenSSH uses your agent/config. Password stores only a keyring/env reference here."
        ),
        "cluster-credential": (
            "A keyring: or env: reference; never enter the password itself in this field."
        ),
        "cluster-known-hosts": (
            "Optional explicit trust database; unknown SSH hosts remain rejected."
        ),
        "cluster-environment": (
            "Managed creates immutable user-space environments; existing trusts a preinstalled one."
        ),
        "cluster-python-strategy": (
            "Auto resolves a compatible Python; existing requires the named executable; managed "
            "permits a user-space runtime installation."
        ),
        "cluster-python-executable": "Executable name or absolute remote Python path.",
        "cluster-python-version": "Optional exact Python minor/patch requirement, such as 3.11.",
        "cluster-wheelhouse": "Optional remote wheel source for offline managed installation.",
        "cluster-torch-channel": "Automatic or explicitly reviewed official PyTorch wheel channel.",
        "cluster-require-cuda": (
            "Require a working CUDA tensor probe, permit CPU, or infer it from each Work request."
        ),
        "cluster-state-root": "Small durable controller state and dataset registry root.",
        "cluster-cache-root": "Reconstructible bundles, runtimes and immutable environments root.",
        "cluster-run-root": "Attempt logs, telemetry, checkpoints and result envelopes root.",
        "cluster-dataset-root": "Optional durable root for managed dataset placements.",
        "cluster-cache-size": "Optional cache retention ceiling, for example 500GiB.",
        "cluster-cache-age": "Optional cache retention age, for example 30d.",
        "cluster-connect-timeout": "SSH transport connection deadline.",
        "cluster-auth-timeout": "SSH authentication deadline.",
        "cluster-banner-timeout": "SSH server-banner deadline.",
        "cluster-keepalive": "Idle SSH keepalive interval; zero disables it.",
        "cluster-persist": "Idle connection reuse lifetime; zero disables persistence.",
        "cluster-command-timeout": (
            "Optional per-command deadline. Leave empty for long scientific operations."
        ),
        "cluster-gpu-mode": (
            "Who grants GPU access. Command wraps GPU-sensitive processes with the site's argv; "
            "claim/release optionally owns one reservation for the complete submission."
        ),
        "cluster-gpu-prefix": (
            "Shell-style input converted to an argv list, for example: gpu exec --guaranteed. "
            "No shell expansion is performed."
        ),
        "cluster-gpu-claim": (
            "Optional persistent reservation argv. Only {gpu_count} is interpolated; a matching "
            "release command is mandatory."
        ),
        "cluster-gpu-release": (
            "Cleanup argv paired with the claim and attempted after success, failure or "
            "cancellation."
        ),
        "cluster-advanced": (
            "Validated YAML for uncommon OpenSSH and SLURM dialect fields. It cannot override the "
            "guided fields on the other tabs."
        ),
    }

    ADVANCED_KEYS = frozenset(
        {
            "ssh_options",
            "command_prefix",
            "scheduler_options",
            "resource_mapping",
            "scheduler_directives",
            "scheduler_commands",
            "job_script",
        }
    )

    def __init__(self, services: ConsoleServices, profile: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.services = services
        self.profile = dict(profile or {})

    @staticmethod
    def _nested(source: Mapping[str, Any], key: str) -> Mapping[str, Any]:
        value = source.get(key, {})
        return value if isinstance(value, Mapping) else {}

    def _advanced_yaml(self) -> str:
        if not self.profile:
            return "# Optional uncommon SSH/SLURM settings.\n"
        name = str(self.profile.get("name", "cluster"))
        descriptor = {key: value for key, value in self.profile.items() if key != "name"}
        compact = ClusterProfile.from_mapping(name, descriptor).to_dict(include_defaults=False)
        advanced = {key: compact[key] for key in self.ADVANCED_KEYS if key in compact}
        return yaml.safe_dump(advanced, sort_keys=False, allow_unicode=True) if advanced else "{}\n"

    def compose(self) -> ComposeResult:
        connection = self._nested(self.profile, "connection")
        storage = self._nested(self.profile, "storage")
        python = self.profile.get("python", {})
        python = (
            python
            if isinstance(python, Mapping)
            else {
                "strategy": "existing",
                "executable": str(python or "python3"),
                "version": None,
                "allow_managed_install": True,
            }
        )
        pytorch = self._nested(self.profile, "pytorch")
        auth = self._nested(self.profile, "auth")
        gpu = self._nested(self.profile, "gpu_access")
        with Vertical(id="cluster-editor", classes="modal-card wide-modal"):
            yield Label("Edit cluster" if self.profile else "Add cluster", classes="modal-title")
            with TabbedContent(initial="cluster-identity-tab", id="cluster-editor-tabs"):
                with TabPane("Identity", id="cluster-identity-tab"):
                    with VerticalScroll(classes="cluster-editor-page"):
                        yield Label("Name", classes="field-label")
                        yield Input(
                            value=str(self.profile.get("name", "")),
                            placeholder="Cluster name",
                            id="cluster-name",
                            disabled=bool(self.profile),
                        )
                        yield Label("Host or IP", classes="field-label")
                        yield Input(
                            value=str(self.profile.get("host") or ""),
                            placeholder="gpu.example.org",
                            id="cluster-host",
                        )
                        yield Label("Transport", classes="field-label")
                        yield Select(
                            (("SSH", "ssh"), ("Local process", "local")),
                            value=str(self.profile.get("transport", "ssh")),
                            id="cluster-transport",
                        )
                        yield Label("Remote user", classes="field-label")
                        yield Input(
                            value=str(self.profile.get("user") or ""),
                            placeholder="Research account",
                            id="cluster-user",
                        )
                        yield Label("SSH port", classes="field-label")
                        yield Input(value=str(self.profile.get("port", 22)), id="cluster-port")
                        yield Label("Authentication", classes="field-label")
                        yield Select(
                            (("OpenSSH agent/key", "openssh"), ("Password reference", "password")),
                            value=str(auth.get("mode", "openssh")),
                            id="cluster-auth-mode",
                        )
                        yield Label("Credential reference", classes="field-label")
                        yield Input(
                            value=str(auth.get("credential") or ""),
                            placeholder="keyring:cluster/name/user@host or env:VARIABLE",
                            id="cluster-credential",
                        )
                        yield Label("Known-hosts file", classes="field-label")
                        yield Input(
                            value=str(self.profile.get("known_hosts") or ""),
                            placeholder="Optional explicit known_hosts path",
                            id="cluster-known-hosts",
                        )
                        yield Label("Execution backend", classes="field-label")
                        yield Select(
                            (("Direct processes", "local"), ("SLURM", "slurm")),
                            value=str(self.profile.get("scheduler", "local")),
                            id="cluster-scheduler",
                        )
                        yield Label("Owned workspace", classes="field-label")
                        yield Input(
                            value=str(self.profile.get("workspace") or ""),
                            placeholder="/home/user",
                            id="cluster-workspace",
                        )
                        yield Label("Project mirror", classes="field-label")
                        yield Input(
                            value=str(self.profile.get("project_root") or ""),
                            placeholder="Optional /home/user/project",
                            id="cluster-project",
                        )
                        yield Label("Data environment name", classes="field-label")
                        yield Input(
                            value=str(self.profile.get("data_environment") or ""),
                            placeholder="Defaults to the cluster name",
                            id="cluster-data-environment",
                        )
                        yield Label("Project module", classes="field-label")
                        yield Input(
                            value=str(self.profile.get("project_module") or ""),
                            placeholder="Optional importable.module",
                            id="cluster-project-module",
                        )
                with TabPane("Runtime", id="cluster-runtime-tab"):
                    with VerticalScroll(classes="cluster-editor-page"):
                        yield Label("Environment provider", classes="field-label")
                        yield Select(
                            (
                                ("Managed immutable environment", "managed"),
                                ("Existing environment", "existing"),
                            ),
                            value=str(self.profile.get("environment", "managed")),
                            id="cluster-environment",
                        )
                        yield Label("Python strategy", classes="field-label")
                        yield Select(
                            (
                                ("Automatic compatible runtime", "auto"),
                                ("Use existing Python", "existing"),
                                ("Create managed Python", "managed"),
                            ),
                            value=str(python.get("strategy", "auto")),
                            id="cluster-python-strategy",
                        )
                        yield Label("Python executable", classes="field-label")
                        yield Input(
                            value=str(python.get("executable", "python3")),
                            id="cluster-python-executable",
                        )
                        yield Label("Requested Python version", classes="field-label")
                        yield Input(
                            value=str(python.get("version") or ""),
                            placeholder="Optional, for example 3.11",
                            id="cluster-python-version",
                        )
                        yield Checkbox(
                            "Allow a user-space managed Python installation",
                            value=bool(python.get("allow_managed_install", True)),
                            id="cluster-python-managed",
                        )
                        yield Label("Wheelhouse", classes="field-label")
                        yield Input(
                            value=str(self.profile.get("wheelhouse") or ""),
                            placeholder="Optional remote offline wheel directory",
                            id="cluster-wheelhouse",
                        )
                        yield Label("PyTorch channel", classes="field-label")
                        yield Select(
                            tuple(
                                (item, item)
                                for item in (
                                    "auto",
                                    "cpu",
                                    "cu118",
                                    "cu121",
                                    "cu124",
                                    "cu126",
                                    "cu128",
                                    "cu130",
                                )
                            ),
                            value=str(pytorch.get("channel", "auto")),
                            id="cluster-torch-channel",
                        )
                        yield Label("CUDA requirement", classes="field-label")
                        required = pytorch.get("require_cuda", "auto")
                        yield Select(
                            (
                                ("Automatic", "auto"),
                                ("CUDA required", "true"),
                                ("CPU permitted", "false"),
                            ),
                            value=("auto" if required in {None, "auto"} else str(required).lower()),
                            id="cluster-require-cuda",
                        )
                with TabPane("Storage & SSH", id="cluster-storage-tab"):
                    with VerticalScroll(classes="cluster-editor-page"):
                        for label, identifier, key, placeholder in (
                            (
                                "State root",
                                "cluster-state-root",
                                "state_root",
                                "Derived from workspace when empty",
                            ),
                            (
                                "Cache/environment root",
                                "cluster-cache-root",
                                "cache_root",
                                "Derived from workspace when empty",
                            ),
                            (
                                "Run root",
                                "cluster-run-root",
                                "run_root",
                                "Derived from workspace when empty",
                            ),
                            (
                                "Dataset root",
                                "cluster-dataset-root",
                                "dataset_root",
                                "Optional durable dataset location",
                            ),
                            (
                                "Cache size limit",
                                "cluster-cache-size",
                                "cache_max_size",
                                "Optional bytes or 500GiB",
                            ),
                            (
                                "Cache age limit",
                                "cluster-cache-age",
                                "cache_max_age",
                                "Optional seconds or 30d",
                            ),
                        ):
                            yield Label(label, classes="field-label")
                            stored = storage.get(key)
                            if key == "cache_max_size" and isinstance(stored, (int, float)):
                                stored = f"{int(stored)}B"
                            elif key == "cache_max_age" and isinstance(stored, (int, float)):
                                stored = f"{int(stored)}s"
                            yield Input(
                                value=str(stored or ""),
                                placeholder=placeholder,
                                id=identifier,
                            )
                        for label, identifier, key, default, placeholder in (
                            (
                                "Connect timeout",
                                "cluster-connect-timeout",
                                "connect_timeout",
                                15,
                                "Seconds or duration",
                            ),
                            (
                                "Authentication timeout",
                                "cluster-auth-timeout",
                                "auth_timeout",
                                15,
                                "Seconds or duration",
                            ),
                            (
                                "Banner timeout",
                                "cluster-banner-timeout",
                                "banner_timeout",
                                15,
                                "Seconds or duration",
                            ),
                            (
                                "Keepalive interval",
                                "cluster-keepalive",
                                "keepalive",
                                30,
                                "0 disables it",
                            ),
                            (
                                "Connection reuse",
                                "cluster-persist",
                                "persist",
                                60,
                                "OpenSSH ControlPersist duration",
                            ),
                            (
                                "Per-command timeout",
                                "cluster-command-timeout",
                                "command_timeout",
                                "",
                                "Empty permits long scientific commands",
                            ),
                        ):
                            yield Label(label, classes="field-label")
                            yield Input(
                                value=str(connection.get(key, default) or ""),
                                placeholder=placeholder,
                                id=identifier,
                            )
                        yield Checkbox(
                            "Reuse SSH connections when supported",
                            value=bool(connection.get("multiplex", True)),
                            id="cluster-multiplex",
                        )
                with TabPane("GPU access", id="cluster-gpu-tab"):
                    with VerticalScroll(classes="cluster-editor-page"):
                        yield Label("GPU ownership policy", classes="field-label")
                        yield Select(
                            (
                                ("Automatic from backend", "auto"),
                                ("Scheduler allocation", "scheduler"),
                                ("Exclusive LambdaForge leases", "exclusive"),
                                ("Shared host; external use allowed", "shared"),
                                ("Site claim/launcher command", "command"),
                            ),
                            value=str(gpu.get("mode", "auto")),
                            id="cluster-gpu-mode",
                        )
                        yield Label("GPU command prefix", classes="field-label")
                        yield Input(
                            value=shlex.join(
                                tuple(str(item) for item in gpu.get("command_prefix", ()))
                            ),
                            placeholder="For example: gpu exec",
                            id="cluster-gpu-prefix",
                        )
                        yield Label("Persistent claim command", classes="field-label")
                        yield Input(
                            value=shlex.join(
                                tuple(str(item) for item in gpu.get("claim_command", ()))
                            ),
                            placeholder='For example: gpu claim --numgpus "{gpu_count}"',
                            id="cluster-gpu-claim",
                        )
                        yield Label("Paired release command", classes="field-label")
                        yield Input(
                            value=shlex.join(
                                tuple(str(item) for item in gpu.get("release_command", ()))
                            ),
                            placeholder="For example: gpu release",
                            id="cluster-gpu-release",
                        )
                        yield Static(
                            "LambdaForge treats CUDA_VISIBLE_DEVICES values as opaque grants and "
                            "never replaces them with guessed physical indices.",
                            classes="muted",
                        )
                with TabPane("Advanced", id="cluster-advanced-tab"):
                    with Vertical(classes="cluster-editor-page"):
                        yield Static(
                            "Optional validated YAML: ssh_options, command_prefix, "
                            "scheduler_options, resource_mapping, scheduler_directives, "
                            "scheduler_commands and job_script.",
                            classes="muted",
                        )
                        yield TextArea(
                            self._advanced_yaml(),
                            language="yaml",
                            show_line_numbers=True,
                            id="cluster-advanced",
                        )
            yield Static(self.HELP_TEXT["cluster-name"], id="cluster-help", classes="help-panel")
            yield Static("", id="cluster-editor-error", classes="status-line")
            with Horizontal(classes="modal-actions"):
                yield Button("Exit without saving", id="cancel")
                yield Button("Save profile", id="save", variant="primary")

    def on_descendant_focus(self, event: Any) -> None:
        selected = getattr(event.control, "id", None)
        if selected in self.HELP_TEXT:
            self.query_one("#cluster-help", Static).update(self.HELP_TEXT[selected])

    def action_cancel(self) -> None:
        self.dismiss(False)

    def _input(self, identifier: str) -> str:
        return self.query_one(f"#{identifier}", Input).value.strip()

    @staticmethod
    def _optional(value: str) -> str | None:
        return value or None

    def _advanced(self) -> dict[str, Any]:
        text = self.query_one("#cluster-advanced", TextArea).text
        value = yaml.safe_load(text) if text.strip() else {}
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError("Advanced cluster YAML must contain a mapping.")
        unknown = set(value) - self.ADVANCED_KEYS
        if unknown:
            raise ValueError(f"Unsupported advanced cluster field(s): {sorted(unknown)}.")
        return dict(value)

    def _descriptor(self) -> tuple[str, dict[str, Any]]:
        name = self._input("cluster-name")
        scheduler = str(self.query_one("#cluster-scheduler", Select).value)
        auth_mode = str(self.query_one("#cluster-auth-mode", Select).value)
        gpu_mode = str(self.query_one("#cluster-gpu-mode", Select).value)
        require_cuda = str(self.query_one("#cluster-require-cuda", Select).value)
        storage_fields = {
            "state_root": self._input("cluster-state-root"),
            "cache_root": self._input("cluster-cache-root"),
            "run_root": self._input("cluster-run-root"),
            "dataset_root": self._input("cluster-dataset-root"),
            "cache_max_size": self._input("cluster-cache-size"),
            "cache_max_age": self._input("cluster-cache-age"),
        }
        gpu_access: dict[str, Any] = {"mode": gpu_mode}
        if gpu_mode == "command":
            gpu_access.update(
                {
                    "command_prefix": list(shlex.split(self._input("cluster-gpu-prefix"))),
                    "claim_command": list(shlex.split(self._input("cluster-gpu-claim"))),
                    "release_command": list(shlex.split(self._input("cluster-gpu-release"))),
                }
            )
        descriptor = self._advanced()
        descriptor.update(
            {
                "transport": str(self.query_one("#cluster-transport", Select).value),
                "scheduler": scheduler,
                "host": self._input("cluster-host"),
                "user": self._optional(self._input("cluster-user")),
                "port": int(self._input("cluster-port")),
                "auth": {
                    "mode": auth_mode,
                    "credential": (
                        self._optional(self._input("cluster-credential"))
                        if auth_mode == "password"
                        else None
                    ),
                },
                "known_hosts": self._optional(self._input("cluster-known-hosts")),
                # Keep the legacy alias synchronized instead of retaining a hidden stale value.
                "ssh_timeout": self._input("cluster-connect-timeout"),
                "connection": {
                    "connect_timeout": self._input("cluster-connect-timeout"),
                    "auth_timeout": self._input("cluster-auth-timeout"),
                    "banner_timeout": self._input("cluster-banner-timeout"),
                    "keepalive": self._input("cluster-keepalive"),
                    "multiplex": self.query_one("#cluster-multiplex", Checkbox).value,
                    "persist": self._input("cluster-persist"),
                    "command_timeout": self._optional(self._input("cluster-command-timeout")),
                },
                "workspace": self._input("cluster-workspace"),
                "storage": {key: value for key, value in storage_fields.items() if value},
                "python": {
                    "strategy": str(self.query_one("#cluster-python-strategy", Select).value),
                    "executable": self._input("cluster-python-executable"),
                    "version": self._optional(self._input("cluster-python-version")),
                    "allow_managed_install": self.query_one(
                        "#cluster-python-managed", Checkbox
                    ).value,
                },
                "environment": str(self.query_one("#cluster-environment", Select).value),
                "wheelhouse": self._optional(self._input("cluster-wheelhouse")),
                "pytorch": {
                    "channel": str(self.query_one("#cluster-torch-channel", Select).value),
                    "require_cuda": ("auto" if require_cuda == "auto" else require_cuda == "true"),
                },
                "project_module": self._optional(self._input("cluster-project-module")),
                "project_root": self._optional(self._input("cluster-project")),
                "data_environment": self._optional(self._input("cluster-data-environment")),
                "gpu_access": gpu_access,
            }
        )
        return name, descriptor

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(False)
            return
        try:
            name, descriptor = self._descriptor()
            profile = ClusterProfile.from_mapping(name, descriptor)
            source = self.services.catalog.source(name) if self.profile else None
            ClusterCatalog.add(source or ClusterCatalog.user_path(), profile)
        except Exception as error:
            self.query_one("#cluster-editor-error", Static).update(
                f"{type(error).__name__}: {error}"
            )
            return
        self.dismiss(True)


class YamlDirectoryTree(DirectoryTree):
    """Show directories and executable Work YAMLs, hiding unrelated files."""

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        for path in paths:
            if path.name.startswith(".") or path.name in {
                "__pycache__",
                "build",
                "dist",
                "node_modules",
            }:
                continue
            try:
                if path.is_dir() or path.suffix.lower() in {".yaml", ".yml"}:
                    yield path
            except OSError:
                continue


class YamlFilePicker(ModalScreen[Path | None]):
    """Project-oriented YAML browser with an editable location for external files."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, root: Path | None = None) -> None:
        super().__init__()
        self.root = (root or Path.cwd()).expanduser().resolve()

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-card wide-modal", id="yaml-picker-dialog"):
            yield Label("Choose a Work YAML", classes="modal-title")
            yield Static(
                "Only .yaml and .yml files are shown. Open directories to navigate; "
                "select a file with Enter or the mouse."
            )
            with Horizontal(id="yaml-picker-location-row"):
                yield Input(value=str(self.root), id="yaml-picker-location")
                yield Button("Open location", id="yaml-picker-open", variant="primary")
            yield Static("", id="yaml-picker-error", classes="status-line")
            yield YamlDirectoryTree(self.root, id="yaml-picker-tree")
            with Horizontal(classes="modal-actions"):
                yield Button("Cancel", id="yaml-picker-cancel")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        event.stop()
        self._select(event.path)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "yaml-picker-cancel":
            self.dismiss(None)
        elif event.button.id == "yaml-picker-open":
            self._open_location()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "yaml-picker-location":
            self._open_location()

    def _open_location(self) -> None:
        raw = self.query_one("#yaml-picker-location", Input).value.strip()
        selected = Path(raw).expanduser().resolve()
        if selected.is_file():
            self._select(selected)
            return
        if not selected.is_dir():
            self.query_one("#yaml-picker-error", Static).update(
                "That location does not exist or is not a readable directory."
            )
            return
        self.query_one("#yaml-picker-error", Static).update("")
        self.query_one("#yaml-picker-tree", YamlDirectoryTree).path = selected

    def _select(self, selected: Path) -> None:
        path = selected.expanduser().resolve()
        if path.suffix.lower() not in {".yaml", ".yml"} or not path.is_file():
            self.query_one("#yaml-picker-error", Static).update(
                "Choose an existing .yaml or .yml file."
            )
            return
        self.dismiss(path)


class WorkLaunchDialog(ModalScreen[dict[str, Any] | None]):
    """Validate, explain and asynchronously submit an authored Work YAML."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, services: ConsoleServices) -> None:
        super().__init__()
        self.services = services
        self._busy = False
        loader = getattr(services, "recent_work_configs", None)
        try:
            values = loader(limit=12) if callable(loader) else ()
        except Exception:
            values = ()
        self._recent = [dict(item) for item in values if isinstance(item, Mapping)]

    def compose(self) -> ComposeResult:
        clusters = self.services.cluster_names()
        options = tuple((name, name) for name in clusters)
        with Vertical(classes="modal-card wide-modal", id="work-launch-dialog"):
            yield Label("Run Work", classes="modal-title")
            with VerticalScroll(id="work-launch-body"):
                yield Static(
                    "Select a recent Work or browse for another YAML. Both routes fill the same "
                    "selector; Submit always validates before starting asynchronous preparation. "
                    "The console never edits science YAML."
                )
                yield Static("RECENT WORKS", classes="section-title")
                recent: DataTable[Any] = DataTable(
                    id="launch-recent-table", cursor_type="row", zebra_stripes=True
                )
                recent.add_columns("Work", "YAML", "Last used")
                for index, item in enumerate(self._recent):
                    path = Path(str(item.get("path", "")))
                    recent.add_row(
                        str(item.get("name") or path.stem),
                        self._display_path(path),
                        self._display_timestamp(item.get("last_used_utc")),
                        key=str(index),
                    )
                yield recent
                yield Static(
                    (
                        "Highlight a recent YAML and press Enter to select it."
                        if self._recent
                        else (
                            "No recent Work YAMLs yet. Browse once and validated selections "
                            "will appear here."
                        )
                    ),
                    id="launch-recent-help",
                    classes="legend",
                )
                with Horizontal(id="launch-config-row"):
                    yield Input(placeholder="experiments/train.yaml", id="launch-config")
                    yield Button("Browse YAML…", id="launch-browse")
                with Horizontal(id="launch-target-row"):
                    yield Label("Run on", id="launch-cluster-label")
                    yield Select(
                        options,
                        value="local" if "local" in clusters else clusters[0],
                        id="launch-cluster",
                    )
                yield Static(
                    "Choose a YAML to continue.", id="launch-status", classes="status-line"
                )
                yield Static("No configuration inspected yet.", id="launch-preview-content")
            with Horizontal(classes="modal-actions"):
                yield Button("Cancel", id="launch-cancel")
                yield Button("Submit Work", id="launch-submit", variant="success", disabled=True)

    def on_mount(self) -> None:
        if self._recent:
            self.query_one("#launch-recent-table", DataTable).focus()
        else:
            self.query_one("#launch-config", Input).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "launch-cancel":
            self.dismiss(None)
            return
        if event.button.id == "launch-browse":
            self.app.push_screen(YamlFilePicker(Path.cwd()), self._yaml_picked)
            return
        raw = self.query_one("#launch-config", Input).value.strip()
        if event.button.id == "launch-submit" and raw:
            path = Path(raw).expanduser().resolve()
            cluster = str(self.query_one("#launch-cluster", Select).value)
            self._validate_and_submit(path, cluster)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id != "launch-recent-table":
            return
        path = self._recent_path(event.cursor_row)
        if path is not None:
            self._select_path(path, "Recent Work selected")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "launch-recent-table":
            path = self._recent_path(event.cursor_row)
            if path is not None:
                self._select_path(path, "Recent Work selected")
                self.query_one("#launch-submit", Button).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "launch-config" or self._busy:
            return
        selected = bool(event.value.strip())
        self.query_one("#launch-submit", Button).disabled = not selected
        self.query_one("#launch-status", Static).update(
            "Ready to validate and submit." if selected else "Choose a YAML to continue."
        )

    def _recent_path(self, row: int) -> Path | None:
        if not 0 <= row < len(self._recent):
            return None
        return Path(str(self._recent[row].get("path", ""))).expanduser().resolve()

    def _yaml_picked(self, path: Path | None) -> None:
        if path is None:
            return
        self._select_path(path, "YAML selected")

    def _select_path(self, path: Path, source: str) -> None:
        self.query_one("#launch-config", Input).value = str(path)
        self.query_one("#launch-submit", Button).disabled = False
        self.query_one("#launch-status", Static).update(
            f"{source} · Submit will validate {path.name} before launch."
        )

    def _validate_and_submit(self, path: Path, cluster: str) -> None:
        if self._busy:
            return
        self._set_busy(True, "Validating the Work signature and resolving its execution plan…")

        def run() -> None:
            try:
                validation = self.services.validate_work(path)
                if not validation.get("valid"):
                    errors = validation.get("errors")
                    detail = (
                        "\n\n".join(str(item) for item in errors)
                        if isinstance(errors, list) and errors
                        else "Work configuration is invalid."
                    )
                    raise ValueError(detail)
                explanation = self.services.explain_work(path)
            except Exception as error:
                self.app.call_from_thread(self._preview_failed, error, "validation")
                return
            remember = getattr(self.services, "remember_work_config", None)
            if callable(remember):
                try:
                    remember(path, name=str(explanation.get("name") or path.stem))
                except Exception:
                    pass
            self.app.call_from_thread(
                self._preview_ready,
                path,
                validation,
                explanation,
                cluster,
            )

        Thread(target=run, daemon=True, name="lambdaforge-tui-work-preview").start()

    def _preview_failed(self, error: Exception, phase: str = "validation") -> None:
        label = "Validation" if phase == "validation" else "Submission"
        self._set_busy(False, f"{label} failed · review the explanation below.")
        preview = self.query_one("#launch-preview-content", Static)
        preview.add_class("validation-error")
        content = Text()
        content.append(f"{label.upper()} FAILED\n", style="bold red")
        content.append(str(error).strip() or type(error).__name__)
        preview.update(content)
        preview.scroll_visible(animate=False)

    def _preview_ready(
        self,
        path: Path,
        validation: dict[str, Any],
        explanation: dict[str, Any],
        cluster: str,
    ) -> None:
        # Validation and submission are one operation from the user's perspective. Keep
        # the modal in a visible busy state between both phases; a large/non-standard
        # explanation must never prevent an already validated Work from being queued.
        self._set_busy(
            True,
            f"Validated {explanation.get('name', path.stem)} · submitting to {cluster}…",
        )
        preview = self.query_one("#launch-preview-content", Static)
        try:
            preview.remove_class("validation-error")
            rendered = structured_text(
                {
                    "configuration": str(path),
                    "validation": validation,
                    "resolved_plan": explanation,
                },
                heading="VALIDATION SUCCEEDED · SUBMISSION STARTED",
                limit=80,
            )
        except Exception as error:
            rendered = (
                "VALIDATION SUCCEEDED · SUBMISSION STARTED\n"
                f"Configuration           {path}\n"
                f"Preview unavailable     {type(error).__name__}: {error}"
            )
        try:
            # Explanations contain authored docstrings and parameter values. They are
            # literal evidence, never Textual markup (``[x]`` is common in type/docs).
            preview.update(Text(rendered))
            preview.scroll_visible(animate=False)
        except Exception as error:
            self._set_busy(
                True,
                f"Validated · preview unavailable ({type(error).__name__}) · "
                f"submitting to {cluster}…",
            )
        finally:
            # Presentation is diagnostic only. A validated Work must still reach the
            # durable enqueue boundary if a terminal/rendering implementation rejects
            # an otherwise harmless preview value.
            self._submit(path, cluster, already_busy=True)

    def _submit(self, path: Path, cluster: str, *, already_busy: bool = False) -> None:
        if self._busy and not already_busy:
            return
        if not already_busy:
            self._set_busy(True, f"Submitting {path.name} to {cluster}…")

        def run() -> None:
            try:
                result = self.services.submit_work(path, cluster)
            except Exception as error:
                self.app.call_from_thread(self._preview_failed, error, "submission")
                return
            self.app.call_from_thread(self.dismiss, result)

        Thread(target=run, daemon=True, name="lambdaforge-tui-work-submit").start()

    def _set_busy(self, busy: bool, message: str) -> None:
        self._busy = busy
        self.query_one("#launch-status", Static).update(message)
        self.query_one("#launch-browse", Button).disabled = busy
        self.query_one("#launch-cluster", Select).disabled = busy
        self.query_one("#launch-config", Input).disabled = busy
        self.query_one("#launch-submit", Button).disabled = busy or not bool(
            self.query_one("#launch-config", Input).value.strip()
        )

    @staticmethod
    def _display_timestamp(value: Any) -> str:
        text = str(value or "")
        return text[:16].replace("T", " ") if text else "unknown"

    @staticmethod
    def _display_path(path: Path) -> str:
        try:
            return str(path.relative_to(Path.cwd()))
        except ValueError:
            return str(path)


class LambdaForgeApp(App[None]):
    """One responsive shell for Work, Studies, Clusters, Datasets and Results."""

    CSS_PATH = "theme.tcss"
    TITLE = "LambdaForge Research Console"
    COMMANDS = {ActionProvider}
    BINDINGS = [
        ("ctrl+p", "command_palette", "Commands"),
        ("question_mark", "help", "Help"),
        ("escape", "go_back", "Back"),
        ("q", "quit_console", "Exit"),
    ]

    def __init__(self, services: ConsoleServices | None = None) -> None:
        super().__init__()
        self.services = services or ConsoleServices()
        project = getattr(getattr(self.services, "catalog", None), "project", None)
        if project is not None:
            self.sub_title = f"Project: {project.project_id}"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="console-body"):
            with Vertical(id="sidebar"):
                yield Label("LambdaForge", id="brand")
                yield Label("ACTION", classes="sidebar-section-label")
                yield Button(
                    "Run Work…",
                    id="nav-run",
                    classes="nav-button sidebar-action",
                    variant="primary",
                )
                yield Label("BROWSE", classes="sidebar-section-label")
                with Vertical(id="sidebar-navigation"):
                    for label, target in (
                        ("Overview", "overview"),
                        ("Work", "work"),
                        ("Studies", "studies"),
                        ("Clusters", "clusters"),
                        ("Datasets", "datasets"),
                        ("Results", "results"),
                    ):
                        yield Button(label, id=f"nav-{target}", classes="nav-button", flat=True)
                yield Static("", classes="sidebar-spacer")
                yield Label("SESSION", classes="sidebar-section-label")
                yield Button(
                    "Exit LambdaForge",
                    id="nav-exit",
                    classes="nav-button sidebar-action",
                    flat=True,
                )
            with Vertical(id="content"):
                yield OverviewScreen(self.services.overview_snapshot, id="overview")
                yield WorkScreen(self.services.overview_snapshot, id="work")
                yield StudyScreen(self.services.overview_snapshot, id="studies")
                yield ClusterScreen(self.services.cluster_rows, id="clusters")
                yield DatasetScreen(self.services.dataset_rows, id="datasets")
                yield ResultsScreen(self.services.result_rows, id="results")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "nav-exit":
            self.exit()
            return
        if event.button.id == "nav-run":
            self.push_screen(WorkLaunchDialog(self.services), self._work_submitted)
            return
        if event.button.id and event.button.id.startswith("nav-"):
            self.show_screen(event.button.id.removeprefix("nav-"))

    def on_mount(self) -> None:
        self.show_screen("overview")

    def on_resize(self, event: Resize) -> None:
        self.screen.set_class(event.size.width < 90, "narrow")

    def show_screen(self, name: str) -> None:
        selected: DataScreen | None = None
        for screen in self.query(DataScreen):
            screen.display = screen.id == name
            if screen.id == name:
                selected = screen
        if selected is not None:
            selected.reload()
            table = (
                selected.query_one("#overview-clusters")
                if isinstance(selected, OverviewScreen)
                else selected.query_one("#screen-table")
            )
            table.focus()

    def on_entity_selected(self, message: EntitySelected) -> None:
        """Push the real entity workspace selected with Enter or the right arrow."""
        message.stop()
        if message.kind == "study":
            self.push_screen(StudyWorkspace(message.value, self.services))
        elif message.kind == "work":
            self.push_screen(WorkWorkspace(message.value, self.services))
        elif message.kind == "cluster":
            self.push_screen(ClusterWorkspace(str(message.value), self.services))
        elif message.kind == "dataset":
            self.push_screen(DatasetWorkspace(message.value, self.services))
        elif message.kind == "result":
            self.push_screen(ResultWorkspace(message.value, self.services))

    def on_resource_dashboard_export_requested(
        self, message: ResourceDashboard.ExportRequested
    ) -> None:
        """Generate the optional resource report without blocking live refresh."""
        message.stop()
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", message.cluster).strip("-._") or "cluster"
        destination = Path.cwd() / ".lambdaforge" / "reports" / f"{safe}-resources.html"

        def export() -> None:
            try:
                path = write_resource_html(message.cluster, message.series, destination)
                opened = webbrowser.open(path.as_uri())
            except Exception as error:
                self.call_from_thread(
                    self.notify,
                    f"{type(error).__name__}: {error}",
                    title="Resource export failed",
                    severity="error",
                )
                return
            suffix = " and opened" if opened else ""
            self.call_from_thread(
                self.notify,
                f"Interactive resource report written{suffix}: {path}",
                title="Resource report",
            )

        Thread(target=export, daemon=True, name="lambdaforge-tui-resource-report").start()

    def open_console_action(self, action: ConsoleAction) -> None:
        if action.status != "IMPLEMENTED" or action.handler is None:
            self.notify(
                f"{action.name} remains available through the stable CLI.",
                severity="warning",
            )
            return
        self.show_screen(action.screen)
        if action.handler == "navigate":
            return
        if action.handler == "cluster_add":
            self.push_screen(ClusterEditor(self.services), self._cluster_saved)
            return
        if action.handler == "work_launch":
            self.push_screen(WorkLaunchDialog(self.services), self._work_submitted)
            return
        if action.handler in {"result_analyze", "result_report"}:
            results = self.query_one("#results", ResultsScreen)
            selector = results.selected_selector()
            if selector is None:
                self.notify("Select one completed Execution first.", severity="warning")
                return
            self._run_result_action(action.operation, selector)
            return

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

    def edit_cluster(self, name: str, callback: Any = None) -> None:
        profile = self.services.catalog.inspect(name)["profile"]

        def saved(value: bool | None) -> None:
            self._cluster_saved(value)
            if callback is not None:
                callback(value)

        self.push_screen(ClusterEditor(self.services, profile), saved)

    def _work_submitted(self, result: dict[str, Any] | None) -> None:
        if result is None:
            return
        self.notify(
            f"Submission accepted: {result.get('job_id', result)}",
            title="Work queued",
        )
        self.query_one("#overview", OverviewScreen).reload()
        self.query_one("#work", WorkScreen).reload()
        self.query_one("#studies", StudyScreen).reload()

    def action_help(self) -> None:
        self.push_screen(ContextHelp())

    def action_go_back(self) -> None:
        if len(self.screen_stack) > 1:
            self.pop_screen()
        else:
            self.show_screen("overview")

    def action_quit_console(self) -> None:
        """Make the visible q/Exit affordance deterministic on every root panel."""
        self.exit()


def run_console() -> int:
    """Run the full-screen application and return a console-script exit code."""
    LambdaForgeApp().run()
    return 0


__all__ = ["ClusterEditor", "LambdaForgeApp", "WorkLaunchDialog", "run_console"]
