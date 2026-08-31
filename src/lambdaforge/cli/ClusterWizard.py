"""Dependency-light guided editor for the native cluster CLI operations."""

from __future__ import annotations

import getpass
import importlib
import os
import re
import select
import shlex
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import yaml

from lambdaforge.cli.TerminalTheme import TerminalTheme
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.TorchInstallationPolicy import TorchInstallationPolicy

NativeClusterCommand = Callable[[Sequence[str]], int]


class _WizardExit(Exception):
    """Private control-flow signal for an explicit, side-effect-free wizard exit."""


@dataclass(frozen=True, slots=True)
class _Setting:
    key: str
    label: str
    explanation: str
    choices: tuple[tuple[Any, ...], ...] = ()


_SETTINGS: Mapping[str, tuple[_Setting, ...]] = {
    "Connection and authentication": (
        _Setting("transport", "Transport", "Use ssh for a remote host; local is in-process."),
        _Setting("host", "Host or IP", "SSH login hostname or address."),
        _Setting("user", "SSH user", "Remote account used for every connection."),
        _Setting("port", "SSH port", "TCP port used by the SSH service."),
        _Setting(
            "auth.mode",
            "Authentication mode",
            "openssh uses your agent/config; password resolves a secret at runtime.",
            (("OpenSSH", "openssh"), ("Password", "password")),
        ),
        _Setting("known_hosts", "Known-hosts file", "Explicit host-key trust file."),
        _Setting("ssh_options", "OpenSSH options", "YAML argv list passed to OpenSSH."),
        _Setting("ssh_timeout", "Legacy SSH timeout", "Connect timeout for older profiles."),
        _Setting("connection.connect_timeout", "Connect timeout", "Seconds to open TCP/SSH."),
        _Setting("connection.auth_timeout", "Authentication timeout", "Seconds for login."),
        _Setting("connection.banner_timeout", "Banner timeout", "Seconds for the SSH banner."),
        _Setting(
            "connection.keepalive", "Keepalive", "Seconds between keepalive packets; 0 disables."
        ),
        _Setting(
            "connection.multiplex", "Connection reuse", "Reuse one private OpenSSH ControlMaster."
        ),
        _Setting("connection.persist", "Idle reuse time", "Seconds before an idle master closes."),
        _Setting(
            "connection.command_timeout", "Command timeout", "Optional default command deadline."
        ),
    ),
    "Workspace and storage": (
        _Setting("workspace", "Workspace", "Absolute remote root used by LambdaForge."),
        _Setting(
            "project_root",
            "Project mirror",
            "Absolute remote mirror for large inputs and published outputs.",
        ),
        _Setting(
            "data_environment",
            "Data environment",
            "Stable name used to resolve dataset placements.",
        ),
        _Setting("storage.state_root", "State root", "Small durable control-plane records."),
        _Setting(
            "storage.cache_root",
            "Cache root",
            "Reconstructible bundles, runtimes and environments.",
        ),
        _Setting("storage.run_root", "Run root", "Attempt workspaces, logs and compact results."),
        _Setting("storage.dataset_root", "Dataset root", "Durable managed dataset storage."),
        _Setting(
            "storage.cache_max_size",
            "Cache size limit",
            "Maximum reconstructible cache size, e.g. 200GiB.",
        ),
        _Setting("storage.cache_max_age", "Cache age limit", "Maximum unused age, e.g. 30d."),
    ),
    "Python and packages": (
        _Setting(
            "environment",
            "Environment ownership",
            "managed creates immutable user-space environments; existing trusts one "
            "already installed.",
            (("Managed", "managed"), ("Existing", "existing")),
        ),
        _Setting(
            "python.strategy",
            "Python strategy",
            "auto finds or installs safely; existing and managed are strict choices.",
            (("Automatic", "auto"), ("Existing", "existing"), ("Managed", "managed")),
        ),
        _Setting("python.executable", "Python executable", "Name or absolute interpreter path."),
        _Setting("python.version", "Python version", "Optional exact minor/patch, such as 3.11."),
        _Setting(
            "python.allow_managed_install",
            "Allow managed Python",
            "Permit user-space micromamba fallback.",
        ),
        _Setting(
            "wheelhouse", "Wheelhouse", "Controller-local directory for fully offline installation."
        ),
        _Setting(
            "pytorch.channel",
            "PyTorch channel",
            "Automatic or explicit official CPU/CUDA wheel channel.",
            tuple((value, value) for value in sorted(TorchInstallationPolicy.CHANNELS)),
        ),
        _Setting("pytorch.require_cuda", "Require CUDA", "true, false or auto/null."),
        _Setting(
            "project_module", "Project import", "Optional module checked by doctor after bootstrap."
        ),
        _Setting(
            "command_prefix",
            "General command prefix",
            "YAML argv prepended to all remote commands, such as module wrappers.",
        ),
    ),
    "GPU access": (
        _Setting(
            "gpu_access.mode",
            "GPU access mode",
            "scheduler, exclusive, shared, or a site command wrapper.",
            tuple(
                (value, value) for value in ("auto", "scheduler", "exclusive", "shared", "command")
            ),
        ),
        _Setting(
            "gpu_access.command_prefix",
            "GPU command wrapper",
            "YAML argv placed before each GPU Work, e.g. [gpu, exec]. The managed Python path "
            "remains absolute.",
        ),
        _Setting(
            "gpu_access.claim_command",
            "GPU claim command",
            "Optional YAML argv run before each GPU submission; only {gpu_count} is expanded.",
        ),
        _Setting(
            "gpu_access.release_command",
            "GPU release command",
            "Required cleanup argv paired with an explicit claim, e.g. [gpu, release].",
        ),
    ),
    "Scheduler dialect": (
        _Setting(
            "scheduler",
            "Scheduler",
            "local means direct processes; slurm uses the configured dialect.",
        ),
        _Setting(
            "resource_mapping",
            "Resource mapping",
            "YAML mapping from portable resources to scheduler options.",
        ),
        _Setting(
            "scheduler_directives", "Static directives", "YAML mapping of fixed SLURM directives."
        ),
        _Setting(
            "scheduler_commands",
            "Scheduler commands",
            "YAML mapping overriding submit/queue/accounting/cancel commands.",
        ),
        _Setting("job_script", "Job script", "YAML mapping for shell, prologue and epilogue."),
        _Setting(
            "scheduler_options",
            "Legacy scheduler options",
            "Compatibility mapping visible in exported profiles.",
        ),
    ),
}


class ClusterWizard:
    """Ask explained questions, then invoke only native cluster subcommands."""

    def __init__(
        self,
        execute: NativeClusterCommand,
        *,
        catalog_path: Path | None = None,
        input_stream: TextIO | None = None,
        output: TextIO | None = None,
    ) -> None:
        self.execute = execute
        self.catalog_path = catalog_path
        self.input = input_stream or sys.stdin
        self.output = output or sys.stdout
        self.colour = TerminalTheme.enabled(self.output)

    def setup(self, *, offer_test: bool = True) -> int:
        """Run setup and turn an exit from any prompt into a clean CLI cancellation."""
        try:
            return self._setup(offer_test=offer_test)
        except (_WizardExit, KeyboardInterrupt):
            self._note("Cluster setup exited; no unconfirmed settings were written.")
            return 130

    def _setup(self, *, offer_test: bool = True) -> int:
        """Create a common profile first, with the complete editor one step away."""
        self._heading("CLUSTER SETUP", "A guided, secret-safe LambdaForge cluster profile")
        name = self._text(
            "Cluster name",
            "Short identifier used by --on, for example citius-ctgpgpu16.",
            required=True,
            validator=lambda value: bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value)),
        )
        host = self._text("Host or IP", "SSH login endpoint.", required=True)
        user = self._text(
            "SSH user", "Remote account name.", default=os.environ.get("USER") or getpass.getuser()
        )
        port = self._text("SSH port", "Usually 22.", default="22", validator=self._port)
        auth = self._choice(
            "Authentication",
            "OpenSSH reuses ~/.ssh/config, keys and agents. Password mode never writes "
            "the secret to YAML.",
            (
                (
                    "OpenSSH keys/agent",
                    "openssh",
                    "Reuse ~/.ssh/config, host aliases, keys and ssh-agent without storing a "
                    "new secret.",
                ),
                (
                    "Password",
                    "password",
                    "Resolve a password interactively, from the system keyring or by environment "
                    "reference; never store it in YAML.",
                ),
            ),
            default="openssh",
        )
        credential: str | None = None
        store_password = False
        if auth == "password":
            secret_mode = self._choice(
                "Password source",
                "Interactive asks on every new session; keyring stores it securely; "
                "environment stores only a variable reference.",
                (
                    (
                        "System keyring",
                        "keyring",
                        "Store the password through the OS keyring and resolve it only when "
                        "connecting.",
                    ),
                    (
                        "Ask when needed",
                        "interactive",
                        "Persist no credential; prompt when a new authenticated session is "
                        "required.",
                    ),
                    (
                        "Environment variable",
                        "env",
                        "Store only a variable name in the profile; the secret remains outside "
                        "LambdaForge state.",
                    ),
                ),
                default="keyring",
            )
            if secret_mode == "keyring":
                store_password = True
            elif secret_mode == "env":
                variable = self._text(
                    "Environment variable",
                    "Variable containing the password; its value is never read during setup.",
                    default="LAMBDAFORGE_SSH_PASSWORD",
                    validator=lambda value: bool(re.fullmatch(r"[A-Za-z_]\w*", value)),
                )
                credential = f"env:{variable}"
        scheduler = self._choice(
            "Execution backend",
            "This chooses how processes are launched, not whether they use a GPU. Direct runs "
            "processes on the host; SLURM submits through sbatch. GPU allocation is configured "
            "separately in the next GPU access step.",
            (
                (
                    "Direct processes",
                    "local",
                    "Launch durable supervised processes directly on the login/compute host. "
                    "Choose this when the site does not submit jobs through SLURM; GPU access "
                    "is still controlled separately below.",
                ),
                (
                    "SLURM",
                    "slurm",
                    "Submit jobs through sbatch and let SLURM allocate CPUs, memory, time and "
                    "GPUs. Choose this only when the cluster exposes a usable SLURM queue.",
                ),
            ),
            default="local",
        )
        workspace = self._text(
            "LambdaForge workspace",
            "Absolute remote root for managed state, caches and attempts.",
            default=f"/home/{user}",
            validator=self._absolute_remote,
        )
        project_root = self._text(
            "Remote project mirror",
            "Optional absolute counterpart of the local project for large inputs and "
            "published outputs.",
            default="",
            validator=self._optional_absolute_remote,
        )
        dataset_root = self._text(
            "Managed dataset root",
            "Optional durable location; keep it outside ephemeral job storage.",
            default="",
            validator=self._optional_absolute_remote,
        )
        environment = self._choice(
            "Python environment",
            "Managed is recommended and uses immutable user-space paths without shell activation.",
            (
                (
                    "Managed by LambdaForge",
                    "managed",
                    "Create/reuse an immutable user-space Python environment identified by exact "
                    "project dependencies.",
                ),
                (
                    "Existing installation",
                    "existing",
                    "Trust a pre-provisioned interpreter and packages; LambdaForge validates but "
                    "does not install them.",
                ),
            ),
            default="managed",
        )
        python_strategy = "auto" if environment == "managed" else "existing"
        python = self._text(
            "Python executable",
            "A name such as python3 or an absolute path. Managed mode can fall back to micromamba.",
            default="python3",
            required=True,
        )
        require_cuda = self._choice(
            "CUDA requirement",
            "Auto follows detected hardware; required fails closed when a real CUDA tensor "
            "cannot run.",
            (
                (
                    "Automatic",
                    "auto",
                    "Require CUDA when compatible NVIDIA hardware is detected; otherwise allow "
                    "CPU.",
                ),
                (
                    "Required",
                    "yes",
                    "Fail bootstrap/doctor unless PyTorch can execute a real CUDA tensor probe.",
                ),
                (
                    "CPU only / not required",
                    "no",
                    "Do not require CUDA usability; GPU Works will still fail if they later "
                    "request a GPU.",
                ),
            ),
            default="auto",
        )
        gpu_choices = (
            (
                (
                    "Automatic (SLURM allocation)",
                    "auto",
                    "Use the scheduler allocation. LambdaForge never rewrites "
                    "CUDA_VISIBLE_DEVICES and sees only the devices granted by SLURM.",
                ),
                (
                    "SLURM allocation",
                    "scheduler",
                    "Require every GPU Work to run inside the scheduler allocation; fail closed "
                    "if no scheduler-owned devices are visible.",
                ),
                (
                    "Site command wrapper",
                    "command",
                    "Prefix each GPU Work with a site launcher such as `gpu exec`. The wrapper "
                    "owns allocation and its CUDA visibility is preserved unchanged.",
                ),
            )
            if scheduler == "slurm"
            else (
                (
                    "Automatic (exclusive local leases)",
                    "auto",
                    "Use conservative per-user device leases and current free-memory admission. "
                    "External GPU processes are treated as unavailable capacity.",
                ),
                (
                    "Exclusive LambdaForge leases",
                    "exclusive",
                    "Require LambdaForge to hold an exclusive lease for every selected visible "
                    "device. Best for dedicated hosts shared by several LambdaForge Works.",
                ),
                (
                    "Shared host, external use allowed",
                    "shared",
                    "Allow admission beside external processes when the declared gpu_memory fits. "
                    "Use only when the site's sharing policy permits it; OOM risk is higher.",
                ),
                (
                    "Site claim/launcher command",
                    "command",
                    "Use a site-owned command such as `gpu exec`, optionally paired with claim "
                    "and release commands. LambdaForge preserves the granted CUDA visibility.",
                ),
            )
        )
        gpu_access = self._choice(
            "GPU access",
            "Choose command for sites where GPU work must run as `gpu COMMAND` or an "
            "equivalent wrapper.",
            gpu_choices,
            default="auto",
        )
        gpu_prefix: tuple[str, ...] = ()
        gpu_claim: tuple[str, ...] = ()
        gpu_release: tuple[str, ...] = ()
        if gpu_access == "command":
            gpu_prefix = self._argv(
                "GPU command wrapper",
                "Exact argv before the scientific command. For CITIUS gpuctl enter: gpu exec",
                default="gpu exec",
            )
            if scheduler == "local" and self._yes_no(
                "Run a claim command before each GPU submission?",
                "Enable only when claims are safe and persistent for your account. Otherwise "
                "use a self-contained wrapper such as `gpu exec`.",
                default=False,
            ):
                gpu_claim = self._argv(
                    "GPU claim command",
                    "Use {gpu_count} where the requested number belongs, for example: "
                    "gpu claim --numgpus {gpu_count}",
                    default="gpu claim --numgpus {gpu_count}",
                )
                gpu_release = self._argv(
                    "GPU release command",
                    "Always paired with the claim and run after success, failure or cancellation.",
                    default="gpu release",
                )
        scope = self._choice(
            "Catalog scope",
            "User profiles work from every project; project profiles stay beside this repository.",
            (
                (
                    "Current user",
                    "user",
                    "Save the profile in the user catalog so it is available from every project.",
                ),
                (
                    "Current project",
                    "project",
                    "Keep the profile in this project scope for repository-specific configuration.",
                ),
            ),
            default="user",
        )

        command = [
            "add",
            name,
            "--host",
            host,
            "--user",
            user,
            "--port",
            port,
            "--auth",
            auth,
            "--scope",
            scope,
            "--scheduler",
            scheduler,
            "--workspace",
            workspace,
            "--environment",
            environment,
            "--python",
            python,
            "--python-strategy",
            python_strategy,
            "--torch-channel",
            "auto",
            "--gpu-access",
            gpu_access,
        ]
        if credential:
            command.extend(("--credential", credential))
        if store_password:
            command.append("--store-password")
        if project_root:
            command.extend(("--project-root", project_root))
        if dataset_root:
            command.extend(("--dataset-root", dataset_root))
        if require_cuda == "yes":
            command.append("--require-cuda")
        elif require_cuda == "no":
            command.append("--no-require-cuda")
        if gpu_prefix:
            command.extend(("--gpu-command-prefix", *gpu_prefix))
        if gpu_claim:
            command.extend(("--gpu-claim-command", *gpu_claim))
            command.extend(("--gpu-release-command", *gpu_release))

        self._summary(
            (
                ("Name", name),
                ("Endpoint", f"{user}@{host}:{port}"),
                ("Scheduler", scheduler),
                ("Workspace", workspace),
                ("Environment", f"{environment} ({python_strategy}, {python})"),
                ("GPU access", gpu_access),
                ("Catalog", scope if self.catalog_path is None else str(self.catalog_path)),
            )
        )
        if not self._yes_no("Save this profile?", "No remote command has run yet.", default=True):
            self._note("Setup cancelled; nothing was written.")
            return 130
        self.execute(command)
        self._success(f"Cluster {name!r} was saved through `lf clusters add`.")
        if self._yes_no(
            "Open the complete settings editor?",
            "Every advanced SSH, storage, scheduler, runtime and GPU field is available there.",
            default=False,
        ):
            self._modify(name)
        if offer_test and self._yes_no(
            "Test the profile now?",
            "Runs the native read-only `lf clusters test` diagnostics.",
            default=True,
        ):
            return self.execute(("test", name))
        self._note(f"Next: lf clusters bootstrap {name} --project .")
        return 0

    def modify(self, name: str | None = None) -> int:
        """Run the editor and permit a clean exit from every nested prompt."""
        try:
            return self._modify(name)
        except (_WizardExit, KeyboardInterrupt):
            self._note("Cluster editor exited; the current prompt was not applied.")
            return 130

    def _modify(self, name: str | None = None) -> int:
        """Edit one existing profile through set/unset/credential subcommands."""
        catalog = ClusterCatalog.load(self.catalog_path)
        available = tuple(value for value in catalog.names() if value != "local")
        if not available:
            raise ValueError("No editable clusters exist. Run `lf clusters setup` first.")
        selected = name or self._choice(
            "Cluster",
            "Select the profile to inspect and modify.",
            tuple((value, value) for value in available),
            default=available[0],
        )
        if selected not in available:
            raise KeyError(f"Unknown editable cluster {selected!r}; configured: {available}.")
        while True:
            catalog = ClusterCatalog.load(self.catalog_path)
            inspected = catalog.inspect(selected)
            profile = inspected["profile"]
            self._heading(
                f"MODIFY {selected}",
                f"Source: {inspected['source']} · authentication: "
                f"{inspected['authentication_status']}",
            )
            actions = tuple((section, section) for section in _SETTINGS) + (
                ("Password credentials", "credentials"),
                ("Show effective profile", "show"),
                ("Test connection/runtime", "test"),
                ("Done", "done"),
            )
            action = self._choice("Section", "Choose what to inspect or change.", actions)
            if action == "done":
                return 0
            if action == "show":
                self._block(yaml.safe_dump(profile, sort_keys=False, allow_unicode=True).rstrip())
                continue
            if action == "test":
                self.execute(("test", selected))
                continue
            if action == "credentials":
                credential_action = self._choice(
                    "Password credential",
                    "The secret stays in the system keyring and never enters profile YAML.",
                    (
                        ("Store or replace", "set"),
                        ("Delete stored value", "delete"),
                        ("Back", "back"),
                    ),
                )
                if credential_action != "back":
                    self.execute(("credentials", credential_action, selected))
                continue
            if action == "GPU access":
                self._modify_gpu_policy(selected, profile)
                continue
            settings = _SETTINGS[str(action)]
            setting = self._choice(
                "Setting",
                "The effective current value is shown before editing.",
                tuple((item.label, item) for item in settings),
            )
            assert isinstance(setting, _Setting)
            current = self._dotted(profile, setting.key)
            self._note(
                f"{setting.key} = {yaml.safe_dump(current, default_flow_style=True).strip()}"
            )
            change = self._choice(
                setting.label,
                setting.explanation,
                (("Set a value", "set"), ("Reset/unset", "unset"), ("Back", "back")),
                default="set",
            )
            if change == "back":
                continue
            if change == "unset":
                if self._yes_no(
                    "Reset this setting?", "The profile default will apply when one exists."
                ):
                    self.execute(("unset", selected, setting.key))
                continue
            if setting.choices:
                value = self._choice(setting.label, setting.explanation, setting.choices)
            else:
                raw = self._text(
                    setting.label,
                    setting.explanation + " Enter YAML for lists/maps; plain text is accepted.",
                    default=self._editable_default(current),
                    required=True,
                )
                value = yaml.safe_load(raw)
            encoded = yaml.safe_dump(value, default_flow_style=True, sort_keys=False).strip()
            self.execute(("set", selected, setting.key, encoded))
            self._success(f"Updated {setting.key} through `lf clusters set`.")

    def _modify_gpu_policy(self, cluster: str, profile: Mapping[str, Any]) -> None:
        """Edit the interdependent GPU policy as one native atomic setting."""
        current = profile.get("gpu_access", {})
        current = current if isinstance(current, Mapping) else {}
        scheduler = str(profile.get("scheduler", "local"))
        modes = (
            (
                (
                    "Automatic",
                    "auto",
                    "Use the scheduler allocation and preserve its visible-device grant.",
                ),
                (
                    "Scheduler",
                    "scheduler",
                    "Require a scheduler-owned GPU allocation and fail closed without one.",
                ),
                (
                    "Command wrapper",
                    "command",
                    "Run each GPU Work through a site launcher and preserve its CUDA visibility.",
                ),
            )
            if scheduler == "slurm"
            else (
                (
                    "Automatic",
                    "auto",
                    "Use conservative exclusive leases and live free-memory admission.",
                ),
                (
                    "Exclusive leases",
                    "exclusive",
                    "Prevent other LambdaForge Works from sharing selected visible devices.",
                ),
                (
                    "Shared",
                    "shared",
                    "Permit external GPU occupancy only when the site's sharing policy allows it.",
                ),
                (
                    "Command wrapper",
                    "command",
                    "Use a site launcher such as gpu exec and preserve its device grant.",
                ),
            )
        )
        mode = self._choice(
            "GPU access mode",
            "The complete policy is validated and written atomically.",
            modes,
            default=str(current.get("mode", "auto")),
        )
        value: dict[str, Any] = {"mode": mode}
        if mode == "command":
            prefix = current.get("command_prefix", ("gpu", "exec"))
            prefix_default = " ".join(str(item) for item in prefix) or "gpu exec"
            value["command_prefix"] = list(
                self._argv(
                    "GPU command wrapper",
                    "For CITIUS gpuctl use `gpu exec`.",
                    default=prefix_default,
                )
            )
            current_claim = current.get("claim_command", ())
            if scheduler == "local" and self._yes_no(
                "Persistent claim and release?",
                "Use only when a separate account reservation is required.",
                default=bool(current_claim),
            ):
                claim_default = " ".join(str(item) for item in current_claim)
                release_default = " ".join(str(item) for item in current.get("release_command", ()))
                value["claim_command"] = list(
                    self._argv(
                        "Claim command",
                        "{gpu_count} is the only expanded placeholder.",
                        default=claim_default or "gpu claim --numgpus {gpu_count}",
                    )
                )
                value["release_command"] = list(
                    self._argv(
                        "Release command",
                        "Runs even after failure or cancellation.",
                        default=release_default or "gpu release",
                    )
                )
        rendered = yaml.safe_dump(value, default_flow_style=True, sort_keys=False).strip()
        self._block(yaml.safe_dump(value, sort_keys=False).rstrip())
        if self._yes_no(
            "Save this GPU policy?",
            "This invokes one `lf clusters set` operation.",
            default=True,
        ):
            self.execute(("set", cluster, "gpu_access", rendered))
            self._success("Updated the complete GPU policy atomically.")

    def _choice(
        self,
        label: str,
        explanation: str,
        choices: Sequence[tuple[Any, ...]],
        *,
        default: Any | None = None,
    ) -> Any:
        normalized = self._normalise_choices(choices, explanation=explanation)
        if self._supports_navigation():
            return self._arrow_choice(label, explanation, normalized, default=default)
        self._question(label, explanation)
        for index, (display, value, detail) in enumerate(normalized, 1):
            marker = " (default)" if value == default else ""
            self._write(f"    {index}. {display}{marker}")
            self._write(self._style(f"       {detail}", TerminalTheme.DIM))
        self._write("    0. Exit wizard")
        while True:
            raw = self._read("  Select: ").strip()
            if not raw and default is not None:
                return default
            if raw.isdigit() and 1 <= int(raw) <= len(normalized):
                return normalized[int(raw) - 1][1]
            for display, value, _detail in normalized:
                if raw.lower() in {str(value).lower(), display.lower()}:
                    return value
            self._warning("Choose one listed number or value.")

    @staticmethod
    def _normalise_choices(
        choices: Sequence[tuple[Any, ...]], *, explanation: str
    ) -> tuple[tuple[str, Any, str], ...]:
        """Give every selectable value contextual help without changing native commands."""
        output: list[tuple[str, Any, str]] = []
        for choice in choices:
            if len(choice) not in {2, 3}:
                raise ValueError("Wizard choices require display, value and optional help text.")
            display, value = str(choice[0]), choice[1]
            detail = str(choice[2]) if len(choice) == 3 else f"{explanation} Select {display}."
            output.append((display, value, detail))
        return tuple(output)

    def _supports_navigation(self) -> bool:
        """Use the richer selector only when both streams are real terminals."""
        try:
            importlib.import_module("termios")
            importlib.import_module("tty")
            return self.input.isatty() and self.output.isatty() and self.input.fileno() >= 0
        except (AttributeError, ImportError, OSError):
            return False

    def _arrow_choice(
        self,
        label: str,
        explanation: str,
        choices: Sequence[tuple[str, Any, str]],
        *,
        default: Any | None,
    ) -> Any:
        """Render an arrow/Enter selector whose focused option explains its consequences."""
        selected = next(
            (index for index, (_label, value, _detail) in enumerate(choices) if value == default),
            0,
        )
        termios = importlib.import_module("termios")
        tty = importlib.import_module("tty")
        descriptor = self.input.fileno()
        previous = termios.tcgetattr(descriptor)
        try:
            tty.setcbreak(descriptor)
            self.output.write("\x1b[?25l")
            while True:
                display, _value, detail = choices[selected]
                self.output.write("\x1b[2J\x1b[H")
                self.output.write(
                    self._style(f"LambdaForge cluster setup · {label}\n", TerminalTheme.BOLD_CYAN)
                )
                self.output.write(f"{explanation}\n\n")
                self.output.write(
                    self._style(
                        "↑/↓ or j/k move · Enter selects · q exits safely\n\n",
                        TerminalTheme.DIM,
                    )
                )
                for index, (option, value, _option_detail) in enumerate(choices):
                    prefix = "  › " if index == selected else "    "
                    suffix = "  default" if value == default else ""
                    style = TerminalTheme.BOLD_BLUE if index == selected else ""
                    line = f"{prefix}{option}{suffix}"
                    self.output.write(self._style(line, style) + "\n" if style else line + "\n")
                self.output.write("\n")
                self.output.write(self._style("What this option means\n", TerminalTheme.BOLD_CYAN))
                self.output.write(f"{detail}\n")
                self.output.flush()
                key = os.read(descriptor, 1)
                if key == b"\x1b":
                    sequence = key
                    for _ in range(2):
                        ready, _, _ = select.select([descriptor], [], [], 0.04)
                        if not ready:
                            break
                        sequence += os.read(descriptor, 1)
                    if sequence.endswith(b"A"):
                        selected = (selected - 1) % len(choices)
                    elif sequence.endswith(b"B"):
                        selected = (selected + 1) % len(choices)
                    continue
                if key in {b"\r", b"\n"}:
                    chosen = choices[selected]
                    break
                if key in {b"q", b"Q", b"0", b"\x03"}:
                    raise _WizardExit
                if key in {b"k", b"K"}:
                    selected = (selected - 1) % len(choices)
                elif key in {b"j", b"J"}:
                    selected = (selected + 1) % len(choices)
                elif key.isdigit() and b"1" <= key <= b"9":
                    index = int(key) - 1
                    if index < len(choices):
                        selected = index
        finally:
            termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)
            self.output.write("\x1b[?25h\x1b[2J\x1b[H")
            self.output.flush()
        self._note(f"{label}: {display}")
        return chosen[1]

    def _text(
        self,
        label: str,
        explanation: str,
        *,
        default: str | None = None,
        required: bool = False,
        validator: Callable[[str], bool] | None = None,
    ) -> str:
        self._question(label, explanation)
        while True:
            suffix = f" [{default}]" if default is not None and default != "" else ""
            raw = self._read(f"  {label}{suffix}: ").strip()
            value = default if not raw and default is not None else raw
            if required and not value:
                self._warning("A value is required.")
                continue
            if validator is not None and not validator(value or ""):
                self._warning("That value is not valid for this setting.")
                continue
            return value or ""

    def _argv(self, label: str, explanation: str, *, default: str) -> tuple[str, ...]:
        while True:
            raw = self._text(label, explanation, default=default, required=True)
            try:
                values = tuple(shlex.split(raw))
            except ValueError as error:
                self._warning(str(error))
                continue
            if values:
                return values

    def _yes_no(self, label: str, explanation: str, *, default: bool = False) -> bool:
        value = self._choice(
            label,
            explanation,
            (("Yes", True), ("No", False)),
            default=default,
        )
        return bool(value)

    def _heading(self, title: str, subtitle: str) -> None:
        self._write("")
        self._write(
            self._style(f"╭─ {title} " + "─" * max(2, 58 - len(title)), TerminalTheme.BOLD_CYAN)
        )
        self._write(f"│ {subtitle}")
        self._write("╰" + "─" * 62)

    def _question(self, label: str, explanation: str) -> None:
        self._write("")
        self._write(self._style(f"◆ {label}", TerminalTheme.BOLD_BLUE))
        self._write(f"  {explanation}")
        self._write(self._style("  0/q: exit wizard", TerminalTheme.DIM))

    def _summary(self, values: Sequence[tuple[str, str]]) -> None:
        self._heading("REVIEW", "Nothing is written until you confirm")
        width = max(len(label) for label, _ in values)
        for label, value in values:
            self._write(f"  {label:<{width}}  {value or '-'}")

    def _success(self, value: str) -> None:
        self._write(self._style(f"✓ {value}", TerminalTheme.GREEN))

    def _warning(self, value: str) -> None:
        self._write(self._style(f"! {value}", TerminalTheme.YELLOW))

    def _note(self, value: str) -> None:
        self._write(self._style(value, TerminalTheme.DIM))

    def _block(self, value: str) -> None:
        self._write("")
        for line in value.splitlines():
            self._write(f"  {line}")

    def _read(self, prompt: str) -> str:
        self.output.write(self._style(prompt, TerminalTheme.CYAN))
        self.output.flush()
        value = self.input.readline()
        if value == "":
            raise _WizardExit
        selected = value.rstrip("\r\n")
        if selected.strip().lower() in {"0", "q", "quit", "exit", ":q"}:
            raise _WizardExit
        return selected

    def _write(self, value: str) -> None:
        print(value, file=self.output, flush=True)

    def _style(self, value: str, style: str) -> str:
        return f"{style}{value}{TerminalTheme.RESET}" if self.colour else value

    @staticmethod
    def _port(value: str) -> bool:
        return value.isdigit() and 1 <= int(value) <= 65535

    @staticmethod
    def _absolute_remote(value: str) -> bool:
        return value.startswith("/") and value != "/" and "\n" not in value

    @classmethod
    def _optional_absolute_remote(cls, value: str) -> bool:
        return not value or cls._absolute_remote(value)

    @staticmethod
    def _dotted(value: Mapping[str, Any], key: str) -> Any:
        current: Any = value
        for component in key.split("."):
            if not isinstance(current, Mapping) or component not in current:
                return None
            current = current[component]
        return current

    @staticmethod
    def _editable_default(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return yaml.safe_dump(value, default_flow_style=True, sort_keys=False).strip()


__all__ = ["ClusterWizard", "NativeClusterCommand"]
