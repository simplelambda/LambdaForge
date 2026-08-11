"""OpenSSH-based control-plane transport."""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path

from lambdaforge.controlplane.CommandResult import CommandResult
from lambdaforge.controlplane.Transport import Transport


class SshTransport(Transport):
    """Use the user's audited OpenSSH configuration and host-key policy."""

    def __init__(self, host: str, *, options: Sequence[str] = ()) -> None:
        if not host.strip() or host.startswith("-") or "\n" in host:
            raise ValueError("SSH host must be a non-empty configured host name.")
        if any("\n" in item for item in options):
            raise ValueError("SSH options cannot contain newlines.")
        self.host = host
        self.options = tuple(options)

    def run(self, command: Sequence[str], *, cwd: str | Path | None = None) -> CommandResult:
        """Execute an argument vector remotely through one quoted command string."""
        if not command:
            raise ValueError("SSH commands cannot be empty.")
        remote = shlex.join(tuple(command))
        if cwd is not None:
            remote = f"cd {shlex.quote(str(cwd))} && exec {remote}"
        completed = subprocess.run(
            ("ssh", *self.options, self.host, "--", remote),
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)

    def put(self, source: str | Path, destination: str | Path) -> None:
        """Stage one explicit small bundle with OpenSSH scp."""
        source_path = Path(source).resolve()
        arguments = ["scp", *self.options]
        if source_path.is_dir():
            arguments.append("-r")
        arguments.extend((str(source_path), f"{self.host}:{destination}"))
        completed = subprocess.run(tuple(arguments), check=False, shell=False)
        if completed.returncode:
            raise RuntimeError(f"scp failed with exit code {completed.returncode}.")

    def get(self, source: str | Path, destination: str | Path) -> None:
        """Retrieve one explicit remote path through OpenSSH scp."""
        destination_path = Path(destination).resolve()
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            ("scp", *self.options, f"{self.host}:{source}", str(destination_path)),
            check=False,
            shell=False,
        )
        if completed.returncode:
            raise RuntimeError(f"scp retrieval failed with exit code {completed.returncode}.")
