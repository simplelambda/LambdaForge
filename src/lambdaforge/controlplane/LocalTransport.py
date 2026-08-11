"""Local control-plane transport."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from lambdaforge.controlplane.CommandResult import CommandResult
from lambdaforge.controlplane.Transport import Transport


class LocalTransport(Transport):
    """Execute and stage through the local filesystem."""

    def run(self, command: Sequence[str], *, cwd: str | Path | None = None) -> CommandResult:
        """Execute one validated argument vector synchronously."""
        if not command or any(not isinstance(item, str) or not item for item in command):
            raise ValueError("Transport commands require non-empty string arguments.")
        completed = subprocess.run(
            tuple(command),
            cwd=Path(cwd) if cwd is not None else None,
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)

    def put(self, source: str | Path, destination: str | Path) -> None:
        """Copy one explicit source, replacing only the exact destination."""
        source_path = Path(source).resolve()
        destination_path = Path(destination).resolve()
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.is_dir():
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, destination_path)
