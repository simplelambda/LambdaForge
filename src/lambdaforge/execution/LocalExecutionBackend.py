"""Local subprocess execution backend."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

from lambdaforge.execution.BackendSubmission import BackendSubmission
from lambdaforge.execution.ExecutionBackend import ExecutionBackend
from lambdaforge.execution.ResourceRequest import ResourceRequest


class LocalExecutionBackend(ExecutionBackend):
    """Run an argument-vector command locally when execution is explicit."""

    def submit(
        self,
        command: Sequence[str],
        resources: ResourceRequest,
        *,
        work_dir: str | Path,
        dry_run: bool = True,
    ) -> BackendSubmission:
        """Validate the plan and optionally execute it synchronously."""
        del resources
        if not command or any(not isinstance(item, str) or not item for item in command):
            raise ValueError("Backend commands require non-empty string arguments.")
        directory = Path(work_dir).resolve()
        if dry_run:
            return BackendSubmission("local")
        completed = subprocess.run(tuple(command), cwd=directory, check=False, shell=False)
        if completed.returncode:
            raise RuntimeError(
                f"Local backend command failed with exit code {completed.returncode}."
            )
        return BackendSubmission("local", submitted=True)
