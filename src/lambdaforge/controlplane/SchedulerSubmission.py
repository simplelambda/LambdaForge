"""Scheduler-neutral submission response."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lambdaforge.controlplane.jobs import JobState


@dataclass(frozen=True, slots=True)
class SchedulerSubmission:
    """Return scheduler identity, state and generated script/log locations."""

    scheduler_id: str | None
    state: JobState
    script: Path | None = None
    stdout: str = ""
    stderr: str = ""
    command: tuple[str, ...] = ()
    directives: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    work_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a complete dry-run/audit payload without secrets."""
        return {
            "scheduler_id": self.scheduler_id,
            "state": self.state.value,
            "script": str(self.script) if self.script is not None else None,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "command": list(self.command),
            "directives": list(self.directives),
            "warnings": list(self.warnings),
            "work_dir": self.work_dir,
        }
