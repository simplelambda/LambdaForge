"""Scheduler-neutral submission response."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lambdaforge.controlplane.JobState import JobState


@dataclass(frozen=True, slots=True)
class SchedulerSubmission:
    """Return scheduler identity, state and generated script/log locations."""

    scheduler_id: str | None
    state: JobState
    script: Path | None = None
    stdout: str = ""
    stderr: str = ""
