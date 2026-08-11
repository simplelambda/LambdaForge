"""Synchronous local/SSH scheduler adapter."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from lambdaforge.controlplane.JobState import JobState
from lambdaforge.controlplane.Scheduler import Scheduler
from lambdaforge.controlplane.SchedulerSubmission import SchedulerSubmission
from lambdaforge.controlplane.Transport import Transport
from lambdaforge.execution.ResourceRequest import ResourceRequest


class LocalScheduler(Scheduler):
    """Run a command synchronously through the selected transport."""

    def __init__(self, transport: Transport) -> None:
        self.transport = transport
        self._states: dict[str, JobState] = {}
        self._logs: dict[str, str] = {}
        self._next_id = 1

    def submit(
        self,
        command: Sequence[str],
        resources: ResourceRequest,
        *,
        work_dir: str | Path,
        dry_run: bool = False,
    ) -> SchedulerSubmission:
        """Preview or run one command to completion."""
        del resources
        if dry_run:
            return SchedulerSubmission(None, JobState.CREATED)
        scheduler_id = f"local-{self._next_id}"
        self._next_id += 1
        result = self.transport.run(command, cwd=work_dir)
        state = JobState.SUCCEEDED if result.returncode == 0 else JobState.FAILED
        self._states[scheduler_id] = state
        self._logs[scheduler_id] = result.stdout + result.stderr
        return SchedulerSubmission(
            scheduler_id,
            state,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def state(self, scheduler_id: str) -> JobState:
        """Return the terminal state retained by this process."""
        return self._states.get(scheduler_id, JobState.UNKNOWN)

    def logs(self, scheduler_id: str, *, tail: int | None = None) -> str:
        """Return captured output, optionally restricted to final lines."""
        value = self._logs.get(scheduler_id, "")
        return "\n".join(value.splitlines()[-tail:]) if tail is not None else value

    def cancel(self, scheduler_id: str) -> None:
        """Reject cancellation after synchronous completion."""
        if self.state(scheduler_id).terminal:
            return
        self._states[scheduler_id] = JobState.CANCELLED
