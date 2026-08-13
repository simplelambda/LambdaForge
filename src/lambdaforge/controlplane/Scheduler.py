"""Scheduler provider boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from lambdaforge.controlplane.JobState import JobState
from lambdaforge.controlplane.SchedulerCapabilities import SchedulerCapabilities
from lambdaforge.controlplane.SchedulerSubmission import SchedulerSubmission
from lambdaforge.execution.ResourceRequest import ResourceRequest


class Scheduler(ABC):
    """Submit, observe and control jobs through a portable contract."""

    @abstractmethod
    def submit(
        self,
        command: Sequence[str],
        resources: ResourceRequest,
        *,
        work_dir: str | Path,
        dry_run: bool = False,
        job_id: str | None = None,
    ) -> SchedulerSubmission:
        """Submit or preview one job."""

    @abstractmethod
    def state(self, scheduler_id: str) -> JobState:
        """Refresh one scheduler state."""

    @abstractmethod
    def logs(self, scheduler_id: str, *, tail: int | None = None) -> str:
        """Return scheduler-owned output where available."""

    @abstractmethod
    def cancel(self, scheduler_id: str) -> None:
        """Request cancellation of one scheduler job."""

    @property
    def capabilities(self) -> SchedulerCapabilities:
        """Return explicit optional lifecycle support."""
        return SchedulerCapabilities()

    def pause(self, scheduler_id: str) -> None:
        raise NotImplementedError("Pause is not supported by this scheduler.")

    def resume(self, scheduler_id: str) -> None:
        raise NotImplementedError("Resume is not supported by this scheduler.")

    def inventory(self) -> tuple[dict[str, object], ...]:
        """Return durable provider-owned LambdaForge jobs when discoverable."""
        return ()
