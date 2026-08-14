"""Durable scheduler for local or SSH hosts without a batch scheduler."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import cast

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ClusterStoragePolicy import ClusterStoragePolicy
from lambdaforge.controlplane.jobs import JobState
from lambdaforge.controlplane.Scheduler import Scheduler
from lambdaforge.controlplane.SchedulerCapabilities import SchedulerCapabilities
from lambdaforge.controlplane.SchedulerSubmission import SchedulerSubmission
from lambdaforge.controlplane.Transport import Transport
from lambdaforge.execution.ResourceRequest import ResourceRequest


class ProcessScheduler(Scheduler):
    """Launch one detached supervisor per job and reconnect through durable JSON state."""

    def __init__(self, transport: Transport, profile: ClusterProfile) -> None:
        self.transport = transport
        self.profile = profile
        self.storage = cast(ClusterStoragePolicy, profile.storage)

    @property
    def capabilities(self) -> SchedulerCapabilities:
        return SchedulerCapabilities(
            supports_pause=True,
            supports_resume=True,
            durable=True,
            resources_released_when_paused=False,
        )

    def submit(
        self,
        command: Sequence[str],
        resources: ResourceRequest,
        *,
        work_dir: str | Path,
        dry_run: bool = False,
        job_id: str | None = None,
    ) -> SchedulerSubmission:
        """Stage a request, detach its supervisor and return after acknowledgement."""
        if not command:
            raise ValueError("Process jobs require a command.")
        if job_id is None:
            raise ValueError("Durable process submission requires a LambdaForge job id.")
        job_dir = PurePosixPath(self.storage.job_root) / job_id
        scientific_work = job_dir / "work"
        request_path = job_dir / "request.json"
        python = self._control_python(command)
        launch = (
            *self.profile.command_prefix,
            python,
            "-m",
            "lambdaforge.controlplane.ProcessSupervisor",
            "launch",
            str(request_path),
        )
        if dry_run:
            return SchedulerSubmission(
                None,
                JobState.CREATED,
                command=launch,
                work_dir=str(scientific_work),
            )
        created = self.transport.run(("mkdir", "-p", str(job_dir)))
        if created.returncode:
            raise RuntimeError(f"Could not create durable job directory: {created.stderr.strip()}")
        request = {
            "process_request_version": 1,
            "job_id": job_id,
            "cluster": self.profile.name,
            "scheduler": "local",
            "command": [
                str(scientific_work) + str(value)[len(str(work_dir)) :]
                if self.profile.transport == "ssh" and str(value).startswith(str(work_dir))
                else str(value)
                for value in command
            ],
            "source_work_dir": str(work_dir),
            "stage_source": self.profile.transport == "ssh",
            "work_dir": str(scientific_work),
            "resources": resources.to_dict(),
            "lease_root": str(PurePosixPath(self.storage.state_root) / "gpu-leases"),
            "resource_lease_root": str(PurePosixPath(self.storage.state_root) / "process-leases"),
            "dataset_registry": str(PurePosixPath(self.storage.state_root) / "datasets.json"),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with tempfile.TemporaryDirectory(prefix="lambdaforge-process-request-") as temporary:
            local = Path(temporary) / "request.json"
            local.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.transport.put(local, str(request_path))
        launched = self.transport.run(launch, timeout=30.0)
        if launched.returncode:
            raise RuntimeError(f"Could not launch detached process supervisor: {launched.stderr}")
        return SchedulerSubmission(
            job_id,
            JobState.STAGING,
            stdout=launched.stdout,
            command=launch,
            work_dir=str(scientific_work),
        )

    def state(self, scheduler_id: str) -> JobState:
        value = self._state_payload(scheduler_id)
        return JobState(str(value.get("state", JobState.UNKNOWN.value)))

    def logs(self, scheduler_id: str, *, tail: int | None = None) -> str:
        job_dir = PurePosixPath(self.storage.job_root) / scheduler_id
        command = (
            ("tail", "-n", str(tail), str(job_dir / "stdout.log"), str(job_dir / "stderr.log"))
            if tail is not None
            else ("cat", str(job_dir / "stdout.log"), str(job_dir / "stderr.log"))
        )
        result = self.transport.run(command)
        return result.stdout if result.returncode == 0 else result.stderr

    def cancel(self, scheduler_id: str) -> None:
        self._control(scheduler_id, "cancel")

    def pause(self, scheduler_id: str) -> None:
        self._control(scheduler_id, "pause")

    def resume(self, scheduler_id: str) -> None:
        self._control(scheduler_id, "resume")

    def inventory(self) -> tuple[dict[str, object], ...]:
        command = (
            *self.profile.command_prefix,
            self._active_python(),
            "-m",
            "lambdaforge.controlplane.ProcessSupervisor",
            "inventory",
            self.storage.job_root,
        )
        result = self.transport.run(command, timeout=30.0)
        if result.returncode:
            raise RuntimeError(f"Could not read remote process inventory: {result.stderr.strip()}")
        value = json.loads(result.stdout or "[]")
        if not isinstance(value, list):
            raise TypeError("Process inventory response must be a list.")
        return tuple(item for item in value if isinstance(item, dict))

    def _control(self, scheduler_id: str, operation: str) -> None:
        job_dir = PurePosixPath(self.storage.job_root) / scheduler_id
        command = (
            *self.profile.command_prefix,
            self._active_python(),
            "-m",
            "lambdaforge.controlplane.ProcessSupervisor",
            "control",
            str(job_dir),
            operation,
        )
        result = self.transport.run(command, timeout=30.0)
        if result.returncode:
            raise RuntimeError(
                f"Process scheduler {operation} failed for {scheduler_id}: {result.stderr.strip()}"
            )

    def _state_payload(self, scheduler_id: str) -> dict[str, object]:
        path = PurePosixPath(self.storage.job_root) / scheduler_id / "state.json"
        result = self.transport.run(("cat", str(path)), timeout=15.0)
        if result.returncode:
            raise RuntimeError(
                f"Could not read durable state for {scheduler_id}: {result.stderr.strip()}"
            )
        value = json.loads(result.stdout)
        if not isinstance(value, dict) or value.get("job_id") != scheduler_id:
            raise RuntimeError("Remote process state does not match the requested job id.")
        return value

    def _active_python(self) -> str:
        if self.profile.environment != "managed":
            return self.profile.python
        pointer = PurePosixPath(self.storage.state_root) / "active-environment"
        result = self.transport.run(("cat", str(pointer)), timeout=15.0)
        if result.returncode or not result.stdout.strip():
            legacy = PurePosixPath(self.profile.workspace) / ".lambdaforge" / "active-environment"
            result = self.transport.run(("cat", str(legacy)), timeout=15.0)
        if result.returncode or not result.stdout.strip():
            raise RuntimeError(
                f"No managed environment is active on {self.profile.name}; run clusters bootstrap."
            )
        return result.stdout.strip()

    def _control_python(self, command: Sequence[str]) -> str:
        index = len(self.profile.command_prefix)
        if index < len(command) and command[index]:
            return str(command[index])
        return self._active_python()
