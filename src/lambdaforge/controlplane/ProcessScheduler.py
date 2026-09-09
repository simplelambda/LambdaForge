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
        self._last_state: dict[str, dict[str, object]] = {}

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
        source_work = PurePosixPath(str(work_dir))
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
            "source_work_dir": str(source_work),
            # ControlPlane already gives remote Jobs a mutable workspace copied from the
            # immutable bundle.  Only stage when a caller supplies a genuinely different
            # source directory; never copy ``job/work`` onto itself.
            "stage_source": (self.profile.transport == "ssh" and source_work != scientific_work),
            "work_dir": str(scientific_work),
            "resources": resources.to_dict(),
            "gpu_access": self.profile.gpu_access.to_dict(),
            "storage": self.storage.to_dict(),
            "cache_root": self.storage.cache_root,
            "lease_root": str(
                PurePosixPath(self.storage.lease_root or self.storage.state_root) / "gpu-leases"
            ),
            "resource_lease_root": str(
                PurePosixPath(self.storage.lease_root or self.storage.state_root) / "process-leases"
            ),
            "dataset_registry": self._dataset_registry(work_dir),
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

    def _dataset_registry(self, work_dir: str | Path) -> str:
        if self.profile.transport == "local":
            from lambdaforge.data.DatasetRegistry import DatasetRegistry

            return str(DatasetRegistry.project_path(work_dir))
        return str(PurePosixPath(self.storage.state_root) / "datasets.json")

    def state(self, scheduler_id: str) -> JobState:
        value = self._state_payload(scheduler_id)
        self._last_state[scheduler_id] = value
        return JobState(str(value.get("state", JobState.UNKNOWN.value)))

    def details(self, scheduler_id: str) -> dict[str, object]:
        """Return the durable supervisor state, reusing the current refresh read."""
        return dict(self._last_state.get(scheduler_id) or self._state_payload(scheduler_id))

    def logs(self, scheduler_id: str, *, tail: int | None = None) -> str:
        job_dir = PurePosixPath(self.storage.job_root) / scheduler_id
        streams: list[str] = []
        errors: list[str] = []
        for name in ("stdout.log", "stderr.log"):
            path = str(job_dir / name)
            if self.transport.run(("test", "-f", path), timeout=15.0).returncode != 0:
                # Pre-0.12.0 supervisors could fail before creating the stream.  Absence means
                # no scientific output, not a replacement error from ``cat``.
                continue
            command = ("tail", "-n", str(tail), path) if tail is not None else ("cat", path)
            result = self.transport.run(command, timeout=30.0)
            if result.returncode == 0:
                streams.append(result.stdout)
            elif result.stderr.strip():
                errors.append(result.stderr.strip())
        if streams:
            return "".join(streams)
        return "\n".join(errors)

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
        # ControlPlane may wrap the scientific command in ``env NAME=value ...``
        # (notably for dataset builds), after an optional site command prefix.  The
        # interpreter is therefore not at a fixed offset.  Prefer the exact Python
        # that owns the LambdaForge ``-m`` entry point so retries keep using the
        # environment recorded in their original command.
        for index in range(1, len(command) - 1):
            if command[index] != "-m":
                continue
            module = str(command[index + 1])
            candidate = str(command[index - 1])
            if (module == "lambdaforge" or module.startswith("lambdaforge.")) and not (
                candidate.startswith("-")
            ):
                return candidate
        return self._active_python()
