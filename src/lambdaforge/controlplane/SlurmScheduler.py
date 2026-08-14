"""Transport-aware configurable SLURM scheduler provider."""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from lambdaforge.controlplane.jobs import JobState
from lambdaforge.controlplane.Scheduler import Scheduler
from lambdaforge.controlplane.SchedulerCapabilities import SchedulerCapabilities
from lambdaforge.controlplane.SchedulerSubmission import SchedulerSubmission
from lambdaforge.controlplane.SlurmProfile import SlurmProfile
from lambdaforge.controlplane.Transport import Transport
from lambdaforge.execution.ResourceRequest import ResourceRequest


class SlurmScheduler(Scheduler):
    """Translate portable resources through one per-cluster SLURM dialect."""

    def __init__(
        self,
        transport: Transport,
        *,
        profile: SlurmProfile | None = None,
        options: Mapping[str, object] | None = None,
    ) -> None:
        self.transport = transport
        self.profile = profile or SlurmProfile.from_mapping(None, legacy_options=options)

    @property
    def capabilities(self) -> SchedulerCapabilities:
        return SchedulerCapabilities(
            supports_pause=self.profile.pause is not None,
            supports_resume=self.profile.resume is not None,
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
        """Create an auditable script and optionally stage/submit it."""
        del job_id
        if not command:
            raise ValueError("Scheduled commands cannot be empty.")
        directory = str(work_dir)
        script = (
            Path.cwd()
            / ".lambdaforge"
            / "control"
            / "scripts"
            / (re.sub(r"[^A-Za-z0-9_.-]", "-", directory) + ".sbatch")
        )
        script.parent.mkdir(parents=True, exist_ok=True)
        resource_directives, warnings = self.profile.resource_mapping.render(resources)
        directives = (*resource_directives, *self.profile.render_directives())
        lines = [
            f"#!{self.profile.shell}",
            "set -euo pipefail",
            *(f"#SBATCH {directive}" for directive in directives),
            *self.profile.prologue,
        ]
        rendered_command = shlex.join(tuple(str(item) for item in command))
        if self.profile.epilogue:
            lines.extend(
                (
                    "set +e",
                    rendered_command,
                    "_lambdaforge_status=$?",
                    "set -e",
                    *self.profile.epilogue,
                    'exit "$_lambdaforge_status"',
                )
            )
        else:
            lines.append(f"exec {rendered_command}")
        script.write_text("\n".join(lines) + "\n", encoding="utf-8")
        remote_script = str(PurePosixPath(directory) / "submit.sbatch")
        values = {"script": remote_script, "work_dir": directory}
        submit_command = self.profile.submit.render(values, allowed={"script", "work_dir"})
        if not any("{script}" in item for item in self.profile.submit.arguments):
            submit_command = (*submit_command, remote_script)
        preview = SchedulerSubmission(
            None,
            JobState.CREATED,
            script=script,
            command=submit_command,
            directives=directives,
            warnings=warnings,
            work_dir=directory,
        )
        if dry_run:
            return preview
        mkdir = self.transport.run(("mkdir", "-p", directory))
        if mkdir.returncode:
            raise RuntimeError(f"Could not create remote workspace: {mkdir.stderr}")
        self.transport.put(script, remote_script)
        submitted = self.transport.run(submit_command, cwd=directory)
        if submitted.returncode:
            raise RuntimeError(f"SLURM submission failed: {submitted.stderr}")
        pattern = self.profile.submit.job_id_pattern
        if pattern is None:
            raise RuntimeError("SLURM submit command requires job_id_pattern.")
        match = re.fullmatch(pattern, submitted.stdout.strip())
        if match is None:
            raise RuntimeError(
                "Could not parse SLURM job id with the configured job_id_pattern: "
                f"{submitted.stdout!r}"
            )
        scheduler_id = match.group(1)
        self._validate_id(scheduler_id)
        return SchedulerSubmission(
            scheduler_id,
            JobState.QUEUED,
            script=script,
            command=submit_command,
            directives=directives,
            warnings=warnings,
            work_dir=directory,
        )

    def state(self, scheduler_id: str) -> JobState:
        """Query configured queue then accounting commands."""
        self._validate_id(scheduler_id)
        values = {"job_id": scheduler_id}
        queued_command = self.profile.queue.render(values, allowed={"job_id"})
        queued = self.transport.run(queued_command)
        raw = queued.stdout.strip().splitlines()
        if queued.returncode == 0 and raw:
            return self._state(raw[0])
        accounting_command = self.profile.accounting.render(values, allowed={"job_id"})
        accounting = self.transport.run(accounting_command)
        rows = accounting.stdout.strip().splitlines()
        return self._state(rows[0]) if accounting.returncode == 0 and rows else JobState.UNKNOWN

    def logs(self, scheduler_id: str, *, tail: int | None = None) -> str:
        """Read the conventional output file through the transport."""
        self._validate_id(scheduler_id)
        command = (
            ("tail", "-n", str(tail), f"lambdaforge-{scheduler_id}.out")
            if tail
            else ("cat", f"lambdaforge-{scheduler_id}.out")
        )
        result = self.transport.run(command)
        return result.stdout if result.returncode == 0 else result.stderr

    def cancel(self, scheduler_id: str) -> None:
        """Cancel one validated job with the configured command."""
        self._validate_id(scheduler_id)
        command = self.profile.cancel.render({"job_id": scheduler_id}, allowed={"job_id"})
        result = self.transport.run(command)
        if result.returncode:
            raise RuntimeError(f"SLURM cancellation failed: {result.stderr}")

    def pause(self, scheduler_id: str) -> None:
        self._optional_control(scheduler_id, "pause")

    def resume(self, scheduler_id: str) -> None:
        self._optional_control(scheduler_id, "resume")

    def _optional_control(self, scheduler_id: str, operation: str) -> None:
        self._validate_id(scheduler_id)
        descriptor = getattr(self.profile, operation)
        if descriptor is None:
            raise NotImplementedError(f"{operation.title()} is not supported by this scheduler.")
        command = descriptor.render({"job_id": scheduler_id}, allowed={"job_id"})
        result = self.transport.run(command, timeout=30.0)
        if result.returncode:
            raise RuntimeError(f"SLURM {operation} failed: {result.stderr}")

    @staticmethod
    def _validate_id(value: str) -> None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", value) is None:
            raise ValueError("Scheduler job id contains unsafe characters.")

    @staticmethod
    def _state(value: str) -> JobState:
        state = value.strip().upper().split("+")[0].split("|")[0].strip()
        if state in {"PENDING", "CONFIGURING", "REQUEUED"}:
            return JobState.QUEUED
        if state in {"RUNNING", "COMPLETING"}:
            return JobState.RUNNING
        if state == "SUSPENDED":
            return JobState.PAUSED
        if state == "COMPLETED":
            return JobState.SUCCEEDED
        if state in {"CANCELLED", "PREEMPTED"}:
            return JobState.CANCELLED
        if state == "TIMEOUT":
            return JobState.TIMEOUT
        if state in {"FAILED", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL"}:
            return JobState.FAILED
        return JobState.UNKNOWN
