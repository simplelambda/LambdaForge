"""Transport-aware SLURM scheduler provider."""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from lambdaforge.controlplane.JobState import JobState
from lambdaforge.controlplane.Scheduler import Scheduler
from lambdaforge.controlplane.SchedulerSubmission import SchedulerSubmission
from lambdaforge.controlplane.Transport import Transport
from lambdaforge.execution.ResourceRequest import ResourceRequest


class SlurmScheduler(Scheduler):
    """Generate, submit and reconnect to SLURM jobs through any transport."""

    _ID = re.compile(r"^(\d+)(?:;.*)?$")

    def __init__(
        self, transport: Transport, *, options: Mapping[str, object] | None = None
    ) -> None:
        self.transport = transport
        self.options = dict(options or {})

    def submit(
        self,
        command: Sequence[str],
        resources: ResourceRequest,
        *,
        work_dir: str | Path,
        dry_run: bool = False,
    ) -> SchedulerSubmission:
        """Write a local preview script, stage it and optionally call sbatch."""
        directory = str(work_dir)
        script = (
            Path.cwd()
            / ".lambdaforge"
            / "control"
            / "scripts"
            / (re.sub(r"[^A-Za-z0-9_.-]", "-", directory) + ".sbatch")
        )
        script.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "#!/bin/bash",
            "set -euo pipefail",
            f"#SBATCH --ntasks={resources.processes}",
            f"#SBATCH --cpus-per-task={max(1, resources.cpu_cores // resources.processes)}",
            "#SBATCH --output=lambdaforge-%j.out",
            "#SBATCH --error=lambdaforge-%j.out",
        ]
        if resources.ram_bytes:
            lines.append(f"#SBATCH --mem={max(1, (resources.ram_bytes + 1048575) // 1048576)}M")
        if resources.gpu_count:
            lines.append(f"#SBATCH --gpus={resources.gpu_count}")
        if resources.runtime_seconds:
            lines.append(f"#SBATCH --time={max(1, int((resources.runtime_seconds + 59) // 60))}")
        for key, value in sorted(self.options.items()):
            if not re.fullmatch(r"[a-z][a-z0-9-]*", str(key)) or "\n" in str(value):
                raise ValueError(f"Unsafe SLURM option: {key!r}.")
            lines.append(f"#SBATCH --{key}={value}")
        lines.append("exec " + shlex.join(tuple(command)))
        script.write_text("\n".join(lines) + "\n", encoding="utf-8")
        remote_script = str(PurePosixPath(directory) / "submit.sbatch")
        if dry_run:
            return SchedulerSubmission(None, JobState.CREATED, script=script)
        mkdir = self.transport.run(("mkdir", "-p", directory))
        if mkdir.returncode:
            raise RuntimeError(f"Could not create remote workspace: {mkdir.stderr}")
        self.transport.put(script, remote_script)
        submitted = self.transport.run(("sbatch", "--parsable", remote_script), cwd=directory)
        if submitted.returncode:
            raise RuntimeError(f"SLURM submission failed: {submitted.stderr}")
        match = self._ID.fullmatch(submitted.stdout.strip())
        if match is None:
            raise RuntimeError(f"Could not parse SLURM job id: {submitted.stdout!r}")
        return SchedulerSubmission(match.group(1), JobState.QUEUED, script=script)

    def state(self, scheduler_id: str) -> JobState:
        """Map squeue/sacct state to the portable lifecycle."""
        self._validate_id(scheduler_id)
        queued = self.transport.run(("squeue", "-h", "-j", scheduler_id, "-o", "%T"))
        raw = queued.stdout.strip().splitlines()
        if queued.returncode == 0 and raw:
            return self._state(raw[0])
        accounting = self.transport.run(
            ("sacct", "-n", "-X", "-j", scheduler_id, "-o", "State", "--parsable2")
        )
        values = accounting.stdout.strip().splitlines()
        return self._state(values[0]) if accounting.returncode == 0 and values else JobState.UNKNOWN

    def logs(self, scheduler_id: str, *, tail: int | None = None) -> str:
        """Read the standard SLURM stdout file through the transport."""
        self._validate_id(scheduler_id)
        command = (
            ("tail", "-n", str(tail), f"lambdaforge-{scheduler_id}.out")
            if tail
            else ("cat", f"lambdaforge-{scheduler_id}.out")
        )
        result = self.transport.run(command)
        return result.stdout if result.returncode == 0 else result.stderr

    def cancel(self, scheduler_id: str) -> None:
        """Cancel one validated numeric SLURM job."""
        self._validate_id(scheduler_id)
        result = self.transport.run(("scancel", scheduler_id))
        if result.returncode:
            raise RuntimeError(f"SLURM cancellation failed: {result.stderr}")

    @staticmethod
    def _validate_id(value: str) -> None:
        if not value.isdigit():
            raise ValueError("SLURM job ids must be numeric.")

    @staticmethod
    def _state(value: str) -> JobState:
        state = value.strip().upper().split("+")[0]
        if state in {"PENDING", "CONFIGURING", "REQUEUED"}:
            return JobState.QUEUED
        if state in {"RUNNING", "COMPLETING", "SUSPENDED"}:
            return JobState.RUNNING
        if state in {"COMPLETED"}:
            return JobState.SUCCEEDED
        if state in {"CANCELLED", "PREEMPTED"}:
            return JobState.CANCELLED
        if state in {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL"}:
            return JobState.FAILED
        return JobState.UNKNOWN
