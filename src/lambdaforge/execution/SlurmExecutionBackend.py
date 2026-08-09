"""Preview-first SLURM execution backend."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from uuid import uuid4

from lambdaforge.execution.BackendSubmission import BackendSubmission
from lambdaforge.execution.ExecutionBackend import ExecutionBackend
from lambdaforge.execution.ResourceRequest import ResourceRequest


class SlurmExecutionBackend(ExecutionBackend):
    """Generate safe ``sbatch`` scripts and submit only on explicit request."""

    _JOB_ID = re.compile(r"(?:Submitted batch job\s+)?(\d+)")

    def __init__(
        self,
        *,
        partition: str | None = None,
        nodes: int = 1,
        array: str | None = None,
        dependency: str | None = None,
        container_command: Sequence[str] = (),
        environment: Mapping[str, str] | None = None,
        requeue: bool = False,
    ) -> None:
        if nodes < 1:
            raise ValueError("SLURM nodes must be positive.")
        self.partition = partition
        self.nodes = nodes
        self.array = array
        self.dependency = dependency
        self.container_command = tuple(container_command)
        self.environment = dict(environment or {})
        self.requeue = requeue

    def submit(
        self,
        command: Sequence[str],
        resources: ResourceRequest,
        *,
        work_dir: str | Path,
        dry_run: bool = True,
    ) -> BackendSubmission:
        """Write the exact script; call ``sbatch --parsable`` only when requested."""
        if not command:
            raise ValueError("SLURM command cannot be empty.")
        directory = Path(work_dir).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        script = directory / "submit.sbatch"
        lines = [
            "#!/bin/bash",
            "set -euo pipefail",
            f"#SBATCH --nodes={self.nodes}",
            f"#SBATCH --cpus-per-task={resources.cpu_cores}",
        ]
        if resources.ram_bytes:
            lines.append(f"#SBATCH --mem={max(1, (resources.ram_bytes + 1048575) // 1048576)}M")
        if resources.gpu_count:
            lines.append(f"#SBATCH --gpus={resources.gpu_count}")
        if resources.runtime_seconds:
            minutes = max(1, int((resources.runtime_seconds + 59) // 60))
            lines.append(f"#SBATCH --time={minutes}")
        for flag, value in (
            ("partition", self.partition),
            ("array", self.array),
            ("dependency", self.dependency),
        ):
            if value:
                if "\n" in value:
                    raise ValueError(f"Unsafe SLURM {flag} value.")
                lines.append(f"#SBATCH --{flag}={value}")
        if self.requeue:
            lines.append("#SBATCH --requeue")
        for key, value in sorted(self.environment.items()):
            if not key.isidentifier():
                raise ValueError(f"Unsafe environment variable name: {key!r}.")
            lines.append(f"export {key}={shlex.quote(value)}")
        lines.append("exec " + shlex.join((*self.container_command, *command)))
        temporary = script.with_name(f".{script.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
            os.replace(temporary, script)
        finally:
            temporary.unlink(missing_ok=True)
        if dry_run:
            return BackendSubmission("slurm", artifact=script)
        completed = subprocess.run(
            ("sbatch", "--parsable", str(script)),
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        )
        match = self._JOB_ID.search(completed.stdout.strip())
        if match is None:
            raise RuntimeError(f"Could not parse SLURM job id: {completed.stdout!r}")
        return BackendSubmission("slurm", artifact=script, job_id=match.group(1), submitted=True)

    @staticmethod
    def cancel(job_id: str) -> None:
        """Cancel one validated numeric job id."""
        if not job_id.isdigit():
            raise ValueError("SLURM job ids must be numeric.")
        subprocess.run(("scancel", job_id), check=True, shell=False)

    @staticmethod
    def requeue_job(job_id: str) -> None:
        """Requeue one validated numeric job id."""
        if not job_id.isdigit():
            raise ValueError("SLURM job ids must be numeric.")
        subprocess.run(("scontrol", "requeue", job_id), check=True, shell=False)
