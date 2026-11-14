"""Durable scheduler for local or SSH hosts without a batch scheduler."""

from __future__ import annotations

import ast
import inspect
import json
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, cast

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ClusterStoragePolicy import ClusterStoragePolicy
from lambdaforge.controlplane.jobs import JobState
from lambdaforge.controlplane.NativeLibraryPolicy import NativeLibraryPolicy
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
        execution_queue: Mapping[str, Any] | None = None,
        stdin_data: bytes | None = None,
    ) -> SchedulerSubmission:
        """Stage a request, detach its supervisor and return after acknowledgement."""
        if not command:
            raise ValueError("Process jobs require a command.")
        if stdin_data is not None and (
            not isinstance(stdin_data, bytes) or not 0 < len(stdin_data) <= 4096
        ):
            raise ValueError("Ephemeral Job input must be 1–4096 bytes; never use argv/metadata.")
        if job_id is None:
            raise ValueError("Durable process submission requires a LambdaForge job id.")
        job_dir = PurePosixPath(self.storage.job_root) / job_id
        scientific_work = job_dir / "work"
        source_work = PurePosixPath(str(work_dir))
        request_path = job_dir / "request.json"
        if execution_queue is not None:
            from lambdaforge.controlplane.ExecutionQueue import ExecutionQueue

            ExecutionQueue.check_id(job_id)
            if execution_queue.get("version") != 1 or execution_queue.get("job_id") != job_id:
                raise ValueError("Queue admission belongs to a different Job.")
        python = self._control_python(command)
        library_policy = NativeLibraryPolicy.from_command(command)
        control_prefix = (
            *self.profile.command_prefix,
            *(library_policy.launch_prefix(self.profile.python) if library_policy else ()),
        )
        launch = (
            *control_prefix,
            python,
            "-m",
            "lambdaforge.controlplane.ProcessSupervisor",
            "launch-input" if stdin_data is not None else "launch",
            str(request_path),
        )
        if dry_run:
            return SchedulerSubmission(
                None,
                JobState.CREATED,
                command=launch,
                work_dir=str(scientific_work),
            )
        if stdin_data is not None:
            supported = self.transport.run(
                (
                    *control_prefix,
                    python,
                    "-c",
                    "import inspect;from lambdaforge.controlplane.ProcessSupervisor "
                    "import ProcessSupervisor;"
                    "p=inspect.signature(ProcessSupervisor.launch).parameters;"
                    "raise SystemExit(0 if 'stdin_data' in p else 2)",
                ),
                timeout=30,
            )
            if supported.returncode:
                raise ValueError(
                    "Target LF runtime lacks ephemeral Job input. "
                    "Update/bootstrap it before password transfers."
                )
        if execution_queue is not None:
            supported = self.transport.run(
                (
                    *control_prefix,
                    python,
                    "-c",
                    "from lambdaforge.controlplane.ProcessSupervisor import ProcessSupervisor;"
                    "raise SystemExit(0 if callable(getattr(ProcessSupervisor,"
                    "'_wait_for_execution_queue',None)) else 2)",
                ),
                timeout=30,
            )
            if supported.returncode:
                raise RuntimeError(
                    "The target LambdaForge runtime cannot honor sequential execution admission. "
                    "Bootstrap the updated managed runtime, or update the declared existing "
                    "environment before submitting --queue. No scientific process was launched."
                )
        created = self.transport.run(("mkdir", "-p", str(job_dir)))
        if created.returncode:
            raise RuntimeError(f"Could not create durable job directory: {created.stderr.strip()}")
        if execution_queue is not None:
            guard = self.transport.run(
                (
                    *self.profile.command_prefix,
                    python,
                    "-c",
                    "import os,pathlib,sys;root=pathlib.Path(sys.argv[1]);"
                    "\nif any(p.is_symlink() for p in (root,*root.parents)):"
                    "raise ValueError('Unsafe queued Job workspace')\n"
                    "fd=os.open(root/'.queue-provider-submitted',"
                    "os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);os.close(fd)",
                    str(job_dir),
                )
            )
            if guard.returncode:
                raise RuntimeError(
                    "Queued Job dispatch is already reserved or unsafe; "
                    "inspect its native Job before retrying. No duplicate launched."
                )
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
            "resource_safety": self.profile.resource_safety.to_dict(),
            "storage": self.storage.to_dict(),
            "cache_root": self.storage.cache_root,
            "product_root": self._product_root(work_dir),
            "lease_root": str(
                PurePosixPath(self.storage.lease_root or self.storage.state_root) / "gpu-leases"
            ),
            "resource_lease_root": str(
                PurePosixPath(self.storage.lease_root or self.storage.state_root) / "process-leases"
            ),
            "dataset_registry": self._dataset_registry(work_dir),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            **({"execution_queue": dict(execution_queue)} if execution_queue else {}),
        }
        with tempfile.TemporaryDirectory(prefix="lambdaforge-process-request-") as temporary:
            local = Path(temporary) / "request.json"
            local.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.transport.put(local, str(request_path))
        if stdin_data is None:
            launched = self.transport.run(launch, timeout=30.0)
        else:
            from lambdaforge.controlplane.CommandResult import CommandResult

            with self.transport.stream(launch) as stream:
                stream.stdin.write(stdin_data)
                stream.finish_input()
                response = stream.stdout.read(65537)
                stream.wait()
                if len(response) > 65536:
                    raise RuntimeError("Supervisor acknowledgement exceeds its metadata bound.")
                launched = CommandResult(0, response.decode(), "")
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

    def _product_root(self, work_dir: str | Path) -> str:
        if self.profile.transport == "local":
            from lambdaforge.ProjectContext import ProjectContext

            return str(ProjectContext.discover(work_dir).root / ".lambdaforge" / "products")
        return self.storage.product_root

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
            "-c",
            self._observation_code("inventory"),
            self.storage.job_root,
        )
        result = self.transport.run(command, timeout=30.0)
        if result.returncode:
            raise RuntimeError(f"Could not read remote process inventory: {result.stderr.strip()}")
        value = json.loads(result.stdout)
        if not isinstance(value, list):
            raise TypeError("Process inventory response must be a list.")
        if any(not isinstance(item, dict) for item in value):
            raise TypeError(
                "Process inventory contains a malformed record; inventory is incomplete."
            )
        return tuple(value)

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
        if not scheduler_id.startswith("job-") or PurePosixPath(scheduler_id).name != scheduler_id:
            raise ValueError("Unsafe process scheduler id.")
        path = PurePosixPath(self.storage.job_root) / scheduler_id
        result = self.transport.run(
            (
                *self.profile.command_prefix,
                self._active_python(),
                "-c",
                self._observation_code("observe"),
                str(path),
            ),
            timeout=15.0,
        )
        if result.returncode:
            raise RuntimeError(
                f"Could not read durable state for {scheduler_id}: {result.stderr.strip()}"
            )
        value = json.loads(result.stdout)
        if not isinstance(value, dict) or value.get("job_id") != scheduler_id:
            raise RuntimeError("Remote process state does not match the requested job id.")
        return value

    @staticmethod
    def _observation_code(operation: str) -> str:
        """Ship a bounded read-only probe to old immutable runtimes, never install into them.

        The implementation is the same host authority used by the native supervisor. Only
        process metadata is read, no consumer code, CUDA initialization or scientific bytes.
        """
        from lambdaforge.controlplane.ProcessIdentity import ProcessIdentity
        from lambdaforge.controlplane.ProcessSupervisor import ProcessSupervisor
        from lambdaforge.controlplane.ResearchWork import study_overview

        # inspect.getsource(function) trusts the loaded code's line offsets. An editable
        # installation can change those offsets while a Console/monitor is still open,
        # accidentally shipping another method (and losing _required_mapping). Select
        # definitions and their complete helper closure from one current file snapshot.
        source = Path(inspect.getfile(ProcessSupervisor)).read_text(encoding="utf-8")
        supervisor = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.ClassDef) and node.name == "ProcessSupervisor"
        )
        definitions = {
            node.name: node for node in supervisor.body if isinstance(node, ast.FunctionDef)
        }
        required = {"observe", "inventory"}
        pending = list(required)
        while pending:
            name = pending.pop()
            for node in ast.walk(definitions[name]):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "cls"
                    and node.attr in definitions
                    and node.attr not in required
                ):
                    required.add(node.attr)
                    pending.append(node.attr)
        lines = source.splitlines(keepends=True)
        methods = []
        for name, node in definitions.items():
            if name in required:
                start = min([node.lineno, *(item.lineno for item in node.decorator_list)])
                methods.append("".join(lines[start - 1 : node.end_lineno]))
        overview_source = Path(inspect.getfile(study_overview)).read_text(encoding="utf-8")
        overview = next(
            node
            for node in ast.parse(overview_source).body
            if isinstance(node, ast.FunctionDef) and node.name == "study_overview"
        )
        overview_code = "".join(
            overview_source.splitlines(keepends=True)[overview.lineno - 1 : overview.end_lineno]
        )
        return (
            "from __future__ import annotations\n"
            "import os,json,sys,copy\nfrom pathlib import Path\nfrom enum import Enum\n"
            "from typing import Any\nfrom collections.abc import Mapping,Sequence\n"
            + inspect.getsource(JobState)
            + "\n"
            + overview_code
            + "\n"
            + f"exec({Path(inspect.getfile(ProcessIdentity)).read_text(encoding='utf-8')!r})\n"
            "class _Observer:\n"
            + "\n".join(methods)
            + f"\nprint(json.dumps(_Observer.{operation}(sys.argv[1])))\n"
        )

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
