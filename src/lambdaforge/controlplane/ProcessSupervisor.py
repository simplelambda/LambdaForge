"""Detached per-job supervisor for direct local and SSH execution."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from lambdaforge.controlplane.jobs import JobState
from lambdaforge.controlplane.ProcessIdentity import ProcessIdentity
from lambdaforge.controlplane.ResearchWork import study_overview
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.training.orchestration.ProcessGuard import ProcessGuard


class ProcessSupervisor:
    """Own one durable process job without requiring a resident global daemon."""

    POLL_SECONDS = 0.5
    HEARTBEAT_SECONDS = 5.0
    _lease_wait_reason: str | None = None

    @classmethod
    def launch(cls, request_path: str | Path, *, stdin_data: bytes | None = None) -> dict[str, Any]:
        """Detach a supervisor, wait for durable acknowledgement and return quickly."""
        if stdin_data is not None and (
            not isinstance(stdin_data, bytes) or not 0 < len(stdin_data) <= 4096
        ):
            raise ValueError("Ephemeral Job input must be 1–4096 bytes.")
        request = Path(request_path).resolve()
        job_dir = request.parent
        process = subprocess.Popen(
            (
                sys.executable,
                "-m",
                "lambdaforge.controlplane.ProcessSupervisor",
                "serve-input" if stdin_data is not None else "serve",
                str(request),
            ),
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        if stdin_data is not None:
            assert process.stdin is not None
            with process.stdin as channel:
                channel.write(stdin_data)
        state_path = job_dir / "state.json"
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            state = cls._read_json(state_path)
            if state and state.get("supervisor_identity"):
                return {"supervisor_pid": process.pid, "state": state.get("state")}
            if process.poll() is not None:
                break
            time.sleep(0.05)
        raise RuntimeError(f"Process supervisor did not acknowledge launch for {job_dir.name}.")

    @classmethod
    def serve(cls, request_path: str | Path, *, stdin_data: bytes | None = None) -> int:
        """Queue resources, run the scientific child and publish authoritative state."""
        request_file = Path(request_path).resolve()
        request = cls._required_mapping(cls._read_json(request_file), "request")
        job_dir = request_file.parent
        job_id = str(request["job_id"])
        command = tuple(str(item) for item in request["command"])
        work_dir = job_dir / "work"
        work_dir.mkdir(parents=True, exist_ok=True)
        supervisor_command = tuple(__import__("psutil").Process(os.getpid()).cmdline())
        supervisor = ProcessIdentity.create(os.getpid(), os.getpgrp(), supervisor_command, job_id)
        now = cls._now()
        cls._update_state(
            job_dir,
            {
                "job_state_version": 1,
                "job_id": job_id,
                "cluster": str(request.get("cluster", "local")),
                "state": JobState.STAGING.value,
                "created_at_utc": str(request.get("created_at_utc", now)),
                "updated_at_utc": now,
                "heartbeat_at_utc": now,
                "supervisor_identity": supervisor.to_dict(),
                "process_identity": None,
                "allocated_gpus": [],
                "allocated_cpus": [],
                "resource_safety": request.get("resource_safety", {}),
                "requested_resources": request.get("resources", {}),
                "exit_code": None,
                "message": "Supervisor started.",
            },
        )
        # Log files belong to the Job lifecycle, not only to the scientific child.  Create
        # them before staging/resource admission so a pre-launch failure still has a stable,
        # readable (possibly empty) scientific stream.
        stdout_path = job_dir / "stdout.log"
        stderr_path = job_dir / "stderr.log"
        stdout_path.touch(exist_ok=True)
        stderr_path.touch(exist_ok=True)
        leases: tuple[int, ...] = ()
        cpu_leases: tuple[int, ...] = ()
        process: subprocess.Popen[bytes] | None = None
        identity: ProcessIdentity | None = None
        gpu_access: Mapping[str, Any] = {}
        gpu_mode = "exclusive"
        queue_handled = False
        try:
            from lambdaforge.runtime.ResourceSafetyPolicy import ResourceSafetyPolicy

            resource_safety = ResourceSafetyPolicy.from_value(request.get("resource_safety"))
            # Configure this detached supervisor before outer admission, without
            # changing the user's parent environment or provider grant.
            os.environ["LAMBDAFORGE_RESOURCE_SAFETY"] = json.dumps(resource_safety.to_dict())
            queue_ready = cls._wait_for_execution_queue(job_dir, request, supervisor)
            queue_handled = bool(request.get("execution_queue"))
            if not queue_ready:
                return 0
            if not cls._wait_for_storage(job_dir, request, supervisor):
                return 0
            if bool(request.get("stage_source", False)):
                source = Path(str(request["source_work_dir"])).resolve()
                if not source.is_dir() or source.is_symlink():
                    raise RuntimeError(f"Unsafe or missing staged source: {source}")
                # ControlPlane normally pre-stages an SSH workspace.  Older requests may still
                # ask the supervisor to stage that exact directory; copying a directory onto
                # itself makes shutil recurse into SameFileError noise instead of launching.
                if source != work_dir.resolve():
                    shutil.copytree(source, work_dir, dirs_exist_ok=True)
            resources = cls._required_mapping(request.get("resources", {}), "resources")
            cpu_leases = cls._wait_for_capacity(job_dir, request, supervisor)
            gpu_count = int(resources.get("gpu_count", 0))
            gpu_access = cls._required_mapping(request.get("gpu_access", {}), "gpu_access")
            gpu_mode = str(gpu_access.get("mode", "exclusive"))
            if gpu_count and gpu_mode == "command":
                from lambdaforge.controlplane.GpuAccessPolicy import GpuAccessPolicy

                GpuAccessPolicy.from_mapping(gpu_access).require_owned_reservation_contract()
            if gpu_mode == "auto":
                gpu_mode = "exclusive"
            leases = (
                ()
                if gpu_mode == "command"
                else cls._wait_for_gpus(
                    job_dir,
                    request,
                    supervisor,
                    gpu_count,
                    allow_external_use=gpu_mode == "shared",
                )
            )
            current = cls._read_json(job_dir / "state.json") or {}
            if current.get("state") == JobState.CANCELLED.value:
                return 0
            environment = os.environ.copy()
            environment["LAMBDAFORGE_RESOURCE_SAFETY"] = json.dumps(resource_safety.to_dict())
            environment["LAMBDAFORGE_CLUSTER"] = str(request.get("cluster", "local"))
            environment["LAMBDAFORGE_JOB_ID"] = job_id
            environment["LAMBDAFORGE_EXECUTION_MODE"] = "worker"
            if isinstance(request.get("storage"), Mapping):
                environment["LAMBDAFORGE_STORAGE_POLICY"] = json.dumps(request["storage"])
                environment["LAMBDAFORGE_STORAGE_RESERVED"] = "1"
            environment["LAMBDAFORGE_PROGRESS_PATH"] = str(job_dir / "progress.json")
            environment["LAMBDAFORGE_JOB_RESULT_PATH"] = str(job_dir / "result.json")
            environment["LAMBDAFORGE_STUDY_PATH"] = str(job_dir / "study")
            environment["LAMBDAFORGE_GPU_ACCESS_MODE"] = gpu_mode
            environment["LAMBDAFORGE_REQUESTED_GPUS"] = str(gpu_count)
            if gpu_mode in {"shared", "command"}:
                from lambdaforge.runtime.HostCoordination import coordination_root

                environment["LAMBDAFORGE_GPU_ADMISSION_ROOT"] = str(
                    coordination_root("gpu-admission")
                )
            raw_visibility = gpu_access.get("visibility_command", ())
            visibility = (
                tuple(str(value) for value in raw_visibility)
                if isinstance(raw_visibility, Sequence)
                and not isinstance(raw_visibility, str | bytes)
                else ()
            )
            prefix = gpu_access.get("command_prefix", ())
            if not visibility and gpu_mode == "command" and list(prefix) == ["gpu", "exec"]:
                visibility = ("gpu", "env", "--shell")
            if visibility:
                environment["LAMBDAFORGE_GPU_VISIBILITY_COMMAND"] = json.dumps(visibility)
            cache_root = request.get("cache_root")
            if cache_root:
                environment["LAMBDAFORGE_CACHE_ROOT"] = str(cache_root)
            if request.get("product_root"):
                environment["LAMBDAFORGE_PRODUCT_ROOT"] = str(request["product_root"])
            dataset_registry = request.get("dataset_registry")
            if dataset_registry is not None:
                environment["LAMBDAFORGE_DATASET_REGISTRY"] = str(dataset_registry)
            if leases:
                environment["CUDA_VISIBLE_DEVICES"] = cls._leased_gpu_visibility(
                    Path(str(request["lease_root"])), job_id, leases
                )
            environment = cls._limit_thread_environment(environment, max(1, len(cpu_leases)))
            with (
                stdout_path.open("ab", buffering=0) as stdout,
                stderr_path.open("ab", buffering=0) as stderr,
            ):
                process = subprocess.Popen(
                    command,
                    cwd=work_dir,
                    env=environment,
                    stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                    close_fds=True,
                )
                if stdin_data is not None:
                    assert process.stdin is not None
                    with process.stdin as channel:
                        channel.write(stdin_data)
                    stdin_data = None
                identity = ProcessIdentity.create(
                    process.pid, os.getpgid(process.pid), command, job_id
                )
                try:
                    __import__("psutil").Process(process.pid).cpu_affinity(list(cpu_leases))
                except Exception:
                    pass
                started = cls._now()
                cls._update_state(
                    job_dir,
                    {
                        "state": JobState.RUNNING.value,
                        "started_at_utc": started,
                        "updated_at_utc": started,
                        "heartbeat_at_utc": started,
                        "process_identity": identity.to_dict(),
                        "allocated_gpus": list(leases),
                        "allocated_cpus": list(cpu_leases),
                        "message": "Scientific process is running.",
                    },
                )
                cls._monitor(job_dir, request, process, identity, started)
            return 0
        except Exception as error:
            state = cls._read_json(job_dir / "state.json") or {}
            if state.get("state") not in {JobState.CANCELLED.value, JobState.TIMEOUT.value}:
                cls._update_state(
                    job_dir,
                    {
                        "state": JobState.FAILED.value,
                        "updated_at_utc": cls._now(),
                        "message": f"{error.__class__.__name__}: {error}",
                    },
                )
            return 1
        finally:
            # A successful/failed main process may still leave multiprocessing, trainer or
            # dataloader descendants behind.  Always enforce the Job ownership boundary, not
            # only while the original parent PID is alive.
            cleanup_complete = True
            if identity is not None:
                try:
                    cls._terminate(identity)
                except Exception as error:
                    cleanup_complete = False
                    cls._update_state(
                        job_dir,
                        {"process_cleanup_warning": (f"{type(error).__name__}: {error}")},
                    )
            cls._release_gpus(Path(str(request["lease_root"])), job_id, leases)
            if gpu_mode == "command" and gpu_access.get("claim_command"):
                cls._release_external_gpu(job_dir, work_dir, gpu_access)
            resource_root = request.get("resource_lease_root")
            if resource_root is not None:
                cls._release_capacity(Path(str(resource_root)), job_id)
            if cleanup_complete:
                cls._update_state(
                    job_dir,
                    {"scientific_cleanup_complete": True, "resources_released_at_utc": cls._now()},
                )
            if cleanup_complete and (queue_handled or not request.get("execution_queue")):
                from lambdaforge.controlplane.ExecutionQueue import ExecutionQueue

                state = cls._read_json(job_dir / "state.json") or {}
                if state.get("state") in ExecutionQueue.TERMINAL:
                    queue_storage = cls._required_mapping(request.get("storage"), "storage")
                    try:
                        ExecutionQueue(
                            queue_storage["state_root"],
                            queue_storage["run_root"],
                            str(request.get("cluster", "local")),
                        ).release(job_id)
                    except (OSError, ValueError, RuntimeError) as error:
                        # An admission metadata failure blocks followers, never replaces
                        # completed scientific evidence or skips the remaining retention.
                        cls._update_state(
                            job_dir, {"queue_release_warning": f"{type(error).__name__}: {error}"}
                        )
            storage = request.get("storage")
            if isinstance(storage, Mapping):
                try:
                    from lambdaforge.controlplane.StorageAdmission import StorageAdmission
                    from lambdaforge.controlplane.StorageOperations import StorageOperations

                    StorageAdmission.release(storage, supervisor)
                    StorageOperations.compact_job(storage, job_id, apply=True)
                    StorageOperations.gc({**storage, "automatic_gc": True}, {}, apply=True)
                except Exception as error:
                    # Retention must never replace the authoritative scientific exit state.
                    cls._update_state(
                        job_dir,
                        {"retention_warning": f"{type(error).__name__}: {error}"},
                    )

    @classmethod
    def _wait_for_execution_queue(
        cls, job_dir: Path, request: Mapping[str, Any], supervisor: ProcessIdentity
    ) -> bool:
        """FIFO gate before any CPU/RAM/GPU/storage admission; independent of the controller."""
        from lambdaforge.controlplane.ExecutionQueue import ExecutionQueue

        admission = request.get("execution_queue")
        if admission is None:
            return True
        if (
            not isinstance(admission, Mapping)
            or admission.get("version") != 1
            or admission.get("job_id") != job_dir.name
        ):
            raise ValueError("Execution queue admission does not match this Job.")
        storage = cls._required_mapping(request.get("storage"), "storage")
        queue = ExecutionQueue(
            storage["state_root"], storage["run_root"], str(request.get("cluster", "local"))
        )
        try:
            queue.mark(job_dir.name, "dispatched")
        except InterruptedError:
            cls._update_state(
                job_dir,
                {
                    "state": JobState.CANCELLED.value,
                    "updated_at_utc": cls._now(),
                    "message": "Queue ticket cancelled.",
                },
            )
            return False

        markers: tuple[dict[str, list[int]], list[dict[str, Any]]] | None = None

        def observe(previous: str) -> Mapping[str, Any]:
            nonlocal markers
            directory = queue.jobs / previous
            if directory.is_symlink():
                raise ValueError("Unsafe execution queue predecessor path.")
            stored = cls._read_json(directory / "state.json")
            if stored is None:
                return {"state": "unknown", "released": False}
            if stored.get("state") not in ExecutionQueue.TERMINAL:
                # Waiting does not require rescanning all host processes on every poll.
                # Nonterminal/stale evidence blocks; it is never promoted to success.
                return {"state": stored.get("state", "unknown"), "released": False}
            if markers is None:
                markers = cls._process_markers()
            state = cls.observe(directory, markers=markers)
            released = (
                str(state.get("state")) in ExecutionQueue.TERMINAL
                and state.get("provider_observation") == "terminal"
            )
            for key in ("supervisor_identity", "process_identity"):
                owner = state.get(key)
                if isinstance(owner, Mapping):
                    released = released and ProcessIdentity.from_mapping(
                        owner
                    ).storage_status() in {"dead", "reused"}
            return {"state": state.get("state"), "released": released}

        previous_wait = None
        while True:
            markers = None
            state = cls._read_json(job_dir / "state.json") or {}
            if state.get("state") == JobState.CANCELLED.value:
                return False
            try:
                waiting = queue.waiting_on(job_dir.name, observe)
            except InterruptedError:
                cls._update_state(
                    job_dir,
                    {
                        "state": JobState.CANCELLED.value,
                        "updated_at_utc": cls._now(),
                        "message": "Queue ticket cancelled.",
                    },
                )
                return False
            if not waiting:
                cls._update_state(
                    job_dir,
                    {
                        "queue_waiting_on": [],
                        "queue_admitted_at_utc": cls._now(),
                        "message": "Execution queue admitted Job.",
                    },
                )
                return True
            # A bounded heartbeat, not a notification/event flood.
            now = time.monotonic()
            if previous_wait is None or now - previous_wait >= cls.HEARTBEAT_SECONDS:
                cls._update_state(
                    job_dir,
                    {
                        "state": JobState.QUEUED.value,
                        "heartbeat_at_utc": cls._now(),
                        "updated_at_utc": cls._now(),
                        "queue_waiting_on": waiting,
                        "supervisor_identity": supervisor.to_dict(),
                        "message": f"Execution queue: waiting for {len(waiting)} preceding Job(s).",
                    },
                )
                previous_wait = now
            time.sleep(cls.POLL_SECONDS)

    @classmethod
    def _wait_for_storage(
        cls, job_dir: Path, request: Mapping[str, Any], supervisor: ProcessIdentity
    ) -> bool:
        """Queue disk commitments without consuming CPU/GPU leases or killing siblings."""
        from lambdaforge.controlplane.StorageAdmission import StorageAdmission
        from lambdaforge.controlplane.StorageOperations import StorageOperations

        descriptor = request.get("storage")
        if not isinstance(descriptor, Mapping):
            return True  # Legacy direct requests have no authoritative storage descriptor.
        resources = cls._required_mapping(request.get("resources", {}), "resources")
        requested = int(resources.get("storage_bytes", 0))
        last_gc = float("-inf")
        while True:
            if (cls._read_json(job_dir / "state.json") or {}).get("state") == "cancelled":
                return False
            observation = StorageAdmission.reserve(descriptor, supervisor, requested)
            if observation.get("admitted"):
                cls._update_state(job_dir, {"storage_admission": observation})
                return True
            if time.monotonic() - last_gc >= 60:
                last_gc = time.monotonic()
                try:
                    deficit = max(
                        0,
                        requested
                        + int(observation.get("reserved_bytes", 0))
                        + int(observation["safety_bytes"])
                        - int(observation["free_bytes"]),
                    )
                    cache_observation = StorageAdmission.filesystem(
                        Path(str(descriptor["cache_root"])), descriptor
                    )
                    StorageOperations.gc(
                        {
                            **descriptor,
                            "automatic_gc": True,
                            "target_reclaim_bytes": deficit
                            if cache_observation["device"] == observation["device"]
                            else 0,
                        },
                        {},
                        apply=True,
                    )
                except (OSError, TimeoutError):
                    pass  # Live leases/another collector must not fail the queued Job.
                observation = StorageAdmission.reserve(descriptor, supervisor, requested)
                if observation.get("admitted"):
                    cls._update_state(job_dir, {"storage_admission": observation})
                    return True
            cls._update_state(
                job_dir,
                {
                    "state": JobState.QUEUED.value,
                    "updated_at_utc": cls._now(),
                    "heartbeat_at_utc": cls._now(),
                    "storage_admission": observation,
                    "message": (
                        f"Waiting: storage admission; requested {requested} bytes, "
                        f"physical free {observation['free_bytes']}, "
                        f"reserved {observation.get('reserved_bytes', 'unknown')}, "
                        f"safety {observation['safety_bytes']} bytes."
                    ),
                },
            )
            time.sleep(2.0)

    @classmethod
    def _release_external_gpu(
        cls, job_dir: Path, work_dir: Path, policy: Mapping[str, Any]
    ) -> None:
        """Legacy request cleanup: without a scoped receipt, never release site grants."""
        del work_dir, policy
        cls._update_state(
            job_dir,
            {
                "gpu_release_warning": "External release skipped: no verified Job-owned "
                "reservation receipt. "
                "Manual and sibling reservations remain provider/user-owned."
            },
        )

    @classmethod
    def control(cls, operation: str, job_dir: str | Path) -> dict[str, Any]:
        """Safely cancel, pause or resume only the identity recorded for this job."""
        directory = Path(job_dir).resolve()
        state = cls._required_mapping(cls._read_json(directory / "state.json"), "state")
        job_state = JobState(str(state["state"]))
        if operation == "cancel":
            # Re-run cancellation for an already-cancelled direct Job.  Older versions could
            # persist CANCELLED after signalling only the primary process, so idempotent control
            # must converge any surviving owned workers after an upgrade.
            if job_state.terminal and job_state is not JobState.CANCELLED:
                return dict(state)
            cls._update_state(
                directory,
                {
                    "state": JobState.CANCELLED.value,
                    "updated_at_utc": cls._now(),
                    "message": "Cancellation requested.",
                },
            )
            identity = state.get("process_identity")
            if isinstance(identity, Mapping):
                outcome = cls._terminate(ProcessIdentity.from_mapping(identity))
                cls._update_state(
                    directory,
                    {
                        "updated_at_utc": cls._now(),
                        "message": "Cancellation completed; owned processes were stopped.",
                        "cancellation": outcome,
                    },
                )
            return cls._read_json(directory / "state.json") or {}
        if operation not in {"pause", "resume"}:
            raise ValueError(f"Unknown process control operation: {operation}.")
        expected = JobState.RUNNING if operation == "pause" else JobState.PAUSED
        if job_state is not expected:
            raise RuntimeError(f"Cannot {operation} a {job_state.value} job.")
        raw_identity = state.get("process_identity")
        if not isinstance(raw_identity, Mapping):
            raise RuntimeError("The job has no verified process identity.")
        identity = ProcessIdentity.from_mapping(raw_identity)
        if not identity.matches():
            raise RuntimeError(
                "Process identity no longer matches; refusing to signal a reused PID."
            )
        os.killpg(
            identity.process_group, signal.SIGSTOP if operation == "pause" else signal.SIGCONT
        )
        cls._update_state(
            directory,
            {
                "state": (JobState.PAUSED if operation == "pause" else JobState.RUNNING).value,
                "updated_at_utc": cls._now(),
                "message": (
                    "Paused; CPU execution stopped but RAM/VRAM and leases remain allocated."
                    if operation == "pause"
                    else "Resumed."
                ),
            },
        )
        return cls._read_json(directory / "state.json") or {}

    @classmethod
    def observe(
        cls,
        job_dir: str | Path,
        *,
        markers: tuple[dict[str, list[int]], list[dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        """Verify owned process lifetimes on the execution host, without writes/signals.

        A reachable state file is not proof of a running process. Inaccessible same-user
        processes prevent a loss conclusion; different-user processes are never inspected.
        """
        directory = Path(job_dir).absolute()
        if directory.resolve() != directory or not directory.name.startswith("job-"):
            raise ValueError("Unsafe process observation directory.")
        records: list[Mapping[str, Any]] = []
        legacy_terminal = not (directory / "request.json").exists()
        for name in ("request.json", "state.json"):
            path = directory / name
            if path.is_symlink():
                raise ValueError(f"Unsafe process observation record: {name}.")
            if name == "request.json" and legacy_terminal:
                # Old native terminal records can be shown, not adopted as an owned execution.
                records.append({"job_id": directory.name})
                continue
            value = cls._required_mapping(cls._read_json(path), name)
            if value.get("job_id") != directory.name:
                raise ValueError(f"Process observation ownership mismatch: {name}.")
            records.append(value)
        state = dict(records[1])
        recorded = JobState(str(state.get("state", "unknown")))
        state.update(source="process-host", raw_state=recorded.value)
        if legacy_terminal:
            if not recorded.terminal:
                raise ValueError("Nonterminal process ownership request is missing.")
            return {
                **state,
                "provider_observation": "unknown",
                "observation_reason": "LEGACY_TERMINAL_OWNERSHIP_UNVERIFIED",
            }
        boot_time = cls._host_boot_time()
        if boot_time is not None:
            state["observed_boot_time"] = boot_time
        lifetimes = {}
        owner_births = []
        for role in ("supervisor_identity", "process_identity"):
            raw = state.get(role)
            if raw is None:
                lifetimes[role] = "missing"
                continue
            identity = ProcessIdentity.from_mapping(cls._required_mapping(raw, role))
            if identity.job_id != directory.name:
                raise ValueError(f"Process identity ownership mismatch: {role}.")
            if not 0 < identity.create_time < float("inf"):
                raise ValueError(f"Invalid process creation time: {role}.")
            owner_births.append(identity.create_time)
            # A lifetime from a previous host boot cannot survive as this PID. Keep
            # markers authoritative: a newly launched process bearing the same Job
            # marker still blocks recovery below, even after a verified reboot.
            lifetimes[role] = (
                "dead"
                if boot_time is not None and identity.create_time < boot_time - 1.0
                else identity.storage_status()
            )
        state["process_lifetimes"] = lifetimes
        supervisor, child = lifetimes["supervisor_identity"], lifetimes["process_identity"]
        if recorded.terminal:
            owned, unresolved = markers if markers is not None else cls._process_markers()
            complete = cls._marker_scope(state, unresolved, owner_births, boot_time=boot_time)
            complete = complete and "unresolved" not in lifetimes.values()
            live = "alive" in lifetimes.values() or bool(owned.get(directory.name))
            state["owned_process_count"] = len(owned.get(directory.name, ()))
            return {
                **state,
                "provider_observation": "degraded"
                if live
                else "terminal"
                if complete
                else "unknown",
                "observation_reason": "TERMINAL_WITH_LIVE_PROCESSES"
                if live
                else "TERMINAL_RECORDED"
                if complete
                else "PROCESS_OWNERSHIP_UNRESOLVED",
            }
        if supervisor == "alive" and recorded is JobState.RUNNING and child != "alive":
            return {
                **state,
                "provider_observation": "degraded",
                "observation_reason": "SCIENTIFIC_CHILD_UNVERIFIED",
            }
        if supervisor == "alive":
            return {**state, "provider_observation": "alive"}
        if child == "alive":
            return {
                **state,
                "provider_observation": "degraded",
                "observation_reason": "SUPERVISOR_LOST_CHILD_ALIVE",
            }
        owned, unresolved = markers if markers is not None else cls._process_markers()
        complete = cls._marker_scope(state, unresolved, owner_births, boot_time=boot_time)
        state["owned_process_count"] = len(owned.get(directory.name, ()))
        # A legacy identity-less state cannot certify death, even on an otherwise empty host.
        proven = supervisor in {"dead", "reused"} and child in {"dead", "reused", "missing"}
        if proven and complete and not state["owned_process_count"]:
            return {
                **state,
                "state": JobState.UNKNOWN.value,
                "provider_observation": "lost_confirmed",
                "observation_reason": "OWNED_PROCESSES_ABSENT",
            }
        return {
            **state,
            "provider_observation": "unknown",
            "observation_reason": "PROCESS_OWNERSHIP_UNRESOLVED",
        }

    @classmethod
    def detached_owner_absent(
        cls,
        owner: ProcessIdentity,
        *,
        markers: tuple[dict[str, list[int]], list[dict[str, Any]]] | None = None,
    ) -> bool:
        """Prove an operational owner absent after its Job directory was removed.

        This releases only detached operational commitments, not scientific recovery or
        deletion authority. A dead/reused controller alone is insufficient: descendants
        bearing its Job marker or relevant inaccessible processes still protect it.
        """
        if (
            not owner.job_id.startswith("job-")
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in owner.job_id)
            or not 0 < owner.create_time < float("inf")
            or owner.storage_status() not in {"dead", "reused"}
        ):
            return False
        owned, unresolved = markers if markers is not None else cls._process_markers()
        complete = cls._marker_scope(
            {}, unresolved, [owner.create_time], boot_time=cls._host_boot_time()
        )
        return complete and not owned.get(owner.job_id)

    @staticmethod
    def _host_boot_time() -> float | None:
        """Read OS boot evidence; unavailable/invalid evidence never relaxes ownership."""
        import psutil

        try:
            value = float(psutil.boot_time())
            return value if 0 < value < float("inf") else None
        except (OSError, RuntimeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _marker_scope(
        state: dict[str, Any],
        unresolved: list[dict[str, Any]],
        owner_births: list[float],
        *,
        boot_time: float | None = None,
    ) -> bool:
        """An inaccessible unrelated login must not poison every Job on the host.

        Native Jobs create fresh processes/sessions, never attach a preexisting process.
        Descendants can reparent or start new sessions, but cannot join another existing
        session. Only positive lifetime/session evidence excludes an unreadable process;
        unknown birth/ownership and ordinary new same-user workers remain unresolved.
        """
        owner_birth = min(owner_births) if owner_births else None
        previous_boot = (
            boot_time is not None and bool(owner_births) and max(owner_births) < boot_time - 1.0
        )
        relevant = []
        for value in unresolved:
            birth = value.get("create_time")
            origin = value.get("foreign_session_origin")
            unrelated = owner_birth is not None and (
                (isinstance(birth, (int, float)) and 0 < birth < owner_birth)
                or (isinstance(origin, (int, float)) and 0 < origin < owner_birth)
                # Post-reboot access sessions cannot descend from pre-reboot Jobs.
                # Require positive process birth evidence; unknown metadata stays
                # fail-closed. No exclusion is based on a name or htop absence.
                or (
                    previous_boot
                    and isinstance(birth, (int, float))
                    and boot_time is not None
                    and boot_time - 1.0 <= birth < float("inf")
                )
            )
            if not unrelated:
                relevant.append(value)
        state["unresolved_process_count"] = len(relevant)
        state["excluded_unrelated_process_count"] = len(unresolved) - len(relevant)
        state["unresolved_processes"] = relevant[:20]
        return not relevant

    @staticmethod
    def _unreadable_process(process: Any) -> dict[str, Any]:
        """Collect metadata only; never guess non-membership from a process name.

        A foreign-owned session leader spawned by a foreign-owned, preexisting service
        proves an independent access session. Ignore PID 1: reparented scientific workers
        are not independent merely because their parent died. Linux session membership
        cannot be changed to another existing session by an orphan or an exec.
        """
        import psutil

        value: dict[str, Any] = {"pid": process.pid, "create_time": None}
        try:
            if not process.is_running():
                return value  # process_iter can cache a lifetime whose PID has been reused.
            value["create_time"] = process.create_time()
            if os.name != "posix":
                return value
            sid = os.getsid(process.pid)
            leader = psutil.Process(sid)
            parent = leader.parent()
            if parent is None or parent.pid == 1:
                return value
            own_uid = os.getuid()
            leader_uids, parent_uids = leader.uids(), parent.uids()
            if leader_uids != parent_uids or any(uid == own_uid for uid in leader_uids):
                return value
            leader_birth, parent_birth = leader.create_time(), parent.create_time()
            if (
                os.getsid(leader.pid) == leader.pid
                and 0 < parent_birth <= leader_birth <= value["create_time"]
                and os.getsid(process.pid) == sid
                and process.is_running()
                and leader.is_running()
                and parent.is_running()
            ):
                value["foreign_session_origin"] = parent_birth
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            pass  # Unresolved evidence stays protected, including races during metadata reads.
        return value

    @classmethod
    def _process_markers(cls) -> tuple[dict[str, list[int]], list[dict[str, Any]]]:
        """One host scan, with per-Job uncertainty instead of a host-wide denial flag."""
        import psutil

        markers: dict[str, list[int]] = {}
        unresolved = []
        user = psutil.Process().username()
        for process in psutil.process_iter(("pid", "status", "username")):
            if process.pid == os.getpid():
                continue
            try:
                if process.username() != user or process.status() in {
                    psutil.STATUS_ZOMBIE,
                    psutil.STATUS_DEAD,
                }:
                    continue
                job_id = process.environ().get("LAMBDAFORGE_JOB_ID")
                if job_id:
                    markers.setdefault(job_id, []).append(process.pid)
            except psutil.NoSuchProcess:
                continue
            except (psutil.AccessDenied, OSError):
                unresolved.append(cls._unreadable_process(process))
        return markers, unresolved

    @classmethod
    def inventory(cls, root: str | Path) -> tuple[dict[str, Any], ...]:
        """Read owned Jobs, retaining per-Job errors without hiding healthy siblings.

        An error row identifies only the directory inspected, not verified ownership or
        scientific state. Consumers must not adopt it as a newly discovered Job.
        Root/transport/process-scan failures still invalidate the whole observation.
        """
        directory = Path(root).resolve()
        if not directory.is_dir():
            return ()
        states = []
        markers = cls._process_markers()
        for child in sorted(directory.iterdir()):
            if child.is_symlink() or not child.is_dir() or not child.name.startswith("job-"):
                continue
            if any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in child.name
            ):
                continue  # Not a native JobStore identity; never adopt arbitrary directories.
            if not (child / "state.json").exists() and not (child / "request.json").exists():
                continue
            try:
                state = cls.observe(child, markers=markers)
                request = cls._read_json(child / "request.json")
                if not request:
                    raise ValueError(f"Missing/corrupt process request for {child.name}.")
                study = cls._read_json(child / "study" / "overview.json")
                if study is None:
                    study = cls._read_json(child / "study" / "summary.json")
                if study is not None:
                    state = {**state, "study": study_overview(study)}
                states.append({"state": state, "request": request})
            except (ValueError, TypeError, KeyError, OSError, OverflowError) as error:
                states.append(
                    {
                        "request": {"job_id": child.name},
                        "state": {"job_id": child.name, "state": JobState.UNKNOWN.value},
                        "observation_error": {
                            "type": type(error).__name__,
                            "message": str(error)[:1500],
                        },
                    }
                )
        return tuple(states)

    @classmethod
    def _monitor(
        cls,
        job_dir: Path,
        request: Mapping[str, Any],
        process: subprocess.Popen[bytes],
        identity: ProcessIdentity,
        started: str,
    ) -> None:
        start_monotonic = time.monotonic()
        runtime = cls._required_mapping(request.get("resources", {}), "resources").get(
            "runtime_seconds"
        )
        runtime_seconds = float(runtime) if runtime is not None else None
        last_heartbeat = 0.0
        while True:
            result = process.poll()
            if result is not None:
                state = cls._read_json(job_dir / "state.json") or {}
                current = JobState(str(state.get("state", JobState.UNKNOWN.value)))
                terminal = (
                    current
                    if current in {JobState.CANCELLED, JobState.TIMEOUT}
                    else (JobState.SUCCEEDED if result == 0 else JobState.FAILED)
                )
                cls._update_state(
                    job_dir,
                    {
                        "state": terminal.value,
                        "exit_code": result,
                        "finished_at_utc": cls._now(),
                        "updated_at_utc": cls._now(),
                        "message": f"Scientific process exited with code {result}.",
                    },
                )
                return
            state = cls._read_json(job_dir / "state.json") or {}
            if state.get("state") == JobState.CANCELLED.value:
                cls._terminate(identity)
                continue
            if runtime_seconds is not None and time.monotonic() - start_monotonic > runtime_seconds:
                cls._update_state(
                    job_dir,
                    {
                        "state": JobState.TIMEOUT.value,
                        "updated_at_utc": cls._now(),
                        "message": f"Maximum runtime {runtime_seconds:g}s exceeded.",
                    },
                )
                cls._terminate(identity)
                continue
            if time.monotonic() - last_heartbeat >= cls.HEARTBEAT_SECONDS:
                last_heartbeat = time.monotonic()
                cls._heartbeat(job_dir, identity, started)
            time.sleep(cls.POLL_SECONDS)

    @classmethod
    def _heartbeat(cls, job_dir: Path, identity: ProcessIdentity, started: str) -> None:
        now = cls._now()
        (job_dir / "heartbeat").touch()
        usage: dict[str, Any] = {"timestamp_utc": now, "pid": identity.pid}
        state = cls._read_json(job_dir / "state.json") or {}
        try:
            import psutil

            process = psutil.Process(identity.pid)
            if process.create_time() != identity.create_time or (
                os.name == "posix" and process.uids().real != os.getuid()
            ):
                raise RuntimeError("Resource observation owner lifetime is unavailable.")
            processes = (process, *process.children(recursive=True))
            cpu_time = 0.0
            rss_bytes = 0
            threads = 0
            descendants: set[int] = set()
            lifetimes: dict[int, float] = {}
            for child in processes:
                try:
                    with child.oneshot():
                        if os.name == "posix" and child.uids().real != os.getuid():
                            continue
                        times = child.cpu_times()
                        cpu_time += float(times.user + times.system)
                        rss_bytes += int(child.memory_info().rss)
                        threads += int(child.num_threads())
                        descendants.add(child.pid)
                        lifetimes[child.pid] = child.create_time()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            previous = state.get("observed_usage", {})
            previous = previous if isinstance(previous, Mapping) else {}
            previous_time = previous.get("cpu_time_seconds")
            previous_at = previous.get("timestamp_utc")
            cpu_percent: float | None = None
            if isinstance(previous_time, (int, float)) and isinstance(previous_at, str):
                try:
                    elapsed = (
                        datetime.fromisoformat(now) - datetime.fromisoformat(previous_at)
                    ).total_seconds()
                except ValueError:
                    elapsed = 0.0
                if elapsed > 0:
                    cpu_percent = max(0.0, 100.0 * (cpu_time - float(previous_time)) / elapsed)
            usage.update(
                {
                    "cpu_percent": cpu_percent,
                    "cpu_time_seconds": cpu_time,
                    "rss_bytes": rss_bytes,
                    "threads": threads,
                }
            )
            from lambdaforge.runtime.HostObservation import observe_host, process_tree_memory

            usage["host_capacity"] = observe_host()
            usage["memory_attribution"] = process_tree_memory(identity.pid, identity.create_time)
            from lambdaforge.runtime.GPUObservation import GPUObservationCache, GPUObservationPolicy
            from lambdaforge.runtime.ResourceSafetyPolicy import ResourceSafetyPolicy

            requested = state.get("requested_resources")
            gpu: dict[str, Any] = (
                {"quality": "not-requested", "devices": []}
                if isinstance(requested, Mapping) and not requested.get("gpu_count")
                else GPUObservationCache(
                    policy=GPUObservationPolicy.from_operational(
                        ResourceSafetyPolicy.from_value(state.get("resource_safety"))
                    )
                ).read()
            )
            devices = gpu.get("devices", ())
            ambiguous_lifetime = any(
                item.get("pid") in descendants and item.get("create_time") != lifetimes[item["pid"]]
                for row in devices
                for item in row.get("processes", ())
            )
            attributable = [
                item
                for row in devices
                for item in row.get("processes", ())
                if item.get("pid") in descendants
                and item.get("create_time") == lifetimes[item["pid"]]
            ]
            physical_usage: dict[tuple[str, int], int] = {}
            for row in devices:
                for item in row.get("processes", ()):
                    if (
                        item.get("pid") not in descendants
                        or item.get("create_time") != lifetimes[item["pid"]]
                        or item.get("used_bytes") is None
                    ):
                        continue
                    key = (str(row.get("parent_uuid") or row.get("uuid")), item["pid"])
                    physical_usage[key] = max(physical_usage.get(key, 0), item["used_bytes"])
            usage["gpu_observation"] = {
                "quality": gpu.get("quality"),
                "observed_at": gpu.get("observed_at"),
                "error": gpu.get("error"),
                "statistics": gpu.get("statistics", {}),
                "process_attribution": "unavailable" if ambiguous_lifetime else "qualified",
            }
            if (
                gpu.get("quality") == "measured"
                and not ambiguous_lifetime
                and bool(devices)
                and all(
                    row.get("compute_process_count") is not None
                    and row.get("quality") == "measured"
                    for row in devices
                )
                and all(item.get("used_bytes") is not None for item in attributable)
            ):
                usage["gpu_memory_bytes"] = sum(physical_usage.values())
            usage["process_count"] = len(descendants)
        except Exception:
            usage["process_observation"] = "unavailable"
        from lambdaforge.runtime.ResourceUsageHistory import record_usage

        try:
            history_summary = record_usage(job_dir, usage)
        except (OSError, ValueError, KeyError, TypeError) as error:
            history_summary = {"quality": "unavailable", "error": type(error).__name__}
        progress = cls._read_json(job_dir / "progress.json")
        cls._update_state(
            job_dir,
            {
                "heartbeat_at_utc": now,
                "updated_at_utc": now,
                "started_at_utc": state.get("started_at_utc", started),
                "observed_usage": usage,
                "usage_history_summary": history_summary,
                "progress": progress if isinstance(progress, Mapping) else None,
            },
        )

    @classmethod
    def _wait_for_gpus(
        cls,
        job_dir: Path,
        request: Mapping[str, Any],
        supervisor: ProcessIdentity,
        count: int,
        *,
        allow_external_use: bool = False,
    ) -> tuple[int, ...]:
        if count <= 0:
            return ()
        lease_root = Path(str(request["lease_root"])).absolute()
        from lambdaforge.runtime.GPUObservation import GPUObservationUnavailable

        while True:
            state = cls._read_json(job_dir / "state.json") or {}
            if state.get("state") == JobState.CANCELLED.value:
                return ()
            try:
                allocated = cls._acquire_gpus(
                    lease_root, supervisor, count, allow_external_use=allow_external_use
                )
            except (TimeoutError, GPUObservationUnavailable):
                allocated = None  # Ordinary host lease contention keeps the Job queued.
            if allocated is not None:
                return allocated
            cls._update_state(
                job_dir,
                {
                    "state": JobState.QUEUED.value,
                    "updated_at_utc": cls._now(),
                    "heartbeat_at_utc": cls._now(),
                    "message": cls._lease_wait_reason
                    or f"Waiting for {count} LambdaForge GPU lease(s).",
                },
            )
            (job_dir / "heartbeat").touch()
            time.sleep(2.0)

    @classmethod
    def _wait_for_capacity(
        cls,
        job_dir: Path,
        request: Mapping[str, Any],
        supervisor: ProcessIdentity,
    ) -> tuple[int, ...]:
        resources = cls._required_mapping(request.get("resources", {}), "resources")
        cpu_count = int(resources.get("cpu_cores", 1))
        ram_bytes = int(resources.get("ram_bytes", 0))
        root = Path(str(request["resource_lease_root"])).absolute()
        while True:
            try:
                allocated = cls._acquire_capacity(root, supervisor, cpu_count, ram_bytes)
            except TimeoutError:
                allocated = None
            if allocated is not None:
                return allocated
            state = cls._read_json(job_dir / "state.json") or {}
            if state.get("state") == JobState.CANCELLED.value:
                return ()
            cls._update_state(
                job_dir,
                {
                    "state": JobState.QUEUED.value,
                    "updated_at_utc": cls._now(),
                    "heartbeat_at_utc": cls._now(),
                    "message": cls._lease_wait_reason
                    or f"Waiting for {cpu_count} CPU core(s) and {ram_bytes} RAM bytes.",
                },
            )
            (job_dir / "heartbeat").touch()
            time.sleep(2.0)

    @classmethod
    def _acquire_capacity(
        cls,
        root: Path,
        supervisor: ProcessIdentity,
        cpu_count: int,
        ram_bytes: int,
    ) -> tuple[int, ...] | None:
        from lambdaforge.runtime.HostCoordination import boot_identity, host_identity
        from lambdaforge.runtime.HostLeaseStore import lease_directory, lease_records
        from lambdaforge.runtime.HostObservation import (
            observe_host,
            pending_ram_bytes,
            process_tree_memory,
        )

        cls._lease_wait_reason = None
        scoped = lease_directory(root)
        available_cpus = (
            tuple(sorted(os.sched_getaffinity(0)))
            if hasattr(os, "sched_getaffinity")
            else tuple(range(os.cpu_count() or 1))
        )
        if cpu_count > len(available_cpus):
            raise RuntimeError(
                f"Requested {cpu_count} CPU cores but only {len(available_cpus)} are available."
            )
        capacity = observe_host()
        effective_cpu = capacity.get("effective_cpu_cores")
        if effective_cpu is None:
            return None
        if cpu_count > effective_cpu:
            raise RuntimeError("Requested CPU cores exceed effective affinity/cgroup capacity.")
        available_cpus = tuple(capacity.get("cpu_affinity", available_cpus))
        if ram_bytes > capacity["effective_total_bytes"]:
            raise RuntimeError("Requested RAM exceeds physical/cgroup host capacity.")
        with CrossProcessFileLock(
            root / ".leases.lock",
            shared=False,
            timeout_seconds=10.0,
            poll_interval_seconds=0.05,
        ):
            occupied: set[int] = set()
            reserved_ram = 0
            for path, lease in lease_records(root, scoped, "job-"):
                if not cls._host_lease_active(path, lease):
                    continue
                owner = ProcessIdentity.from_mapping(lease["supervisor_identity"])
                occupied.update(int(item) for item in lease.get("cpus", ()))
                observed = (
                    process_tree_memory(owner.pid, owner.create_time)
                    if lease.get("host") == host_identity() and lease.get("boot") == boot_identity()
                    else {"pss_bytes": None}
                )
                reserved_ram += pending_ram_bytes(int(lease.get("ram_bytes", 0)), observed)
            cpus = tuple(item for item in available_cpus if item not in occupied)[:cpu_count]
            capacity = observe_host()
            free_ram = capacity["effective_available_bytes"]
            if (
                free_ram is None
                or free_ram <= 0
                or len(cpus) < cpu_count
                or ram_bytes > max(0, free_ram - reserved_ram)
            ):
                return None
            cls._write_json(
                scoped / f"{supervisor.job_id}.json",
                {
                    "lease_version": 2,
                    "job_id": supervisor.job_id,
                    "cpus": list(cpus),
                    "ram_bytes": ram_bytes,
                    "ram_demand_known": ram_bytes > 0,
                    "host": host_identity(),
                    "boot": boot_identity(),
                    "capacity": capacity,
                    "supervisor_identity": supervisor.to_dict(),
                    "created_at_utc": cls._now(),
                },
            )
            return cpus

    @classmethod
    def _release_capacity(cls, root: Path, job_id: str) -> None:
        from lambdaforge.runtime.HostLeaseStore import lease_directory, read_lease

        if not root.exists():
            return
        with CrossProcessFileLock(
            root / ".leases.lock",
            shared=False,
            timeout_seconds=10.0,
            poll_interval_seconds=0.05,
        ):
            for path in (lease_directory(root) / f"{job_id}.json", root / f"{job_id}.json"):
                if path.exists():
                    value = read_lease(path)
                    if value.get("job_id") == job_id and cls._owned_host_lease(value):
                        path.unlink(missing_ok=True)

    @staticmethod
    def _owned_host_lease(value: Mapping[str, Any]) -> bool:
        from lambdaforge.runtime.HostCoordination import boot_identity, host_identity

        raw = value.get("supervisor_identity")
        return (
            value.get("host") == host_identity()
            and value.get("boot") == boot_identity()
            and isinstance(raw, Mapping)
            and raw.get("job_id") == value.get("job_id")
            and ProcessIdentity.from_mapping(raw).pid == os.getpid()
            and ProcessIdentity.from_mapping(raw).storage_status() == "alive"
        )

    @classmethod
    def _host_lease_active(cls, path: Path, value: Mapping[str, Any]) -> bool:
        from lambdaforge.runtime.HostCoordination import boot_identity, host_identity

        raw = value.get("supervisor_identity")
        if not isinstance(raw, Mapping):
            raise ValueError("Resource lease has missing/corrupt ownership evidence.")
        owner = ProcessIdentity.from_mapping(raw)
        # Legacy unscoped files may belong to another shared-home host. A local PID
        # lookup cannot prove them dead; preserve them until explicit reconciliation.
        if value.get("host") is None or value.get("boot") is None:
            cls._lease_wait_reason = (
                "Waiting for legacy unscoped resource lease ownership to be reconciled; "
                "a local PID lookup cannot prove its shared-home host or lifetime."
            )
            return True
        if value.get("host") != host_identity():
            return False
        if value["boot"] != boot_identity() or owner.storage_status() in {"dead", "reused"}:
            path.unlink(missing_ok=True)
            return False
        return True

    @classmethod
    def _acquire_gpus(
        cls,
        root: Path,
        supervisor: ProcessIdentity,
        count: int,
        *,
        allow_external_use: bool = False,
    ) -> tuple[int, ...] | None:
        from lambdaforge.runtime.HostCoordination import boot_identity, host_identity
        from lambdaforge.runtime.HostLeaseStore import (
            gpu_lease_name,
            lease_directory,
            lease_records,
        )

        cls._lease_wait_reason = None
        scoped = lease_directory(root)
        # Slow observation/identity discovery never runs inside the outer lease lock.
        devices = cls._direct_gpu_devices()
        gpu_indices = tuple(devices)
        if count > len(gpu_indices):
            raise RuntimeError(f"Requested {count} GPUs but only {len(gpu_indices)} are visible.")
        occupied = (
            set()
            if allow_external_use
            else {
                token
                for token, row in devices.items()
                if row.get("quality") != "measured" or row.get("compute_process_count") != 0
            }
        )
        with CrossProcessFileLock(
            root / ".leases.lock",
            shared=False,
            timeout_seconds=10.0,
            poll_interval_seconds=0.05,
        ):
            active = [
                value
                for path, value in lease_records(root, scoped, "gpu-")
                if cls._host_lease_active(path, value)
            ]
            available = []
            for index in gpu_indices:
                device_uuid = str(devices[index].get("uuid", ""))
                gpu_lease_name(device_uuid)  # Fail closed before any lease write.
                if any(
                    # Old ordinals cannot be compared across CUDA ordering; parent
                    # exclusive access also conflicts with any of its MIG instances.
                    not value.get("device_uuid")
                    or (
                        (
                            value["device_uuid"] == device_uuid
                            or value.get("parent_uuid") == device_uuid
                            or devices[index].get("parent_uuid") == value.get("device_uuid")
                        )
                        and (not allow_external_use or value.get("mode") != "shared")
                    )
                    for value in active
                ):
                    continue
                if index not in occupied:
                    available.append(index)
            if len(available) < count:
                return None
            loads = (
                {
                    token: (
                        float(row.get("utilization_percent") or 0),
                        1 - row["free_bytes"] / row["total_bytes"],
                    )
                    for token, row in devices.items()
                    if row.get("quality") == "measured"
                }
                if allow_external_use
                else {}
            )
            selected = tuple(
                sorted(
                    available,
                    key=lambda index: (*loads.get(index, (float("inf"), float("inf"))), index),
                )[:count]
            )
            for index in selected:
                device_uuid = str(devices[index]["uuid"])
                cls._write_json(
                    scoped
                    / (
                        f"{gpu_lease_name(device_uuid)}-shared-{supervisor.job_id}.json"
                        if allow_external_use
                        else f"{gpu_lease_name(device_uuid)}.json"
                    ),
                    {
                        "lease_version": 2,
                        "mode": "shared" if allow_external_use else "exclusive",
                        "device_uuid": device_uuid,
                        "parent_uuid": devices[index].get("parent_uuid"),
                        "gpu": index,
                        "host": host_identity(),
                        "boot": boot_identity(),
                        "job_id": supervisor.job_id,
                        "supervisor_identity": supervisor.to_dict(),
                        "created_at_utc": cls._now(),
                    },
                )
            return selected

    @staticmethod
    def _limit_thread_environment(environment: Mapping[str, str], cpu_cores: int) -> dict[str, str]:
        """Clamp only native child thread limits, preserving lower authored values and grants."""
        result = dict(environment)
        for key, proposed in ProcessGuard().cpu_thread_env_updates(max(1, cpu_cores), 1).items():
            if key.endswith("THREADS") or key in {
                "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
                "OPENCV_FOR_THREADS_NUM",
            }:
                try:
                    authored = int(result.get(key, proposed))
                except ValueError:
                    authored = int(proposed)
                result[key] = str(min(max(1, authored), int(proposed)))
            else:
                result.setdefault(key, proposed)
        return result

    @classmethod
    def _leased_gpu_visibility(cls, root: Path, job_id: str, indices: Sequence[int]) -> str:
        """Narrow child visibility to attested lease UUIDs, independent of CUDA ordering."""
        from lambdaforge.runtime.HostLeaseStore import lease_directory, lease_records

        selected = {
            int(value["gpu"]): str(value["device_uuid"])
            for _, value in lease_records(root, lease_directory(root), "gpu-")
            if value.get("job_id") == job_id and cls._owned_host_lease(value)
        }
        if len(selected) != len(indices) or any(index not in selected for index in indices):
            raise RuntimeError("Granted GPU lease identity is missing; refusing ordinal inference.")
        return ",".join(selected[index] for index in indices)

    @classmethod
    def _release_gpus(cls, root: Path, job_id: str, indices: Sequence[int]) -> None:
        from lambdaforge.runtime.HostLeaseStore import lease_directory, lease_records

        if not root.exists():
            return
        with CrossProcessFileLock(
            root / ".leases.lock",
            shared=False,
            timeout_seconds=10.0,
            poll_interval_seconds=0.05,
        ):
            # Release recorded UUIDs, not a fresh topology mapping of the old ordinals.
            for path, value in lease_records(root, lease_directory(root), "gpu-"):
                if value.get("job_id") == job_id and cls._owned_host_lease(value):
                    path.unlink(missing_ok=True)

    @staticmethod
    def _gpu_loads() -> dict[int, tuple[float, float]]:
        """Rank permitted devices by live utilization and occupied VRAM, not index."""
        try:
            devices = ProcessSupervisor._direct_gpu_devices()
        except (OSError, RuntimeError, ValueError):
            return {}  # Ranking is optional; the Run's physical admission probe is not.
        return {
            token: (
                float(row.get("utilization_percent") or 0),
                1 - row["free_bytes"] / row["total_bytes"],
            )
            for token, row in devices.items()
            if row.get("quality") == "measured"
        }

    @staticmethod
    def _gpu_indices() -> tuple[int, ...]:
        return tuple(ProcessSupervisor._direct_gpu_devices())

    @staticmethod
    def _direct_gpu_devices() -> dict[int, Mapping[str, Any]]:
        """Resolve CUDA ordinals once; ordinary memory refreshes share the native cache."""
        from lambdaforge.runtime.GPUDeviceIdentity import visible_devices
        from lambdaforge.runtime.GPUObservation import (
            GPUObservationCache,
            GPUObservationUnavailable,
        )

        inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
        if inherited is not None:
            tokens = tuple(value.strip() for value in inherited.split(",") if value.strip())
            if not tokens or any(
                value.lower() in {"-1", "none", "void", "nodevfiles"} for value in tokens
            ):
                return {}
        else:
            tokens = ()
        snapshot = GPUObservationCache().read(admission=True)
        if snapshot.get("quality") != "measured":
            raise GPUObservationUnavailable(
                "GPU observation unavailable/backoff; awaiting fresh evidence."
            )
        count = (
            len(tokens)
            if inherited is not None
            else sum(row.get("parent_uuid") is None for row in snapshot["devices"])
        )
        if count == 0:
            return {}
        try:
            rows = visible_devices(snapshot["devices"], count)
        except (OSError, RuntimeError, ValueError) as error:
            raise GPUObservationUnavailable("Granted GPU identity is unavailable.") from error
        keys = (
            tuple(int(value) for value in tokens)
            if tokens and all(value.isdigit() for value in tokens)
            else tuple(range(count))
        )
        return dict(zip(keys, rows, strict=True))

    @staticmethod
    def _externally_occupied_gpus() -> set[int]:
        """Conservatively avoid GPUs with observable compute processes."""
        return {
            token
            for token, row in ProcessSupervisor._direct_gpu_devices().items()
            if row.get("quality") != "measured" or row.get("compute_process_count") != 0
        }

    @classmethod
    def _terminate(cls, identity: ProcessIdentity, grace_seconds: float = 5.0) -> dict[str, Any]:
        """Stop every observable process owned by one Job and verify convergence.

        Process-group signalling handles ordinary descendants atomically.  The inherited,
        collision-resistant Job marker additionally finds workers that reparented themselves or
        opened another session.  PID identity is still checked before the recorded primary group
        is signalled, so an old state file can never target a reused unrelated PID.
        """
        primary_matches = identity.matches()
        owned = cls._job_processes(identity.job_id)
        initial_pids = {process.pid for process in owned}
        if primary_matches:
            initial_pids.add(identity.pid)
            try:
                os.killpg(identity.process_group, signal.SIGTERM)
            except ProcessLookupError:
                pass
        ProcessGuard().terminate_process_trees(
            tuple(sorted(initial_pids)),
            grace_seconds=grace_seconds,
            include_parents=True,
        )

        # Close races where a worker spawned another child while termination began.  Only exact
        # processes still carrying this Job's marker are eligible for the forced pass.
        remaining = cls._job_processes(identity.job_id)
        for process in remaining:
            try:
                process.kill()
            except Exception:
                pass
        if remaining:
            try:
                import psutil

                psutil.wait_procs(remaining, timeout=1.0)
            except Exception:
                pass
        survivors = cls._job_processes(identity.job_id)
        if survivors:
            raise RuntimeError(
                "Cancellation could not stop every process owned by Job "
                f"{identity.job_id}: {[process.pid for process in survivors]}."
            )
        return {
            "requested_at_utc": cls._now(),
            "observed_processes": len(initial_pids),
            "forced_processes": len(remaining),
            "remaining_processes": 0,
        }

    @staticmethod
    def _job_processes(job_id: str) -> list[Any]:
        """Find same-user processes carrying one exact inherited Job marker."""
        try:
            import psutil
        except Exception:
            return []
        current = os.getpid()
        processes: list[Any] = []
        for process in psutil.process_iter(("pid", "status")):
            if process.pid == current:
                continue
            try:
                if process.status() == psutil.STATUS_ZOMBIE:
                    continue
                if process.environ().get("LAMBDAFORGE_JOB_ID") == job_id:
                    processes.append(process)
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                continue
        return processes

    @classmethod
    def _update_state(cls, job_dir: Path, changes: Mapping[str, Any]) -> None:
        with CrossProcessFileLock(
            job_dir / ".state.lock",
            shared=False,
            timeout_seconds=10.0,
            poll_interval_seconds=0.05,
        ):
            current = cls._read_json(job_dir / "state.json") or {}
            current.update(changes)
            cls._write_json(job_dir / "state.json", current)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _write_json(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _required_mapping(value: object, name: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError(f"Process supervisor {name} must be a mapping.")
        return value

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def main(cls, argv: Sequence[str] | None = None) -> int:
        values = tuple(argv if argv is not None else sys.argv[1:])
        if len(values) < 2:
            raise SystemExit("Usage: ProcessSupervisor launch|serve|control|inventory|observe PATH")
        operation, path, *rest = values
        if operation in {"launch-input", "serve-input"}:
            payload = sys.stdin.buffer.read(4097)
            if not payload or len(payload) > 4096:
                raise ValueError("Ephemeral Job input is absent or exceeds 4096 bytes.")
            if operation == "serve-input":
                return cls.serve(path, stdin_data=payload)
            print(json.dumps(cls.launch(path, stdin_data=payload), sort_keys=True))
            return 0
        if operation == "launch":
            print(json.dumps(cls.launch(path), sort_keys=True))
            return 0
        if operation == "serve":
            return cls.serve(path)
        if operation == "control":
            if not rest:
                raise SystemExit("control requires cancel, pause or resume")
            print(json.dumps(cls.control(rest[0], path), sort_keys=True))
            return 0
        if operation == "inventory":
            print(json.dumps(cls.inventory(path), sort_keys=True))
            return 0
        if operation == "observe":
            print(json.dumps(cls.observe(path), sort_keys=True))
            return 0
        raise SystemExit(f"Unknown ProcessSupervisor operation: {operation}")


if __name__ == "__main__":
    raise SystemExit(ProcessSupervisor.main())
