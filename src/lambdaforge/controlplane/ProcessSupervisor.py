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
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.training.orchestration.ProcessGuard import ProcessGuard


class ProcessSupervisor:
    """Own one durable process job without requiring a resident global daemon."""

    POLL_SECONDS = 0.5
    HEARTBEAT_SECONDS = 5.0

    @classmethod
    def launch(cls, request_path: str | Path) -> dict[str, Any]:
        """Detach a supervisor, wait for durable acknowledgement and return quickly."""
        request = Path(request_path).resolve()
        job_dir = request.parent
        process = subprocess.Popen(
            (
                sys.executable,
                "-m",
                "lambdaforge.controlplane.ProcessSupervisor",
                "serve",
                str(request),
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
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
    def serve(cls, request_path: str | Path) -> int:
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
                "state": JobState.STAGING.value,
                "created_at_utc": str(request.get("created_at_utc", now)),
                "updated_at_utc": now,
                "heartbeat_at_utc": now,
                "supervisor_identity": supervisor.to_dict(),
                "process_identity": None,
                "allocated_gpus": [],
                "allocated_cpus": [],
                "exit_code": None,
                "message": "Supervisor started.",
            },
        )
        leases: tuple[int, ...] = ()
        cpu_leases: tuple[int, ...] = ()
        process: subprocess.Popen[bytes] | None = None
        try:
            if bool(request.get("stage_source", False)):
                source = Path(str(request["source_work_dir"])).resolve()
                if not source.is_dir() or source.is_symlink():
                    raise RuntimeError(f"Unsafe or missing staged source: {source}")
                shutil.copytree(source, work_dir, dirs_exist_ok=True)
            resources = cls._required_mapping(request.get("resources", {}), "resources")
            cpu_leases = cls._wait_for_capacity(job_dir, request, supervisor)
            gpu_count = int(resources.get("gpu_count", 0))
            leases = cls._wait_for_gpus(job_dir, request, supervisor, gpu_count)
            current = cls._read_json(job_dir / "state.json") or {}
            if current.get("state") == JobState.CANCELLED.value:
                return 0
            environment = os.environ.copy()
            environment["LAMBDAFORGE_CLUSTER"] = str(request.get("cluster", "local"))
            dataset_registry = request.get("dataset_registry")
            if dataset_registry is not None:
                environment["LAMBDAFORGE_DATASET_REGISTRY"] = str(dataset_registry)
            if leases:
                environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(item) for item in leases)
            environment.setdefault("OMP_NUM_THREADS", str(max(1, len(cpu_leases))))
            stdout_path = job_dir / "stdout.log"
            stderr_path = job_dir / "stderr.log"
            with (
                stdout_path.open("ab", buffering=0) as stdout,
                stderr_path.open("ab", buffering=0) as stderr,
            ):
                process = subprocess.Popen(
                    command,
                    cwd=work_dir,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                    close_fds=True,
                )
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
            if process is not None and process.poll() is None:
                cls._terminate(
                    ProcessIdentity.create(process.pid, os.getpgid(process.pid), command, job_id)
                )
            cls._release_gpus(Path(str(request["lease_root"])), job_id, leases)
            resource_root = request.get("resource_lease_root")
            if resource_root is not None:
                cls._release_capacity(Path(str(resource_root)), job_id)

    @classmethod
    def control(cls, operation: str, job_dir: str | Path) -> dict[str, Any]:
        """Safely cancel, pause or resume only the identity recorded for this job."""
        directory = Path(job_dir).resolve()
        state = cls._required_mapping(cls._read_json(directory / "state.json"), "state")
        job_state = JobState(str(state["state"]))
        if operation == "cancel":
            if job_state.terminal:
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
                cls._terminate(ProcessIdentity.from_mapping(identity))
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
    def inventory(cls, root: str | Path) -> tuple[dict[str, Any], ...]:
        """Read only LambdaForge-owned job state directories under one exact root."""
        directory = Path(root).resolve()
        if not directory.is_dir():
            return ()
        states = []
        for child in sorted(directory.iterdir()):
            if child.is_symlink() or not child.is_dir() or not child.name.startswith("job-"):
                continue
            state = cls._read_json(child / "state.json")
            request = cls._read_json(child / "request.json")
            if state and request and state.get("job_id") == child.name:
                states.append({"state": state, "request": request})
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
            processes = (process, *process.children(recursive=True))
            cpu_time = 0.0
            rss_bytes = 0
            threads = 0
            descendants: set[int] = set()
            for child in processes:
                try:
                    with child.oneshot():
                        times = child.cpu_times()
                        cpu_time += float(times.user + times.system)
                        rss_bytes += int(child.memory_info().rss)
                        threads += int(child.num_threads())
                        descendants.add(child.pid)
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
            gpu = subprocess.run(
                (
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ),
                check=False,
                capture_output=True,
                text=True,
            )
            if gpu.returncode == 0:
                usage["gpu_memory_bytes"] = sum(
                    int(float(row[1])) * 1024**2
                    for line in gpu.stdout.splitlines()
                    if len(row := [item.strip() for item in line.split(",")]) == 2
                    and row[0].isdigit()
                    and int(row[0]) in descendants
                )
        except Exception:
            usage["process_observation"] = "unavailable"
        with (job_dir / "usage.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(usage, sort_keys=True) + "\n")
        cls._update_state(
            job_dir,
            {
                "heartbeat_at_utc": now,
                "updated_at_utc": now,
                "started_at_utc": state.get("started_at_utc", started),
                "observed_usage": usage,
            },
        )

    @classmethod
    def _wait_for_gpus(
        cls,
        job_dir: Path,
        request: Mapping[str, Any],
        supervisor: ProcessIdentity,
        count: int,
    ) -> tuple[int, ...]:
        if count <= 0:
            return ()
        lease_root = Path(str(request["lease_root"])).resolve()
        while True:
            state = cls._read_json(job_dir / "state.json") or {}
            if state.get("state") == JobState.CANCELLED.value:
                return ()
            allocated = cls._acquire_gpus(lease_root, supervisor, count)
            if allocated is not None:
                return allocated
            cls._update_state(
                job_dir,
                {
                    "state": JobState.QUEUED.value,
                    "updated_at_utc": cls._now(),
                    "heartbeat_at_utc": cls._now(),
                    "message": f"Waiting for {count} LambdaForge GPU lease(s).",
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
        root = Path(str(request["resource_lease_root"])).resolve()
        while True:
            allocated = cls._acquire_capacity(root, supervisor, cpu_count, ram_bytes)
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
                    "message": f"Waiting for {cpu_count} CPU core(s) and {ram_bytes} RAM bytes.",
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
        import psutil

        root.mkdir(parents=True, exist_ok=True)
        available_cpus = (
            tuple(sorted(os.sched_getaffinity(0)))
            if hasattr(os, "sched_getaffinity")
            else tuple(range(os.cpu_count() or 1))
        )
        if cpu_count > len(available_cpus):
            raise RuntimeError(
                f"Requested {cpu_count} CPU cores but only {len(available_cpus)} are available."
            )
        if ram_bytes > psutil.virtual_memory().total:
            raise RuntimeError("Requested RAM exceeds physical host memory.")
        with CrossProcessFileLock(
            root / ".leases.lock",
            shared=False,
            timeout_seconds=10.0,
            poll_interval_seconds=0.05,
        ):
            occupied: set[int] = set()
            reserved_ram = 0
            for path in root.glob("job-*.json"):
                lease = cls._read_json(path)
                raw = lease.get("supervisor_identity") if lease else None
                if not isinstance(raw, Mapping) or not ProcessIdentity.from_mapping(raw).matches():
                    path.unlink(missing_ok=True)
                    continue
                assert lease is not None
                occupied.update(int(item) for item in lease.get("cpus", ()))
                reserved_ram += int(lease.get("ram_bytes", 0))
            cpus = tuple(item for item in available_cpus if item not in occupied)[:cpu_count]
            free_ram = int(psutil.virtual_memory().available)
            if len(cpus) < cpu_count or ram_bytes > max(0, free_ram - reserved_ram):
                return None
            cls._write_json(
                root / f"{supervisor.job_id}.json",
                {
                    "lease_version": 1,
                    "job_id": supervisor.job_id,
                    "cpus": list(cpus),
                    "ram_bytes": ram_bytes,
                    "supervisor_identity": supervisor.to_dict(),
                    "created_at_utc": cls._now(),
                },
            )
            return cpus

    @classmethod
    def _release_capacity(cls, root: Path, job_id: str) -> None:
        if not root.exists():
            return
        with CrossProcessFileLock(
            root / ".leases.lock",
            shared=False,
            timeout_seconds=10.0,
            poll_interval_seconds=0.05,
        ):
            path = root / f"{job_id}.json"
            value = cls._read_json(path)
            if value and value.get("job_id") == job_id:
                path.unlink(missing_ok=True)

    @classmethod
    def _acquire_gpus(
        cls, root: Path, supervisor: ProcessIdentity, count: int
    ) -> tuple[int, ...] | None:
        root.mkdir(parents=True, exist_ok=True)
        with CrossProcessFileLock(
            root / ".leases.lock",
            shared=False,
            timeout_seconds=10.0,
            poll_interval_seconds=0.05,
        ):
            gpu_indices = cls._gpu_indices()
            if count > len(gpu_indices):
                raise RuntimeError(
                    f"Requested {count} GPUs but only {len(gpu_indices)} are visible."
                )
            occupied = cls._externally_occupied_gpus()
            available = []
            for index in gpu_indices:
                path = root / f"gpu-{index}.json"
                lease = cls._read_json(path)
                if lease:
                    raw = lease.get("supervisor_identity")
                    if isinstance(raw, Mapping) and ProcessIdentity.from_mapping(raw).matches():
                        continue
                    path.unlink(missing_ok=True)
                if index not in occupied:
                    available.append(index)
            if len(available) < count:
                return None
            selected = tuple(available[:count])
            for index in selected:
                cls._write_json(
                    root / f"gpu-{index}.json",
                    {
                        "lease_version": 1,
                        "gpu": index,
                        "job_id": supervisor.job_id,
                        "supervisor_identity": supervisor.to_dict(),
                        "created_at_utc": cls._now(),
                    },
                )
            return selected

    @classmethod
    def _release_gpus(cls, root: Path, job_id: str, indices: Sequence[int]) -> None:
        if not root.exists():
            return
        with CrossProcessFileLock(
            root / ".leases.lock",
            shared=False,
            timeout_seconds=10.0,
            poll_interval_seconds=0.05,
        ):
            for index in indices:
                path = root / f"gpu-{index}.json"
                value = cls._read_json(path)
                if value and value.get("job_id") == job_id:
                    path.unlink(missing_ok=True)

    @staticmethod
    def _gpu_indices() -> tuple[int, ...]:
        result = subprocess.run(
            ("nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"),
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            return ()
        return tuple(
            int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()
        )

    @staticmethod
    def _externally_occupied_gpus() -> set[int]:
        """Conservatively avoid GPUs with observable compute processes."""
        gpu = subprocess.run(
            ("nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"),
            check=False,
            capture_output=True,
            text=True,
        )
        processes = subprocess.run(
            ("nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"),
            check=False,
            capture_output=True,
            text=True,
        )
        if gpu.returncode or processes.returncode:
            return set()
        by_uuid = {
            values[1]: int(values[0])
            for line in gpu.stdout.splitlines()
            if len(values := [item.strip() for item in line.split(",")]) == 2
        }
        return {
            by_uuid[values[0]]
            for line in processes.stdout.splitlines()
            if len(values := [item.strip() for item in line.split(",")]) >= 2
            and values[0] in by_uuid
        }

    @classmethod
    def _terminate(cls, identity: ProcessIdentity, grace_seconds: float = 5.0) -> None:
        if not identity.matches():
            return
        ProcessGuard().terminate_process_tree(
            identity.pid, grace_seconds=grace_seconds, include_parent=False
        )
        try:
            os.killpg(identity.process_group, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            if not identity.matches():
                return
            time.sleep(0.05)
        if identity.matches():
            try:
                os.killpg(identity.process_group, signal.SIGKILL)
            except ProcessLookupError:
                pass

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
            raise SystemExit("Usage: ProcessSupervisor launch|serve|control|inventory PATH")
        operation, path, *rest = values
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
        raise SystemExit(f"Unknown ProcessSupervisor operation: {operation}")


if __name__ == "__main__":
    raise SystemExit(ProcessSupervisor.main())
