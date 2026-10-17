"""One light, shared GPU observation authority; no Torch or compute CUDA contexts.

The optional NVIDIA distribution is nvidia-ml-py (its import name is pynvml).
Actual refreshes, not individual callers, run in a bounded isolated process. Cached
unknown/stale evidence never authorizes admission and does not cancel scientific workers.
"""

from __future__ import annotations

import json
import math
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lambdaforge.controlplane.ProcessIdentity import ProcessIdentity
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.runtime.HostCoordination import (
    boot_identity,
    coordination_root,
    host_identity,
    secure_root,
)
from lambdaforge.runtime.ResourceSafetyPolicy import ResourceSafetyPolicy
from lambdaforge.work.atomic import atomic_write_json

_LIMIT = 1024 * 1024


class GPUObservationUnavailable(RuntimeError):
    """New admission must wait; this is not proof that any scientific worker failed."""


@dataclass(frozen=True)
class GPUObservationPolicy:
    telemetry_ttl: float = 5.0
    admission_ttl: float = 0.5
    minimum_interval: float = 0.25
    timeout: float = 5.0
    backoff_initial: float = 2.0
    backoff_maximum: float = 60.0

    def __post_init__(self) -> None:
        for value in asdict(self).values():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("GPU observation intervals must be finite and positive.")
        if self.minimum_interval > self.admission_ttl or self.admission_ttl > self.telemetry_ttl:
            raise ValueError("GPU observation minimum interval <= admission TTL <= telemetry TTL.")

    @classmethod
    def from_operational(cls, policy: ResourceSafetyPolicy) -> GPUObservationPolicy:
        return cls(
            telemetry_ttl=policy.telemetry_ttl,
            admission_ttl=policy.admission_ttl,
            minimum_interval=policy.minimum_interval,
            timeout=policy.probe_timeout,
            backoff_initial=policy.backoff_initial,
            backoff_maximum=policy.backoff_maximum,
        )


def _text(value: Any) -> str:
    return (value.decode() if isinstance(value, bytes) else str(value))[:200]


def _memory(free: Any, total: Any) -> tuple[int, int]:
    if type(free) is not int or type(total) is not int or not 0 <= free <= total or total <= 0:
        raise ValueError("GPU memory observation is incoherent.")
    return free, total


def collect_nvml(
    nvml: Any, *, suppressed: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Read GPU/MIG memory; missing measurements remain None, not fictitious zeroes."""
    import psutil

    rows: list[dict[str, Any]] = []
    blocked = {row["uuid"]: row for row in suppressed or []}

    def device(handle: Any, index: int, parent: str | None = None) -> None:
        uuid = _text(nvml.nvmlDeviceGetUUID(handle))
        row: dict[str, Any] = {
            "uuid": uuid,
            "index": index,
            "parent_uuid": parent,
            "name": None,
            "source": "nvml",
            "free_bytes": None,
            "total_bytes": None,
            "quality": "unavailable",
            "processes": [],
            "compute_process_count": None,
            "utilization_percent": None,
        }
        if uuid in blocked:
            row.update(blocked[uuid], suppressed=True)
            rows.append(row)
            return
        try:
            row["name"] = _text(nvml.nvmlDeviceGetName(handle))
        except Exception:
            pass
        try:
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)
            row["free_bytes"], row["total_bytes"] = _memory(memory.free, memory.total)
            row["quality"] = "measured"
        except Exception as error:
            row["error"] = type(error).__name__
        try:
            row["pci_bus_id"] = _text(nvml.nvmlDeviceGetPciInfo(handle).busId)
        except Exception:
            row["pci_bus_id"] = None
        try:
            row["utilization_percent"] = int(nvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        except Exception:
            pass
        try:
            processes = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
            row["compute_process_count"] = len(processes)
            for process in processes[:4096]:
                try:
                    owner = psutil.Process(process.pid)
                    if os.name == "posix" and owner.uids().real != os.getuid():
                        continue
                    used = process.usedGpuMemory
                    row["processes"].append(
                        {
                            "pid": int(process.pid),
                            "create_time": owner.create_time(),
                            "used_bytes": int(used)
                            if type(used) is int and 0 <= used < 2**63
                            else None,
                        }
                    )
                except (psutil.Error, OSError):
                    continue
        except Exception:
            pass
        rows.append(row)

    nvml.nvmlInit()
    try:
        count = int(nvml.nvmlDeviceGetCount())
        if not 0 <= count <= 256:
            raise ValueError("NVML device inventory is oversized.")
        for index in range(count):
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                device(handle, index)
            except Exception:
                # No guessed UUID for a lost handle. Other healthy devices remain observable;
                # the pinned grant resolver diagnoses the missing device separately.
                continue
            parent = rows[-1]["uuid"]
            try:
                mig_count = int(nvml.nvmlDeviceGetMaxMigDeviceCount(handle))
            except Exception:
                continue
            for instance in range(min(max(mig_count, 0), 256)):
                try:
                    child = nvml.nvmlDeviceGetMigDeviceHandleByIndex(handle, instance)
                except Exception:
                    continue
                try:
                    device(child, index, parent)
                except Exception:
                    continue
                try:
                    rows[-1]["aliases"] = [
                        f"MIG-{parent}/{nvml.nvmlDeviceGetGpuInstanceId(child)}"
                        f"/{nvml.nvmlDeviceGetComputeInstanceId(child)}"
                    ]
                except Exception:
                    pass
    finally:
        nvml.nvmlShutdown()
    return rows


def _smi_query(fields: str) -> str:
    with tempfile.TemporaryFile() as output:
        subprocess.run(
            ["nvidia-smi", fields, "--format=csv,noheader,nounits"],
            stdout=output,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            timeout=3,
            check=True,
        )
        output.seek(0, 2)
        if output.tell() > _LIMIT:
            raise ValueError("GPU tool inventory is oversized.")
        output.seek(0)
        return output.read().decode("utf-8")


def collect_smi() -> list[dict[str, Any]]:
    """Context-free fallback; queries are batched once per shared refresh, not per caller."""
    import psutil

    # Process attribution is optional. Failure is unknown occupancy, never an empty device.
    processes: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, int] = {}
    process_quality = False
    try:
        output = _smi_query("--query-compute-apps=gpu_uuid,pid,used_memory")
        if len(output.splitlines()) > 4096:
            raise ValueError("GPU process inventory is oversized.")
        for line in output.splitlines():
            uuid, raw_pid, used = (item.strip() for item in line.split(","))
            counts[uuid] = counts.get(uuid, 0) + 1
            try:
                process = psutil.Process(int(raw_pid))
                if os.name == "posix" and process.uids().real != os.getuid():
                    continue
                used_bytes = int(float(used) * 1024**2) if used not in {"N/A", "[N/A]"} else None
                if used_bytes is not None and not 0 <= used_bytes < 2**63:
                    used_bytes = None
                processes.setdefault(uuid, []).append(
                    {
                        "pid": process.pid,
                        "create_time": process.create_time(),
                        "used_bytes": used_bytes,
                    }
                )
            except (psutil.Error, OSError):
                continue
        process_quality = True
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    output = _smi_query("--query-gpu=index,uuid,name,memory.free,memory.total,utilization.gpu")
    rows = []
    for line in output.splitlines()[:256]:
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 6:
            raise ValueError("Unexpected GPU tool output.")
        free, total = _memory(int(float(fields[3]) * 1024**2), int(float(fields[4]) * 1024**2))
        rows.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "parent_uuid": None,
                "free_bytes": free,
                "total_bytes": total,
                "utilization_percent": float(fields[5]),
                "quality": "measured",
                "source": "nvidia-smi",
                "processes": processes.get(fields[1], []) if process_quality else [],
                "compute_process_count": counts.get(fields[1], 0) if process_quality else None,
            }
        )
    return rows


def collect() -> dict[str, Any]:
    """Isolated read-only backend. Import the official optional binding only here."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        version("nvidia-ml-py")
        import pynvml  # type: ignore[import-not-found]

        raw = os.environ.get("LAMBDAFORGE_GPU_OBSERVATION_SUPPRESSED", "[]")
        if len(raw) > _LIMIT:
            raise ValueError("GPU suppressed metadata oversized.")
        suppressed = json.loads(raw)
        if not isinstance(suppressed, list) or len(suppressed) > 512:
            raise ValueError("GPU suppressed metadata invalid.")
        rows = collect_nvml(pynvml, suppressed=suppressed)
    except (ImportError, PackageNotFoundError):
        rows = collect_smi()
    # A malfunctioning installed NVML is not reason to launch a second driver query.
    return {"devices": rows, "observed_at": time.time(), "quality": "measured"}


class GPUObservationCache:
    """Single-flight host cache with distinct informative and admission freshness.

    No process is created on cache hits. One isolated refresh serves concurrent callers.
    Failed/unresolved refreshes back off; an unreaped probe group blocks another spawn.
    The cache is never an external GPU ownership authority.
    """

    def __init__(
        self,
        root: Path | None = None,
        *,
        policy: GPUObservationPolicy | None = None,
        clock: Callable[[], float] = time.time,
        probe: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.root = secure_root(root) if root else coordination_root("gpu-observation")
        if policy is None:
            from lambdaforge.runtime.ResourceSafetyPolicy import ResourceSafetyPolicy

            operational = ResourceSafetyPolicy.inherited()
            policy = GPUObservationPolicy.from_operational(operational)
        self.policy = policy
        self.clock, self.probe = clock, probe or self._probe
        self.host, self.boot = host_identity(), boot_identity()

    def _read(self) -> dict[str, Any]:
        path = self.root / "observation.json"
        if not path.exists():
            return {}
        info = path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size > _LIMIT
            or os.name == "posix"
            and info.st_uid != os.getuid()
        ):
            raise ValueError("Unsafe GPU observation cache.")
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("Corrupt GPU observation cache.")
        return value if value.get("host") == self.host and value.get("boot") == self.boot else {}

    def read(self, *, admission: bool = False) -> dict[str, Any]:
        ttl = self.policy.admission_ttl if admission else self.policy.telemetry_ttl
        with CrossProcessFileLock(
            self.root / "observation.lock",
            shared=False,
            timeout_seconds=self.policy.timeout + 1,
            poll_interval_seconds=0.01,
        ):
            value = self._read()
            stats = value.setdefault("statistics", {})
            stats["requests"] = int(stats.get("requests", 0)) + 1
            now = self.clock()
            age = now - float(value.get("observed_at", float("-inf")))
            if value.get("quality") == "measured" and 0 <= age <= ttl:
                stats["cache_hits"] = int(stats.get("cache_hits", 0)) + 1
                value["age_seconds"] = age
            elif (
                now < float(value.get("retry_after", 0))
                or 0
                <= now - float(value.get("last_query_at", float("-inf")))
                < self.policy.minimum_interval
                or self._unreaped(value.get("probe_owner"))
            ):
                stats["stale_reads"] = int(stats.get("stale_reads", 0)) + 1
                value.update(
                    quality="unavailable",
                    age_seconds=age if math.isfinite(age) and age >= 0 else None,
                )
            else:
                value["last_query_at"] = now
                stats["real_queries"] = int(stats.get("real_queries", 0)) + 1
                started = time.monotonic()
                try:
                    sampled = self.probe()
                    if not isinstance(sampled.get("devices"), list):
                        raise ValueError("Invalid GPU observation response.")
                    observed_at = sampled.get("observed_at")
                    if (
                        not isinstance(observed_at, (int, float))
                        or isinstance(observed_at, bool)
                        or not math.isfinite(observed_at)
                        or not 0 <= self.clock() - observed_at <= ttl
                    ):
                        raise ValueError("GPU observation timestamp is stale/invalid.")
                    for row in sampled["devices"]:
                        if row.get("quality") == "measured":
                            _memory(row.get("free_bytes"), row.get("total_bytes"))
                    from lambdaforge.runtime.GPUDeviceHealth import reconcile_devices
                    from lambdaforge.runtime.ResourceSafetyPolicy import ResourceSafetyPolicy

                    operational = ResourceSafetyPolicy.inherited()
                    sampled["devices"] = reconcile_devices(
                        value.get("devices", []),
                        sampled["devices"],
                        now=self.clock(),
                        initial=operational.device_backoff_initial,
                        maximum=operational.device_backoff_maximum,
                    )
                    value.update(sampled, consecutive_errors=0, retry_after=0, probe_owner=None)
                    value.update(age_seconds=max(0.0, self.clock() - observed_at), error=None)
                except Exception as error:
                    owner = self._read().get("probe_owner")
                    failures = int(value.get("consecutive_errors", 0)) + 1
                    stats["errors"] = int(stats.get("errors", 0)) + 1
                    if isinstance(error, subprocess.TimeoutExpired):
                        stats["timeouts"] = int(stats.get("timeouts", 0)) + 1
                    value.update(
                        quality="unavailable",
                        error=type(error).__name__,
                        consecutive_errors=failures,
                        probe_owner=owner,
                        retry_after=self.clock()
                        + min(
                            self.policy.backoff_maximum,
                            self.policy.backoff_initial * 2 ** min(failures - 1, 10),
                        ),
                    )
                stats["last_latency_seconds"] = time.monotonic() - started
            value.update(version=1, host=self.host, boot=self.boot, statistics=stats)
            atomic_write_json(self.root / "observation.json", value)
            return value

    @staticmethod
    def _unreaped(raw: Any) -> bool:
        if not isinstance(raw, Mapping):
            return False
        try:
            identity = ProcessIdentity.from_mapping(raw)
            if identity.storage_status() not in {"dead", "reused"}:
                return True
            if os.name == "posix":
                import psutil

                for process in psutil.process_iter(["pid", "uids"]):
                    try:
                        if os.getpgid(process.pid) != identity.process_group:
                            continue
                        if process.info["uids"] is None:
                            return True
                        if process.info["uids"].real != os.getuid():
                            continue
                        if (
                            os.getpgid(process.pid) == identity.process_group
                            and process.create_time() >= identity.create_time
                            and process.status() not in {psutil.STATUS_DEAD, psutil.STATUS_ZOMBIE}
                        ):
                            return True
                    except (psutil.NoSuchProcess, ProcessLookupError):
                        continue
                    except (psutil.AccessDenied, OSError):
                        return True
            return False
        except (OSError, ValueError, KeyError, TypeError):
            return True

    def _probe(self) -> dict[str, Any]:
        value = dict(self._bounded_helper("GPUObservation"))
        if value.get("backend_error"):
            raise RuntimeError(str(value["backend_error"])[:120])
        return value

    def visible_identity(self) -> list[str]:
        """Bound the one-time legacy ordinal resolution using the same private probe owner."""
        with CrossProcessFileLock(
            self.root / "observation.lock",
            shared=False,
            timeout_seconds=self.policy.timeout + 1,
            poll_interval_seconds=0.01,
        ):
            if self._unreaped(self._read().get("probe_owner")):
                raise RuntimeError("Previous GPU metadata helper ownership remains unresolved.")
            return list(self._bounded_helper("GPUDeviceIdentity"))

    def _bounded_helper(self, module: str) -> Any:
        if module not in {"GPUObservation", "GPUDeviceIdentity"}:
            raise ValueError("Only native read-only GPU helpers are supported.")
        value = self._read()
        from lambdaforge.runtime.GPUDeviceHealth import suppressed_devices

        environment = os.environ.copy()
        environment["LAMBDAFORGE_GPU_OBSERVATION_SUPPRESSED"] = json.dumps(
            suppressed_devices(value.get("devices", []), self.clock())
        )
        value.update(host=self.host, boot=self.boot, probe_owner={"launch_pending": True})
        # A filesystem failure must not create untracked driver helpers in every caller.
        atomic_write_json(self.root / "observation.json", value)
        with tempfile.TemporaryFile() as output:
            try:
                process = subprocess.Popen(
                    [sys.executable, "-m", f"lambdaforge.runtime.{module}"],
                    stdout=output,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    start_new_session=os.name == "posix",
                    env=environment,
                )
            except Exception:
                value["probe_owner"] = None  # No helper was created.
                atomic_write_json(self.root / "observation.json", value)
                raise
            try:
                owner = ProcessIdentity.create(process.pid, process.pid, (), "gpu-observation")
                value = self._read()
                value.update(host=self.host, boot=self.boot, probe_owner=owner.to_dict())
                atomic_write_json(self.root / "observation.json", value)
            except Exception:
                self._stop_helper(process)
                if process.poll() is not None:
                    value["probe_owner"] = None  # Positive reaping, not timeout-based guessing.
                    atomic_write_json(self.root / "observation.json", value)
                raise RuntimeError("Cannot attest/persist GPU observation ownership.") from None
            try:
                process.wait(timeout=self.policy.timeout)
            except subprocess.TimeoutExpired:
                self._stop_helper(process)
                raise
            if process.returncode:
                raise RuntimeError("GPU observation backend unavailable.")
            output.seek(0, 2)
            if output.tell() > _LIMIT:
                raise ValueError("GPU observation backend response exceeds 1 MiB.")
            output.seek(0)
            return json.load(output)

    @staticmethod
    def _stop_helper(process: subprocess.Popen) -> None:
        """Only the new private helper group; never a scientific worker or external PID."""
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            pass  # Saved owner blocks another spawn until its group is positively absent.


if __name__ == "__main__":
    try:
        observation = collect()
    except Exception as backend_error:
        observation = {"backend_error": type(backend_error).__name__}
    print(json.dumps(observation, allow_nan=False))
