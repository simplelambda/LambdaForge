"""Read-only physical/cgroup host capacity and honest process-tree memory attribution."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _counter(path: Path) -> int | None:
    value = path.read_text().strip()
    if value == "max":
        return None
    number = int(value)
    if number < 0:
        raise ValueError("Negative cgroup memory counter.")
    return number


def _cgroup_directory() -> tuple[Path, Path] | None:
    """Resolve the unified hierarchy through its actual mount, including containers."""
    relative = next(
        (
            line[3:]
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        ),
        None,
    )
    if relative is None or ".." in Path(relative).parts:
        return None
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        before, separator, after = line.partition(" - ")
        if not separator or after.split()[0] != "cgroup2":
            continue
        fields = before.split()
        mount_root = Path(fields[3].replace("\\040", " "))
        mount = Path(fields[4].replace("\\040", " "))
        try:
            suffix = Path(relative).relative_to(mount_root)
        except ValueError:
            continue
        directory = mount / suffix
        if directory.is_dir() and not directory.is_symlink():
            return directory, mount
    return None


def observe_host(*, cgroup: tuple[Path, Path] | None = None) -> dict[str, Any]:
    """Capacity is bounded by physical availability and every observable ancestor limit.

    Unknown cgroup evidence is explicit and never represented by zero consumption. This
    observer writes no cgroup controls and does not infer an enforced allocation from YAML.
    """
    import psutil

    memory = psutil.virtual_memory()
    result: dict[str, Any] = {
        "host_total_bytes": int(memory.total),
        "host_available_bytes": int(memory.available),
        "effective_total_bytes": int(memory.total),
        "effective_available_bytes": int(memory.available),
        "quality": "measured",
        "cgroup": {"available": False, "limits": [], "errors": []},
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(range(os.cpu_count() or 1)),
    }
    result["effective_cpu_cores"] = len(result["cpu_affinity"])
    result["cgroup"]["cpu_quota_cores"] = None
    try:
        hierarchy = cgroup if cgroup is not None else _cgroup_directory()
    except (OSError, ValueError, IndexError) as error:
        result["cgroup"]["errors"].append(type(error).__name__)
        return result
    if hierarchy is None:
        return result
    directory, mount = hierarchy
    directory.relative_to(mount)
    result["cgroup"]["available"] = True
    while True:
        try:
            quota_path = directory / "cpu.max"
            if quota_path.exists():
                quota, period = quota_path.read_text().split()
                if int(period) <= 0 or quota != "max" and int(quota) <= 0:
                    raise ValueError("Invalid cgroup CPU quota.")
                if quota != "max":
                    cores = int(quota) / int(period)
                    previous = result["cgroup"]["cpu_quota_cores"]
                    result["cgroup"]["cpu_quota_cores"] = (
                        cores if previous is None else min(previous, cores)
                    )
                    if result["effective_cpu_cores"] is not None:
                        result["effective_cpu_cores"] = min(
                            result["effective_cpu_cores"], math.ceil(cores)
                        )
            cpuset = directory / "cpuset.cpus.effective"
            if cpuset.exists():
                values: set[int] = set()
                for item in cpuset.read_text().strip().split(","):
                    if not item:
                        continue
                    low, separator, high = item.partition("-")
                    first, last = int(low), int(high) if separator else int(low)
                    if not 0 <= first <= last <= 65535:
                        raise ValueError("Invalid effective cgroup CPU set.")
                    values.update(range(first, last + 1))
                if values:
                    result["cpu_affinity"] = sorted(
                        set(result["cpu_affinity"]).intersection(values)
                    )
                    if result["effective_cpu_cores"] is not None:
                        result["effective_cpu_cores"] = min(
                            result["effective_cpu_cores"], len(result["cpu_affinity"])
                        )
        except (OSError, ValueError) as error:
            result["cgroup"]["errors"].append(type(error).__name__)
            result["cpu_quality"] = "unavailable"
            result["effective_cpu_cores"] = None
        maximum = directory / "memory.max"
        if maximum.exists():
            try:
                limit, used = _counter(maximum), _counter(directory / "memory.current")
                if used is None:
                    raise ValueError("Unmeasured cgroup consumption.")
                result["cgroup"]["limits"].append({"limit_bytes": limit, "current_bytes": used})
                if limit is not None:
                    result["effective_total_bytes"] = min(result["effective_total_bytes"], limit)
                    result["effective_available_bytes"] = min(
                        result["effective_available_bytes"], max(0, limit - used)
                    )
            except (OSError, ValueError) as error:
                result["cgroup"]["errors"].append(type(error).__name__)
                result["quality"] = "unavailable"
                result["effective_available_bytes"] = None
                break
        if directory == hierarchy[0]:
            try:
                result["cgroup"]["pressure"] = _pressure(directory / "memory.pressure")
                result["cgroup"]["events"] = {
                    key: int(value)
                    for key, value in (
                        line.split()
                        for line in (directory / "memory.events").read_text().splitlines()
                    )
                }
            except (OSError, ValueError):
                result["cgroup"]["pressure"] = None
        if directory == mount:
            break
        directory = directory.parent
    return result


def _pressure(path: Path) -> dict[str, dict[str, float]]:
    return {
        fields[0]: {key: float(value) for key, value in (item.split("=") for item in fields[1:])}
        for fields in (line.split() for line in path.read_text().splitlines())
    }


def process_tree_memory(pid: int, create_time: float) -> dict[str, Any]:
    """Attribute only a verified native lifetime; RSS sum is labelled, not unique RAM."""
    import psutil

    result: dict[str, Any] = {
        "quality": "unavailable",
        "rss_bytes": None,
        "pss_bytes": None,
        "uss_bytes": None,
        "process_count": 0,
    }
    try:
        root = psutil.Process(pid)
        if root.create_time() != create_time:
            return result
        processes = [root, *root.children(recursive=True)]
    except (psutil.Error, OSError):
        return result
    rss = pss = uss = 0
    full = True
    for process in processes:
        try:
            if os.name == "posix" and process.uids().real != os.getuid():
                full = False
                continue
            rss += process.memory_info().rss
            result["process_count"] += 1
            try:
                memory = process.memory_full_info()
                pss += memory.pss
                uss += memory.uss
            except (psutil.Error, OSError, AttributeError):
                full = False
        except psutil.NoSuchProcess:
            continue
        except (psutil.Error, OSError):
            full = False
    if result["process_count"]:
        result.update(
            quality="measured" if full else "partial",
            rss_bytes=rss,
            pss_bytes=pss if full else None,
            uss_bytes=uss if full else None,
        )
    return result


def pending_ram_bytes(requested: int, observed: Mapping[str, Any]) -> int:
    """Only unmaterialized demand: physical free already excludes resident RAM."""
    resident = observed.get("pss_bytes")
    # RSS can double-count shared mappings. Without PSS, conservatively retain the full
    # commitment rather than using an overestimated resident value to admit unsafely.
    return max(0, requested - resident) if isinstance(resident, int) else requested
