"""Sanitized bounded administrator report from native saved operational observations."""

from __future__ import annotations

import math
import platform
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from lambdaforge._version import VERSION
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.controlplane.OperationalHealth import timestamp


def _number(value: Any) -> int | float | None:
    if isinstance(value, int) and abs(value) >= 2**63:
        return None
    return (
        value
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
        else None
    )


def _fields(value: Any, fields: tuple[str, ...]) -> dict[str, int | float | None]:
    value = value if isinstance(value, Mapping) else {}
    return {key: _number(value.get(key)) for key in fields}


class ResourceDiagnosticReport:
    """Read only project Job metadata: no transport, user log, environment or asset access."""

    @staticmethod
    def build(
        store: JobStore,
        *,
        cluster: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 500,
    ) -> dict[str, Any]:
        end = timestamp(until) if until else datetime.now(timezone.utc)
        start = timestamp(since) if since else end - timedelta(days=1) if end else None
        if start is None or end is None or start > end:
            raise ValueError("Resource report requires valid ISO-8601 since <= until timestamps.")
        snapshot = store.bounded_metadata(limit=limit)
        rows = []
        for record in snapshot["records"]:
            if cluster is not None and record.cluster != cluster:
                continue
            remote = record.metadata.get("remote_state", {})
            remote = remote if isinstance(remote, Mapping) else {}
            usage = remote.get("observed_usage", {})
            usage = usage if isinstance(usage, Mapping) else {}
            at = timestamp(usage.get("timestamp_utc"))
            saved = timestamp(record.updated_at_utc)
            selected = at or saved
            if selected is None or not start <= selected <= end:
                continue
            observation = usage.get("gpu_observation", {})
            observation = observation if isinstance(observation, Mapping) else {}
            memory = usage.get("memory_attribution", {})
            capacity = usage.get("host_capacity", {})
            capacity = capacity if isinstance(capacity, Mapping) else {}
            cgroup = capacity.get("cgroup", {})
            cgroup = cgroup if isinstance(cgroup, Mapping) else {}
            counters = _fields(cgroup.get("events", {}), ("low", "high", "max", "oom", "oom_kill"))
            history = remote.get("usage_history_summary", {})
            history = history if isinstance(history, Mapping) else {}
            rows.append(
                {
                    "job_id": record.job_id,
                    "state": record.state.value,
                    "execution_type": record.scheduler,
                    "observed_at_utc": at.isoformat() if at else None,
                    "saved_at_utc": record.updated_at_utc,
                    "scientific_state_is_separate": True,
                    "requested": _fields(
                        record.resources,
                        ("cpu_cores", "ram_bytes", "gpu_count", "gpu_memory_bytes"),
                    ),
                    "observed": _fields(
                        usage,
                        (
                            "cpu_percent",
                            "cpu_time_seconds",
                            "rss_bytes",
                            "threads",
                            "process_count",
                            "gpu_memory_bytes",
                        ),
                    ),
                    "memory_attribution": _fields(
                        memory, ("rss_bytes", "pss_bytes", "uss_bytes", "process_count")
                    ),
                    "host_capacity": _fields(
                        capacity,
                        (
                            "host_total_bytes",
                            "host_available_bytes",
                            "effective_total_bytes",
                            "effective_available_bytes",
                        ),
                    ),
                    "cgroup_events": counters,
                    "usage_history": {
                        "counters": _fields(history, ("sample_count", "compacted_windows")),
                        "lifetime_sampled_peaks": _fields(
                            history.get("lifetime_sampled_peaks", {}),
                            (
                                "rss_bytes",
                                "gpu_memory_bytes",
                                "process_count",
                                "threads",
                                "cpu_percent",
                            ),
                        ),
                        "peak_scope": "Job lifetime sampled maxima, "
                        "not exact requested-interval peaks",
                    },
                    "gpu_observation": {
                        "quality": observation.get("quality")
                        if observation.get("quality")
                        in {"measured", "unavailable", "not-requested"}
                        else "unknown",
                        "statistics": _fields(
                            observation.get("statistics", {}),
                            (
                                "requests",
                                "real_queries",
                                "cache_hits",
                                "errors",
                                "timeouts",
                                "last_latency_seconds",
                            ),
                        ),
                    },
                }
            )
        return {
            "resource_report_version": 1,
            "lambdaforge_version": VERSION,
            "report_platform": {"system": platform.system(), "release": platform.release()},
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "interval": {"since": start.isoformat(), "until": end.isoformat()},
            "scope": "project-owned saved Job metadata; no provider refresh",
            "jobs": rows,
            "job_count": len(rows),
            "coverage": {
                "skipped_records": snapshot["skipped"],
                "truncated_records": snapshot["truncated"],
                "complete_timeline": False,
                "peak_concurrency_measured": False,
                "note": "Latest saved observations only. Missing historical samples, "
                "Run OOMs, grants or driver events are unknown, not zero.",
            },
            "causality": "Operational correlation does not establish the cause of a host reboot.",
            "sanitization": "Allowlisted numeric metadata only; no commands, paths, "
            "environment, logs, credentials, datasets, checkpoints or other users' PIDs.",
        }

    @staticmethod
    def summary(value: Mapping[str, Any]) -> str:
        interval = value["interval"]
        lines = [
            "RESOURCE DIAGNOSTIC REPORT · saved project evidence",
            f"Interval: {interval['since']} → {interval['until']}",
            f"Jobs: {value['job_count']} · LambdaForge {value['lambdaforge_version']}",
            "",
        ]
        for row in value["jobs"]:
            usage = row["observed"]
            lines.extend(
                [
                    f"{row['job_id']} · {row['state']} · {row['execution_type']}",
                    f"  Observed: {row['observed_at_utc'] or 'not recorded'}",
                    f"  Processes: {usage['process_count']} · threads: {usage['threads']} · "
                    f"CPU: {usage['cpu_percent']}%",
                    f"  RAM RSS bytes: {usage['rss_bytes']} · "
                    f"GPU bytes: {usage['gpu_memory_bytes']}",
                    f"  GPU observation: {row['gpu_observation']['quality']}",
                ]
            )
        lines.extend(["", str(value["coverage"]["note"]), str(value["causality"])])
        return "\n".join(lines)
