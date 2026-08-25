"""Responsive, dependency-light interactive control-plane monitor."""

from __future__ import annotations

import multiprocessing
import os
import select
import shutil
import sys
import time
from collections import defaultdict, deque
from collections.abc import Mapping
from contextlib import AbstractContextManager
from multiprocessing.connection import Connection
from typing import Any, TextIO

from lambdaforge.cli.common import age
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.OverviewService import OverviewService
from lambdaforge.controlplane.WorkService import WorkService


def _ram(observed: Mapping[str, Any]) -> float | None:
    total, available = observed.get("ram_total_bytes"), observed.get("ram_available_bytes")
    if not isinstance(total, (int, float)) or not total or not isinstance(available, (int, float)):
        return None
    return 100 * (1 - float(available) / float(total))


def _gpu(observed: Mapping[str, Any]) -> float | None:
    values = observed.get("gpus", ())
    if not isinstance(values, (list, tuple)):
        return None
    percentages = [
        float(item.get("utilization_percent", 0)) for item in values if isinstance(item, Mapping)
    ]
    return sum(percentages) / len(percentages) if percentages else None


def _seconds(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    value = max(0, int(value))
    if value < 60:
        return f"{value}s"
    if value < 3600:
        return f"{value // 60}m{value % 60:02d}s"
    if value < 86400:
        return f"{value // 3600}h{value % 3600 // 60:02d}m"
    return f"{value // 86400}d{value % 86400 // 3600:02d}h"


def _bytes(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(amount) < 1024 or unit == "TiB":
            return f"{amount:.0f}{unit}" if unit in {"B", "KiB"} else f"{amount:.1f}{unit}"
        amount /= 1024
    return "-"


def _screen(lines: list[str], *, width: int, height: int) -> str:
    """Fit a screen while preserving its final help or confirmation line."""
    visible = lines if len(lines) <= height else [*lines[: max(0, height - 1)], lines[-1]]
    return "\n".join(line[:width] for line in visible)


class ResourceHistory:
    """Retain a configurable in-memory window of total and personal observations."""

    def __init__(self, window_seconds: float = 60) -> None:
        if window_seconds < 1:
            raise ValueError("Resource history must cover at least one second.")
        self.window_seconds = float(window_seconds)
        self.values: dict[str, deque[tuple[float, dict[str, float | None]]]] = defaultdict(deque)

    def record(self, payload: Mapping[str, Any]) -> None:
        now = time.time()
        for cluster in payload.get("clusters", ()):
            if not isinstance(cluster, Mapping):
                continue
            name = str(cluster.get("cluster", "-"))
            observed = cluster.get("observed", {})
            observed = observed if isinstance(observed, Mapping) else {}
            personal = cluster.get("personal", {})
            personal = personal if isinstance(personal, Mapping) else {}
            mine = personal.get("observed", {})
            mine = mine if isinstance(mine, Mapping) else {}
            cpu_total, ram_total = observed.get("cpu_total"), observed.get("ram_total_bytes")
            gpu_total = (
                sum(
                    float(item.get("memory_total_bytes", 0) or 0)
                    for item in observed.get("gpus", ())
                    if isinstance(item, Mapping)
                )
                if isinstance(observed.get("gpus", ()), (list, tuple))
                else 0
            )
            has_actual = bool(mine.get("job_count"))
            sample = {
                "cpu": float(observed["cpu_load"])
                if isinstance(observed.get("cpu_load"), (int, float))
                else None,
                "ram": _ram(observed),
                "gpu": _gpu(observed),
                "my_cpu": 100 * float(mine.get("cpu_percent", 0) or 0) / float(cpu_total)
                if has_actual and isinstance(cpu_total, (int, float)) and cpu_total
                else None,
                "my_ram": 100 * float(mine.get("rss_bytes", 0) or 0) / float(ram_total)
                if has_actual and isinstance(ram_total, (int, float)) and ram_total
                else None,
                "my_gpu": 100 * float(mine.get("gpu_memory_bytes", 0) or 0) / gpu_total
                if has_actual and gpu_total
                else None,
            }
            self.values[name].append((now, sample))
            while self.values[name] and self.values[name][0][0] < now - self.window_seconds:
                self.values[name].popleft()

    def series(self, cluster: str, key: str) -> tuple[float | None, ...]:
        return tuple(sample.get(key) for _, sample in self.values.get(cluster, ()))


def _move_overview_selection(
    focus: str,
    selected_job: int,
    selected_cluster: int,
    *,
    direction: int,
    job_count: int,
    cluster_count: int,
) -> tuple[str, int, int]:
    """Move through clusters followed by jobs as one vertical selection sequence."""
    if direction not in {-1, 1}:
        raise ValueError("Overview selection direction must be -1 or 1.")
    if focus == "jobs":
        if direction < 0 and selected_job == 0 and cluster_count:
            return "clusters", selected_job, cluster_count - 1
        selected_job = max(0, min(max(0, job_count - 1), selected_job + direction))
        return focus, selected_job, selected_cluster
    if direction > 0 and selected_cluster >= max(0, cluster_count - 1) and job_count:
        return "jobs", 0, selected_cluster
    selected_cluster = max(0, min(max(0, cluster_count - 1), selected_cluster + direction))
    return focus, selected_job, selected_cluster


def _selected_job_id(item: Mapping[str, Any]) -> str:
    """Resolve a semantic work row or an advanced job row to its operational job."""
    value = item.get("primary_job_id", item.get("job_id", ""))
    return str(value)


def _work_attempt_jobs(
    payload: Mapping[str, Any], work: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    """Join one semantic Work's human attempt history to observable Job facts."""
    jobs = payload.get("jobs", {})
    jobs = jobs if isinstance(jobs, Mapping) else {}
    raw = {
        str(item.get("job_id", "")): item
        for item in jobs.get("items", ())
        if isinstance(item, Mapping)
    }
    history = work.get("attempt_history", ())
    history = history if isinstance(history, (list, tuple)) else ()
    if history:
        output: list[Mapping[str, Any]] = []
        for fallback, attempt in enumerate(history, 1):
            if not isinstance(attempt, Mapping):
                continue
            job_id = str(attempt.get("job_id", ""))
            output.append(
                {
                    **dict(attempt),
                    **dict(raw.get(job_id, {})),
                    "attempt_number": int(attempt.get("number", fallback)),
                    "attempt_label": str(attempt.get("label") or f"Attempt {fallback}"),
                }
            )
        return output
    identifiers = work.get("job_ids", ())
    identifiers = identifiers if isinstance(identifiers, (list, tuple)) else ()
    if not identifiers and work.get("primary_job_id"):
        identifiers = (work["primary_job_id"],)
    return [
        {
            **dict(raw.get(str(job_id), {"job_id": str(job_id)})),
            "attempt_number": number,
            "attempt_label": f"Attempt {number}",
        }
        for number, job_id in enumerate(identifiers, 1)
    ]


def _job_label(payload: Mapping[str, Any], job: Mapping[str, Any]) -> str:
    """Return a human Work/Attempt label without exposing an operational ID."""
    job_id = str(job.get("job_id", ""))
    work = payload.get("work", {})
    work = work if isinstance(work, Mapping) else {}
    for item in work.get("items", ()):
        if not isinstance(item, Mapping):
            continue
        attempts = _work_attempt_jobs(payload, item)
        for attempt in attempts:
            if str(attempt.get("job_id", "")) == job_id:
                return f"{item.get('name', 'Work')} · {attempt.get('attempt_label', 'Attempt')}"
    metadata = job.get("metadata", {})
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return f"{metadata.get('name') or job.get('job_type') or 'Work'} · Attempt 1"


class MonitorRenderer:
    """Render immutable snapshots while leaving I/O and actions to the monitor."""

    _SPARK = "▁▂▃▄▅▆▇█"

    @classmethod
    def render(
        cls,
        payload: Mapping[str, Any],
        *,
        selected: int = 0,
        selected_cluster: int = 0,
        focus: str = "jobs",
        history: ResourceHistory | None = None,
        log_text: str = "",
        message: str = "",
        width: int = 120,
        height: int | None = None,
        view: str = "research",
    ) -> str:
        del log_text  # compatibility with the original public renderer signature
        jobs = payload.get("jobs", {})
        jobs = jobs if isinstance(jobs, Mapping) else {}
        raw_items = jobs.get("items", ())
        raw_items = raw_items if isinstance(raw_items, list) else []
        work = payload.get("work", {})
        work = work if isinstance(work, Mapping) else {}
        work_items = work.get("items", ())
        work_items = work_items if isinstance(work_items, list) else []
        research_view = view == "research" and bool(work_items)
        items = work_items if research_view else raw_items
        states = jobs.get("by_state", {})
        states = states if isinstance(states, Mapping) else {}
        clusters = [item for item in payload.get("clusters", ()) if isinstance(item, Mapping)]
        terminal_height = height or shutil.get_terminal_size((width, 30)).lines
        cluster_capacity = max(1, min(len(clusters) or 1, max(2, terminal_height // 4)))
        cluster_start = max(0, min(selected_cluster, len(clusters) - 1) - cluster_capacity + 1)
        generated = str(payload.get("generated_at_utc", ""))
        lines = [
            "LambdaForge research activity" if research_view else "LambdaForge advanced jobs",
            (
                f"updated {generated[11:19] or '-'}  "
                f"preparing={states.get('preparing', 0)} "
                f"staging={states.get('staging', 0)} queued={states.get('queued', 0)} "
                f"running={states.get('running', 0)} failed={states.get('failed', 0)} "
                f"total={jobs.get('total', 0)}"
            ),
            "",
            "CLUSTERS (whole-cluster observation; Enter/→ opens detail)",
            "  CLUSTER           STATUS   CPU    RAM    GPU    CORES  RAM FREE   GPUS  JOBS",
        ]
        for offset, cluster in enumerate(
            clusters[cluster_start : cluster_start + cluster_capacity]
        ):
            index = cluster_start + offset
            name = str(cluster.get("cluster", "-"))
            observed = cluster.get("observed", {})
            observed = observed if isinstance(observed, Mapping) else {}
            personal = cluster.get("personal", {})
            personal = personal if isinstance(personal, Mapping) else {}
            gpus = observed.get("gpus", ())
            gpus = gpus if isinstance(gpus, (list, tuple)) else ()
            marker = "▶" if focus == "clusters" and index == selected_cluster else " "
            lines.append(
                f"{marker} {name:<17.17} "
                f"{'online' if cluster.get('online') else 'offline':<7} "
                f"{cls._number(observed.get('cpu_load')):<6} "
                f"{cls._number(_ram(observed)):<6} {cls._number(_gpu(observed)):<6} "
                f"{str(observed.get('cpu_total', '-')):<6.6} "
                f"{_bytes(observed.get('ram_available_bytes')):<10.10} "
                f"{len(gpus):<5} {str(personal.get('active_jobs', 0)):<4}"
            )
        if cluster_start or cluster_start + cluster_capacity < len(clusters):
            last_cluster = min(len(clusters), cluster_start + cluster_capacity)
            lines.append(f"  clusters {cluster_start + 1}-{last_cluster} of {len(clusters)}")
        lines.append("")
        lines.append(
            "  WORK                       KIND          STATE       TARGET             "
            "PROGRESS       ATT  AGE"
            if research_view
            else "  JOB                             NAME             TYPE           "
            "STATE       CLUSTER      RUN      AGE      USED / REQUESTED"
        )
        capacity = max(1, terminal_height - len(lines) - 4)
        start = max(0, min(selected, len(items) - 1) - capacity + 1)
        visible = items[start : start + capacity]
        for offset, job in enumerate(visible):
            index = start + offset
            marker = "▶" if focus == "jobs" and index == selected else " "
            if research_view:
                progress = job.get("progress", {}) if isinstance(job, Mapping) else {}
                progress = progress if isinstance(progress, Mapping) else {}
                complete = progress.get("completed")
                total = progress.get("total")
                progress_text = (
                    f"{complete if complete is not None else '?'}/"
                    f"{total if total is not None else '?'} {progress.get('unit', 'units')}"
                )
                lines.append(
                    f"{marker} {str(job.get('name', '-')):<26.26} "
                    f"{str(job.get('kind', '-')):<13.13} "
                    f"{str(job.get('state', '-')):<11.11} "
                    f"{str(job.get('cluster', '-')):<18.18} "
                    f"{progress_text:<14.14} {str(job.get('attempts', 0)):<4.4} "
                    f"{age(str(job.get('created_at_utc', ''))):<8}"
                )
                continue
            metadata = job.get("metadata", {}) if isinstance(job, Mapping) else {}
            metadata = metadata if isinstance(metadata, Mapping) else {}
            timing = job.get("timing", {}) if isinstance(job, Mapping) else {}
            timing = timing if isinstance(timing, Mapping) else {}
            runtime = timing.get("runtime_seconds")
            runtime_text = (
                "waiting"
                if runtime is None and job.get("state") in {"preparing", "staging", "queued"}
                else _seconds(runtime if runtime is not None else timing.get("elapsed_seconds"))
            )
            lines.append(
                f"{marker} {str(job.get('job_id', '-')):<32.32} "
                f"{str(metadata.get('name', '-')):<16.16} "
                f"{str(job.get('job_type', '-')):<14.14} "
                f"{str(job.get('state', '-')):<11.11} "
                f"{str(job.get('cluster', '-')):<12.12} {runtime_text:<8.8} "
                f"{age(str(job.get('created_at_utc', ''))):<8} {cls._job_usage(job)}"
            )
        if visible:
            job = items[min(selected, len(items) - 1)]
            if research_view:
                lines.extend(
                    (
                        "",
                        f"Selected: {job.get('name')}  "
                        f"revision={job.get('scientific_revision') or 'legacy'}  "
                        f"attempts={job.get('attempts', 1)}",
                    )
                )
            else:
                metadata = job.get("metadata", {})
                metadata = metadata if isinstance(metadata, Mapping) else {}
                lines.extend(
                    (
                        "",
                        f"Selected: {job.get('job_id')}  "
                        f"scheduler={job.get('scheduler_id') or 'not acknowledged'}  "
                        f"phase={metadata.get('submission_phase', '-')}",
                    )
                )
        lines.extend(
            (
                "",
                message
                or "↑/↓ select · Enter/→ open · ← back · x cancel · d delete · "
                "D clear terminal history · r refresh · q quit",
            )
        )
        return _screen(lines, width=width, height=terminal_height)

    @classmethod
    def _metric(cls, value: object, history: ResourceHistory | None, cluster: str, key: str) -> str:
        current = float(value) if isinstance(value, (int, float)) else None
        series = history.series(cluster, key) if history else ()
        return (
            f"{current:3.0f}% {cls._spark(series):<10}"
            if current is not None
            else f"  ?% {cls._spark(series):<10}"
        )

    @staticmethod
    def _number(value: object) -> str:
        return f"{float(value):.0f}%" if isinstance(value, (int, float)) else "?"

    @classmethod
    def _spark(cls, values: tuple[float | None, ...], width: int = 10) -> str:
        selected = values[-width:]
        if not selected or all(value is None for value in selected):
            return "·"
        return "".join(
            "·" if value is None else cls._SPARK[min(7, max(0, round(float(value) * 7 / 100)))]
            for value in selected
        )

    @staticmethod
    def _job_usage(job: Mapping[str, Any]) -> str:
        usage = job.get("usage", {})
        usage = usage if isinstance(usage, Mapping) else {}
        observed = usage.get("observed", {})
        observed = observed if isinstance(observed, Mapping) else {}
        requested = usage.get("requested", job.get("resources", {}))
        requested = requested if isinstance(requested, Mapping) else {}
        actual = (
            f"C{float(observed.get('cpu_percent', 0) or 0) / 100:.1f} "
            f"R{_bytes(observed.get('rss_bytes'))} "
            f"V{_bytes(observed.get('gpu_memory_bytes'))}"
            if observed
            else "actual ?"
        )
        return (
            f"{actual} / req C{requested.get('cpu_cores', 0)} "
            f"R{_bytes(requested.get('ram_bytes'))} G{requested.get('gpu_count', 0)}"
        )


class WorkAttemptRenderer:
    """Render one human Work and its numbered operational Attempts."""

    @staticmethod
    def attempts(
        payload: Mapping[str, Any], work_index: int
    ) -> tuple[Mapping[str, Any] | None, list[Mapping[str, Any]]]:
        work = payload.get("work", {})
        work = work if isinstance(work, Mapping) else {}
        items = [item for item in work.get("items", ()) if isinstance(item, Mapping)]
        if not items:
            return None, []
        selected = items[min(work_index, len(items) - 1)]
        return selected, _work_attempt_jobs(payload, selected)

    @classmethod
    def render(
        cls,
        payload: Mapping[str, Any],
        work_index: int,
        *,
        selected_attempt: int,
        message: str,
        width: int,
        height: int,
    ) -> str:
        work, attempts = cls.attempts(payload, work_index)
        if work is None:
            return "LambdaForge Work\n\nNo Work is available.\n\n←/b/Esc/q back"
        progress = work.get("progress", {})
        progress = progress if isinstance(progress, Mapping) else {}
        completed, total = progress.get("completed"), progress.get("total")
        progress_text = (
            f"{completed if completed is not None else '?'}/"
            f"{total if total is not None else '?'} {progress.get('unit', 'units')}"
        )
        lines = [
            f"LambdaForge Work · {work.get('name', '-')} · {work.get('state', '-')}",
            (
                f"target: {work.get('cluster', '-')}  |  progress: {progress_text}  |  "
                f"revision: {work.get('scientific_revision') or 'legacy'}"
            ),
            "",
            "  ATTEMPT       STATE       TARGET             RUN       CREATED    USED / REQUESTED",
        ]
        selected_attempt = min(selected_attempt, max(0, len(attempts) - 1))
        capacity = max(1, height - len(lines) - 4)
        start = max(0, selected_attempt - capacity + 1)
        for offset, attempt in enumerate(attempts[start : start + capacity]):
            index = start + offset
            timing = attempt.get("timing", {})
            timing = timing if isinstance(timing, Mapping) else {}
            runtime = timing.get("runtime_seconds")
            runtime_text = (
                "waiting"
                if runtime is None
                and attempt.get("state") in {"preparing", "staging", "queued"}
                else _seconds(runtime if runtime is not None else timing.get("elapsed_seconds"))
            )
            lines.append(
                f"{'▶' if index == selected_attempt else ' '} "
                f"{str(attempt.get('attempt_label', f'Attempt {index + 1}')):<13.13} "
                f"{str(attempt.get('state', '-')):<11.11} "
                f"{str(attempt.get('cluster', work.get('cluster', '-'))):<18.18} "
                f"{runtime_text:<9.9} "
                f"{age(str(attempt.get('created_at_utc', ''))):<10.10} "
                f"{MonitorRenderer._job_usage(attempt)}"
            )
        lines.extend(
            (
                "",
                message
                or "↑/↓ attempts · Enter/→ full log · ←/b/Esc/q back · "
                "x cancel · d delete · D clear terminal history · r refresh",
            )
        )
        return _screen(lines, width=width, height=height)


class HistoryChart:
    """Render paired vertical time bars with an explicit scale and legend."""

    @staticmethod
    def render(
        label: str,
        total: tuple[float | None, ...],
        personal: tuple[float | None, ...],
        *,
        width: int,
        rows: int = 3,
    ) -> list[str]:
        columns = max(4, min(30, (width - 12) // 2))
        left = total[-columns:]
        mine = personal[-columns:]
        missing = columns - max(len(left), len(mine))
        left = (None,) * missing + left
        mine = (None,) * missing + mine
        current_total = next((value for value in reversed(left) if value is not None), None)
        current_mine = next((value for value in reversed(mine) if value is not None), None)
        lines = [
            f"{label:<12} cluster={HistoryChart._current(current_total)}  "
            f"mine={HistoryChart._current(current_mine)}  █ cluster  ▓ mine"
        ]
        for row in range(rows, 0, -1):
            threshold = 100.0 * row / rows
            bars = "".join(
                ("█" if value is not None and value >= threshold else " ")
                + ("▓" if own is not None and own >= threshold else " ")
                for value, own in zip(left, mine, strict=True)
            )
            lines.append(f"{int(threshold):>3}% ┤{bars}")
        lines.append(f"  0% └{'─' * (columns * 2)}→ now")
        return lines

    @staticmethod
    def _current(value: float | None) -> str:
        return f"{value:.0f}%" if value is not None else "unknown"


class ClusterDetailRenderer:
    """Render one cluster's history, personal allocation and only its jobs."""

    @classmethod
    def render(
        cls,
        payload: Mapping[str, Any],
        cluster_index: int,
        *,
        selected_job: int,
        history: ResourceHistory,
        message: str,
        width: int,
        height: int,
    ) -> str:
        clusters = [item for item in payload.get("clusters", ()) if isinstance(item, Mapping)]
        if not clusters:
            return "LambdaForge cluster detail\n\nNo clusters are configured.\n\nb/Esc/q back"
        cluster = clusters[min(cluster_index, len(clusters) - 1)]
        name = str(cluster.get("cluster", "-"))
        observed = cluster.get("observed", {})
        observed = observed if isinstance(observed, Mapping) else {}
        personal = cluster.get("personal", {})
        personal = personal if isinstance(personal, Mapping) else {}
        requested = personal.get("requested", {})
        requested = requested if isinstance(requested, Mapping) else {}
        mine = personal.get("observed", {})
        mine = mine if isinstance(mine, Mapping) else {}
        gpus = observed.get("gpus", ())
        gpus = gpus if isinstance(gpus, (list, tuple)) else ()
        lines = [
            f"LambdaForge cluster · {name} · {'online' if cluster.get('online') else 'offline'}",
            (
                f"capacity: {observed.get('cpu_total', '?')} CPU · "
                f"{_bytes(observed.get('ram_total_bytes'))} RAM · {len(gpus)} GPU  |  "
                f"mine requested: C{requested.get('cpu_cores', 0)} "
                f"R{_bytes(requested.get('ram_bytes'))} G{requested.get('gpu_count', 0)}"
            ),
            (
                f"mine observed: C{float(mine.get('cpu_percent', 0) or 0) / 100:.1f} cores · "
                f"R{_bytes(mine.get('rss_bytes'))} · VRAM {_bytes(mine.get('gpu_memory_bytes'))} "
                f"({mine.get('job_count', 0)} measured jobs)"
                if mine.get("job_count")
                else "mine observed: unavailable (requested allocations remain shown above)"
            ),
            "",
        ]
        lines.extend(
            HistoryChart.render(
                "CPU load",
                history.series(name, "cpu"),
                history.series(name, "my_cpu"),
                width=width,
            )
        )
        lines.extend(
            HistoryChart.render(
                "RAM used",
                history.series(name, "ram"),
                history.series(name, "my_ram"),
                width=width,
            )
        )
        lines.extend(
            HistoryChart.render(
                "GPU util/VRAM",
                history.series(name, "gpu"),
                history.series(name, "my_gpu"),
                width=width,
            )
        )
        all_jobs = payload.get("jobs", {})
        all_jobs = all_jobs if isinstance(all_jobs, Mapping) else {}
        jobs = [
            item
            for item in all_jobs.get("items", ())
            if isinstance(item, Mapping) and item.get("cluster") == name
        ]
        lines.extend(
            (
                "",
                f"ATTEMPTS ON {name} ({len(jobs)})",
                "  WORK / ATTEMPT                   STATE       RUN      AGE      LAST ACTIVITY",
            )
        )
        capacity = max(1, height - len(lines) - 4)
        selected_job = min(selected_job, max(0, len(jobs) - 1))
        start = max(0, selected_job - capacity + 1)
        for offset, job in enumerate(jobs[start : start + capacity]):
            index = start + offset
            timing = job.get("timing", {})
            timing = timing if isinstance(timing, Mapping) else {}
            usage = job.get("usage", {})
            usage = usage if isinstance(usage, Mapping) else {}
            activity = usage.get("observed_at_utc") or job.get("updated_at_utc") or "-"
            runtime = timing.get("runtime_seconds")
            lines.append(
                f"{'▶' if index == selected_job else ' '} "
                f"{_job_label(payload, job):<32.32} {str(job.get('state', '-')):<11.11} "
                f"{_seconds(runtime if runtime is not None else timing.get('elapsed_seconds')):<8} "
                f"{age(str(job.get('created_at_utc', ''))):<8} {str(activity)[11:19] or '-'}"
            )
        lines.extend(
            (
                "",
                message
                or "↑/↓ attempts · Enter/→ full log · ←/b/Esc/q back · "
                "x cancel · d delete · D clear terminal history · r refresh",
            )
        )
        return _screen(lines, width=width, height=height)

    @staticmethod
    def jobs(payload: Mapping[str, Any], cluster_index: int) -> list[Mapping[str, Any]]:
        clusters = [item for item in payload.get("clusters", ()) if isinstance(item, Mapping)]
        if not clusters:
            return []
        name = clusters[min(cluster_index, len(clusters) - 1)].get("cluster")
        jobs = payload.get("jobs", {})
        jobs = jobs if isinstance(jobs, Mapping) else {}
        return [
            item
            for item in jobs.get("items", ())
            if isinstance(item, Mapping) and item.get("cluster") == name
        ]


class LogViewerRenderer:
    """Render a complete, scrollable, automatically refreshed log document."""

    @staticmethod
    def render(
        job_id: str, text: str, *, scroll: int, message: str, width: int, height: int
    ) -> str:
        lines = text.splitlines()
        capacity = max(1, height - 4)
        start = min(max(0, len(lines) - capacity), max(0, scroll))
        end = min(len(lines), start + capacity)
        header = (
            f"LambdaForge log · {job_id} · lines {start + 1 if lines else 0}-{end} of {len(lines)}"
        )
        return "\n".join(
            [
                header[:width],
                "",
                *(line[:width] for line in lines[start:end]),
                "",
                (
                    message
                    or "live refresh · ↑/↓ line · PgUp/PgDn page · Home/End · ←/b/Esc/q back"
                )[:width],
            ]
        )


class _TerminalSession(AbstractContextManager["_TerminalSession"]):
    """Restore terminal flags and screen even when interrupted."""

    def __init__(self, stream: TextIO = sys.stdout) -> None:
        self.stream = stream
        self.fd = sys.stdin.fileno()
        self.previous: list[Any] | None = None
        self.termios: Any = None

    def __enter__(self) -> _TerminalSession:
        import termios
        import tty

        self.termios, self.previous = termios, termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.stream.write("\x1b[?1049h\x1b[?25l")
        self.stream.flush()
        return self

    def __exit__(self, *exc: object) -> None:
        if self.previous is not None:
            self.termios.tcsetattr(self.fd, self.termios.TCSADRAIN, self.previous)
        self.stream.write("\x1b[?25h\x1b[?1049l")
        self.stream.flush()

    def key(self, timeout: float) -> str | None:
        if not select.select([self.fd], [], [], timeout)[0]:
            return None
        value = os.read(self.fd, 8).decode(errors="ignore")
        if value == "\x1b":
            deadline = time.monotonic() + 0.03
            while len(value) < 8 and time.monotonic() < deadline:
                ready = select.select([self.fd], [], [], max(0, deadline - time.monotonic()))[0]
                if not ready:
                    break
                value += os.read(self.fd, 8 - len(value)).decode(errors="ignore")
        return value


def _collect_snapshot(overview: OverviewService, connection: Connection) -> None:
    try:
        connection.send((overview.snapshot(), None))
    except BaseException as error:
        connection.send((None, f"{error.__class__.__name__}: {error}"))
    finally:
        connection.close()


def _collect_logs(jobs: JobService, job_id: str, connection: Connection) -> None:
    try:
        connection.send((jobs.logs(job_id), None))
    except BaseException as error:
        connection.send((None, f"{error.__class__.__name__}: {error}"))
    finally:
        connection.close()


def _apply_history_action(
    works: WorkService,
    action: str,
    selector: str,
    connection: Connection,
) -> None:
    """Apply one confirmed destructive history operation outside the terminal loop."""
    try:
        if action == "delete-work":
            result = works.delete(selector, apply=True)
        elif action == "delete-job":
            result = works.delete_job(selector, apply=True)
        elif action == "clear-history":
            result = works.clear_history(apply=True)
        else:
            raise ValueError(f"Unknown history action: {action}")
        connection.send(({"action": action, "result": result}, None))
    except BaseException as error:
        connection.send((None, f"{error.__class__.__name__}: {error}"))
    finally:
        connection.close()


class BackgroundProcess:
    """Run one cancellable provider call outside the interactive process."""

    def __init__(self, target: Any, arguments: tuple[Any, ...], name: str) -> None:
        self.target, self.arguments, self.name = target, arguments, name
        self.process: Any = None
        self.connection: Connection | None = None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self.close()
        parent, child = multiprocessing.get_context("fork").Pipe(duplex=False)
        self.process = multiprocessing.get_context("fork").Process(
            target=self.target, args=(*self.arguments, child), daemon=True, name=self.name
        )
        self.process.start()
        child.close()
        self.connection = parent

    def take(self) -> tuple[Any | None, str | None] | None:
        if self.connection is None or not self.connection.poll():
            return None
        try:
            result = self.connection.recv()
        except EOFError:
            result = (None, "Background worker exited without returning data.")
        self._release()
        return result

    def close(self) -> None:
        if self.process is not None and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=0.2)
            if self.process.is_alive() and hasattr(self.process, "kill"):
                self.process.kill()
                self.process.join(timeout=0.2)
        self._release()

    def _release(self) -> None:
        if self.connection is not None:
            self.connection.close()
        if self.process is not None:
            self.process.join(timeout=0.1)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=0.1)
            self.process.close()
        self.connection = self.process = None


class SnapshotProcess(BackgroundProcess):
    """Collect a global snapshot without blocking terminal input."""

    def __init__(self, overview: OverviewService) -> None:
        super().__init__(_collect_snapshot, (overview,), "lambdaforge-top-snapshot")


class LogProcess(BackgroundProcess):
    """Load a complete remote log without blocking terminal input."""

    def __init__(self, jobs: JobService, job_id: str) -> None:
        super().__init__(_collect_logs, (jobs, job_id), "lambdaforge-top-logs")


class HistoryActionProcess(BackgroundProcess):
    """Apply confirmed history deletion without freezing keyboard handling."""

    def __init__(self, works: WorkService, action: str, selector: str = "") -> None:
        super().__init__(
            _apply_history_action,
            (works, action, selector),
            "lambdaforge-top-history-action",
        )


class LiveJobMonitor:
    """Coordinate background observations and explicit interactive actions."""

    def __init__(
        self,
        overview: OverviewService,
        jobs: JobService,
        *,
        interval: float = 2,
        history_seconds: float = 60,
        stream: TextIO = sys.stdout,
        works: WorkService | None = None,
    ) -> None:
        if interval < 0.2:
            raise ValueError("Live monitor interval must be at least 0.2 seconds.")
        self.overview, self.jobs, self.interval = overview, jobs, interval
        self.history_seconds, self.stream = history_seconds, stream
        self.works = works

    def run(self) -> int:
        selected = selected_cluster = detail_selected = log_scroll = 0
        focus, mode, log_job, log_label, log_text, message = (
            "jobs",
            "overview",
            "",
            "",
            "",
            "",
        )
        pending_cancel: str | None = None
        pending_history: tuple[str, str] | None = None
        return_mode = "overview"
        payload: Mapping[str, Any] = {}
        history, dirty = ResourceHistory(self.history_seconds), True
        poller = SnapshotProcess(self.overview)
        log_poller: LogProcess | None = None
        action_poller: HistoryActionProcess | None = None
        poller.start()
        next_refresh = time.monotonic() + self.interval
        next_log_refresh = float("inf")
        try:
            with _TerminalSession(self.stream) as terminal:
                while True:
                    completed = poller.take()
                    if completed:
                        updated, error = completed
                        if isinstance(updated, Mapping):
                            payload = updated
                            history.record(payload)
                            if (
                                mode != "logs"
                                and pending_cancel is None
                                and pending_history is None
                                and action_poller is None
                            ):
                                message = ""
                        elif mode != "logs":
                            message = f"Refresh failed: {error}"
                        next_refresh, dirty = time.monotonic() + self.interval, True
                    if log_poller is not None and (loaded := log_poller.take()):
                        value, error = loaded
                        new_text = str(value or "")
                        capacity = max(
                            1, shutil.get_terminal_size((120, 30)).lines - 4
                        )
                        old_maximum = max(0, len(log_text.splitlines()) - capacity)
                        new_maximum = max(0, len(new_text.splitlines()) - capacity)
                        following = not log_text or log_scroll >= old_maximum
                        log_text = new_text
                        log_scroll = (
                            new_maximum if following else min(log_scroll, new_maximum)
                        )
                        message, log_poller, dirty = (
                            (f"Log load failed: {error}" if error else ""),
                            None,
                            True,
                        )
                        next_log_refresh = time.monotonic() + self.interval
                    if action_poller is not None and (applied := action_poller.take()):
                        value, error = applied
                        if error:
                            message = f"History action failed: {error}"
                        else:
                            result = value.get("result", {}) if isinstance(value, Mapping) else {}
                            action = value.get("action") if isinstance(value, Mapping) else None
                            if action == "clear-history":
                                deleted = len(result.get("deleted_jobs", ()))
                                active = len(result.get("active_jobs_preserved", ()))
                                failed = len(result.get("failures", ()))
                                message = (
                                    f"History cleared: {deleted} terminal Jobs removed; "
                                    f"{active} active preserved; {failed} failed."
                                )
                            else:
                                message = "Selected history and owned workspace removed."
                        action_poller = None
                        if not poller.running:
                            poller.start()
                        dirty = True
                    if time.monotonic() >= next_refresh and not poller.running:
                        poller.start()
                        next_refresh = time.monotonic() + self.interval
                    if (
                        mode == "logs"
                        and log_job
                        and log_poller is None
                        and time.monotonic() >= next_log_refresh
                    ):
                        log_poller = LogProcess(self.jobs, log_job)
                        log_poller.start()
                        next_log_refresh = time.monotonic() + self.interval
                    raw_items = payload.get("jobs", {}).get("items", [])
                    work_items = payload.get("work", {}).get("items", [])
                    items = work_items if work_items else raw_items
                    clusters = payload.get("clusters", [])
                    if focus == "jobs" and not items and clusters:
                        focus = "clusters"
                    elif focus == "clusters" and not clusters and items:
                        focus = "jobs"
                    selected = min(selected, max(0, len(items) - 1))
                    selected_cluster = min(selected_cluster, max(0, len(clusters) - 1))
                    detail_items = ClusterDetailRenderer.jobs(payload, selected_cluster)
                    detail_selected = min(detail_selected, max(0, len(detail_items) - 1))
                    _, attempt_items = WorkAttemptRenderer.attempts(payload, selected)
                    if mode == "work":
                        detail_selected = min(
                            detail_selected, max(0, len(attempt_items) - 1)
                        )
                    size = shutil.get_terminal_size((120, 30))
                    if dirty:
                        if mode == "logs":
                            rendered = LogViewerRenderer.render(
                                log_label or log_job,
                                log_text,
                                scroll=log_scroll,
                                message=message or ("Loading complete log…" if log_poller else ""),
                                width=size.columns,
                                height=size.lines,
                            )
                        elif mode == "cluster":
                            rendered = ClusterDetailRenderer.render(
                                payload,
                                selected_cluster,
                                selected_job=detail_selected,
                                history=history,
                                message=message,
                                width=size.columns,
                                height=size.lines,
                            )
                        elif mode == "work":
                            rendered = WorkAttemptRenderer.render(
                                payload,
                                selected,
                                selected_attempt=detail_selected,
                                message=message,
                                width=size.columns,
                                height=size.lines,
                            )
                        else:
                            rendered = MonitorRenderer.render(
                                payload,
                                selected=selected,
                                selected_cluster=selected_cluster,
                                focus=focus,
                                history=history,
                                message=message or ("Loading providers…" if not payload else ""),
                                width=size.columns,
                                height=size.lines,
                                view="research",
                            )
                        self.stream.write("\x1b[H\x1b[2J" + rendered)
                        self.stream.flush()
                        dirty = False
                    key = terminal.key(0.05)
                    if mode == "logs":
                        page, maximum = (
                            max(1, size.lines - 5),
                            max(0, len(log_text.splitlines()) - max(1, size.lines - 4)),
                        )
                        if key in {"q", "Q", "b", "B", "\x1b", "\x1b[D"}:
                            if log_poller:
                                log_poller.close()
                                log_poller = None
                            next_log_refresh = float("inf")
                            mode, message, dirty = return_mode, "", True
                        elif key in {"j", "\x1b[B"}:
                            log_scroll, dirty = min(maximum, log_scroll + 1), True
                        elif key in {"k", "\x1b[A"}:
                            log_scroll, dirty = max(0, log_scroll - 1), True
                        elif key == "\x1b[6~":
                            log_scroll, dirty = min(maximum, log_scroll + page), True
                        elif key == "\x1b[5~":
                            log_scroll, dirty = max(0, log_scroll - page), True
                        elif key in {"g", "\x1b[H", "\x1b[1~"}:
                            log_scroll, dirty = 0, True
                        elif key in {"G", "\x1b[F", "\x1b[4~"}:
                            log_scroll, dirty = maximum, True
                        continue
                    if pending_cancel is not None:
                        if key in {"x", "X", "y", "Y", "\r", "\n"}:
                            try:
                                record = self.jobs.cancel(pending_cancel)
                                message = f"{record.job_id}: {record.state.value}"
                                if not poller.running:
                                    poller.start()
                            except Exception as error:
                                message = (
                                    f"Cancellation failed: {error.__class__.__name__}: {error}"
                                )
                            pending_cancel, dirty = None, True
                        elif key in {"n", "N", "\x1b"}:
                            pending_cancel, message, dirty = None, "Cancellation aborted.", True
                        elif key is not None:
                            message = (
                                f"Cancel {pending_cancel}? Press x again, y or Enter to confirm; "
                                "n/Esc keeps it running."
                            )
                            dirty = True
                        continue
                    if pending_history is not None:
                        action, selector = pending_history
                        confirmation_key = "D" if action == "clear-history" else "d"
                        if key in {confirmation_key, "y", "Y", "\r", "\n"}:
                            action_poller = HistoryActionProcess(
                                self._work_service(), action, selector
                            )
                            action_poller.start()
                            message = (
                                "Clearing terminal history in the background…"
                                if action == "clear-history"
                                else "Deleting selected history in the background…"
                            )
                            pending_history, dirty = None, True
                        elif key in {"n", "N", "\x1b"}:
                            pending_history, message, dirty = (
                                None,
                                "History deletion cancelled.",
                                True,
                            )
                        elif key is not None:
                            message = self._history_confirmation(action, selector)
                            dirty = True
                        continue
                    if action_poller is not None:
                        if key is not None:
                            message = "The confirmed history operation is still running…"
                            dirty = True
                        continue
                    if mode == "work":
                        if key in {"q", "Q", "b", "B", "\x1b", "\x1b[D"}:
                            mode, message, dirty = "overview", "", True
                        elif key in {"j", "\x1b[B"}:
                            detail_selected = min(
                                max(0, len(attempt_items) - 1), detail_selected + 1
                            )
                            dirty = True
                        elif key in {"k", "\x1b[A"}:
                            detail_selected = max(0, detail_selected - 1)
                            dirty = True
                        elif key == "r":
                            if not poller.running:
                                poller.start()
                            message = "Refreshing Work attempts in the background…"
                            dirty = True
                        elif key in {"\r", "\n", "\x1b[C"} and attempt_items:
                            attempt = attempt_items[detail_selected]
                            log_job = str(attempt.get("job_id", ""))
                            work = items[selected]
                            log_label = (
                                f"{work.get('name', 'Work')} · "
                                f"{attempt.get('attempt_label', f'Attempt {detail_selected + 1}')}"
                            )
                            log_text, log_scroll, message = "", 0, ""
                            return_mode, mode = "work", "logs"
                            log_poller = LogProcess(self.jobs, log_job)
                            log_poller.start()
                            next_log_refresh = time.monotonic() + self.interval
                            dirty = True
                        elif key in {"x", "X"} and attempt_items:
                            pending_cancel = str(attempt_items[detail_selected].get("job_id", ""))
                            message = (
                                f"Cancel Attempt {detail_selected + 1}? Press x again, y or Enter "
                                "to confirm; n/Esc keeps it running."
                            )
                            dirty = True
                        elif key == "d" and attempt_items and action_poller is None:
                            selector = str(attempt_items[detail_selected].get("job_id", ""))
                            pending_history = ("delete-job", selector)
                            message, dirty = self._history_confirmation(*pending_history), True
                        elif key == "D" and action_poller is None:
                            pending_history = ("clear-history", "")
                            message, dirty = self._history_confirmation(*pending_history), True
                        continue
                    if mode == "cluster":
                        if key in {"q", "Q", "b", "B", "\x1b", "\x1b[D"}:
                            mode, message, dirty = "overview", "", True
                        elif key in {"j", "\x1b[B"}:
                            detail_selected = min(
                                max(0, len(detail_items) - 1), detail_selected + 1
                            )
                            dirty = True
                        elif key in {"k", "\x1b[A"}:
                            detail_selected = max(0, detail_selected - 1)
                            dirty = True
                        elif key == "r":
                            if not poller.running:
                                poller.start()
                            message = "Refreshing cluster data in the background…"
                            dirty = True
                        elif key in {"\r", "\n", "\x1b[C"} and detail_items:
                            attempt = detail_items[detail_selected]
                            log_job = str(attempt["job_id"])
                            log_label = _job_label(payload, attempt)
                            log_text, log_scroll, message = "", 0, ""
                            return_mode, mode = "cluster", "logs"
                            log_poller = LogProcess(self.jobs, log_job)
                            log_poller.start()
                            next_log_refresh = time.monotonic() + self.interval
                            dirty = True
                        elif key in {"x", "X"} and detail_items:
                            pending_cancel = str(detail_items[detail_selected]["job_id"])
                            message = (
                                f"Cancel {pending_cancel}? Press x again, y or Enter to confirm; "
                                "n/Esc keeps it running."
                            )
                            dirty = True
                        elif key == "d" and detail_items and action_poller is None:
                            selector = str(detail_items[detail_selected]["job_id"])
                            pending_history = ("delete-job", selector)
                            message, dirty = self._history_confirmation(*pending_history), True
                        elif key == "D" and action_poller is None:
                            pending_history = ("clear-history", "")
                            message, dirty = self._history_confirmation(*pending_history), True
                        continue
                    if key in {"q", "Q"}:
                        return 0
                    if key == "\t":
                        if focus == "jobs" and clusters:
                            focus, dirty = "clusters", True
                        elif focus == "clusters" and items:
                            focus, dirty = "jobs", True
                    elif key in {"\r", "\n", "\x1b[C"}:
                        if focus == "clusters" and clusters:
                            detail_selected = 0
                            mode, message, dirty = "cluster", "", True
                        elif items and work_items:
                            detail_selected = 0
                            mode, message, dirty = "work", "", True
                        elif items:
                            return_mode = "overview"
                            log_job = _selected_job_id(items[selected])
                            log_label = _job_label(payload, items[selected])
                            log_text, log_scroll, message, mode = "", 0, "", "logs"
                            log_poller = LogProcess(self.jobs, log_job)
                            log_poller.start()
                            next_log_refresh = time.monotonic() + self.interval
                            dirty = True
                    elif key in {"j", "\x1b[B"}:
                        focus, selected, selected_cluster = _move_overview_selection(
                            focus,
                            selected,
                            selected_cluster,
                            direction=1,
                            job_count=len(items),
                            cluster_count=len(clusters),
                        )
                        dirty = True
                    elif key in {"k", "\x1b[A"}:
                        focus, selected, selected_cluster = _move_overview_selection(
                            focus,
                            selected,
                            selected_cluster,
                            direction=-1,
                            job_count=len(items),
                            cluster_count=len(clusters),
                        )
                        dirty = True
                    elif key == "r":
                        if not poller.running:
                            poller.start()
                        message, dirty = "Refreshing providers in the background…", True
                    elif key in {"x", "X"} and items:
                        pending_cancel = _selected_job_id(items[selected])
                        message = (
                            f"Cancel {pending_cancel}? Press x again, y or Enter to confirm; "
                            "n/Esc keeps it running."
                        )
                        dirty = True
                    elif key == "d" and focus == "jobs" and items and action_poller is None:
                        item = items[selected]
                        if work_items:
                            pending_history = ("delete-work", str(item.get("work_id", "")))
                        else:
                            pending_history = ("delete-job", _selected_job_id(item))
                        message, dirty = self._history_confirmation(*pending_history), True
                    elif key == "D" and action_poller is None:
                        pending_history = ("clear-history", "")
                        message, dirty = self._history_confirmation(*pending_history), True
        finally:
            poller.close()
            if log_poller is not None:
                log_poller.close()
            if action_poller is not None:
                action_poller.close()

    def _work_service(self) -> WorkService:
        if self.works is None:
            catalog = getattr(self.jobs, "catalog", None)
            self.works = WorkService(catalog, jobs=self.jobs)
        return self.works

    @staticmethod
    def _history_confirmation(action: str, selector: str) -> str:
        if action == "clear-history":
            return (
                "Clear every terminal Work/Job history entry and owned workspace? "
                "Press D again, y or Enter; n/Esc preserves it. Active Jobs are never removed."
            )
        label = "Work" if action == "delete-work" else "Job"
        return (
            f"Permanently delete terminal {label} {selector} and its owned workspace? "
            "Press d again, y or Enter; n/Esc preserves it."
        )
