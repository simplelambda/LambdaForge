"""Responsive, dependency-light interactive control-plane monitor."""

from __future__ import annotations

import multiprocessing
import os
import select
import shutil
import sys
import textwrap
import time
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from multiprocessing.connection import Connection
from typing import Any, TextIO

from lambdaforge.cli.common import age
from lambdaforge.cli.TerminalTheme import TerminalTheme
from lambdaforge.cli.TerminalTimeSeriesChart import TerminalTimeSeriesChart
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.OverviewService import OverviewService
from lambdaforge.controlplane.WorkService import WorkService
from lambdaforge.hpo.StudyInsights import StudyInsightAnalyzer


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


def _gpu_memory(observed: Mapping[str, Any]) -> float | None:
    values = observed.get("gpus", ())
    if not isinstance(values, (list, tuple)):
        return None
    used = sum(
        float(item.get("memory_used_bytes", 0) or 0) for item in values if isinstance(item, Mapping)
    )
    total = sum(
        float(item.get("memory_total_bytes", 0) or 0)
        for item in values
        if isinstance(item, Mapping)
    )
    return 100 * used / total if total else None


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


def _horizontal_view(text: str, *, offset: int, width: int) -> str:
    """Return one deterministic terminal-width slice of a potentially long line."""
    return text[max(0, offset) : max(0, offset) + max(1, width)]


def _side_by_side(charts: Sequence[Sequence[str]], *, width: int, row_gap: int = 0) -> list[str]:
    """Compose bounded charts in two columns when the terminal is wide enough."""
    if not charts:
        return []
    columns = 2 if width >= 100 else 1
    gap = "  "
    cell_width = max(24, (width - len(gap) * (columns - 1)) // columns)
    output: list[str] = []
    for start in range(0, len(charts), columns):
        group = charts[start : start + columns]
        group_height = max(len(chart) for chart in group)
        for row in range(group_height):
            cells = [
                (chart[row] if row < len(chart) else "").ljust(cell_width)[:cell_width]
                for chart in group
            ]
            output.append(gap.join(cells).rstrip())
        if row_gap and start + columns < len(charts):
            output.extend("" for _ in range(row_gap))
    return output


def _display_value(value: Any, *, width: int = 30) -> str:
    """Format one user parameter without collapsing the whole mapping into JSON."""
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif value is None:
        text = "null"
    elif isinstance(value, str):
        text = value
    elif isinstance(value, int | float):
        text = f"{value:.6g}" if isinstance(value, float) else str(value)
    else:
        text = json_compact(value, width=max(8, width))
    return text if len(text) <= width else text[: max(1, width - 1)] + "…"


def _key_value_lines(
    values: Mapping[str, Any],
    *,
    width: int,
    heading: str,
    maximum_columns: int = 3,
) -> list[str]:
    """Render a complete selected mapping as aligned, scan-friendly cells."""
    items = sorted((str(name), value) for name, value in values.items())
    lines = [f"{heading} ({len(items)})"]
    if not items:
        return [*lines, "  none"]
    columns = min(maximum_columns, max(1, width // 42), len(items))
    gap = "  "
    cell_width = max(20, (width - len(gap) * (columns - 1)) // columns)
    cells = [
        f"{name} = {_display_value(value, width=max(8, cell_width - len(name) - 5))}"
        for name, value in items
    ]
    for start in range(0, len(cells), columns):
        row = cells[start : start + columns]
        lines.append(
            gap.join(f"  {cell:<{cell_width - 2}.{cell_width - 2}}" for cell in row).rstrip()
        )
    return lines


def _bounded_panel(lines: Sequence[str], *, maximum: int, continuation: str) -> list[str]:
    """Keep navigation tables usable while linking to the complete drill-down panel."""
    if len(lines) <= maximum:
        return list(lines)
    hidden = len(lines) - max(1, maximum - 1)
    return [*lines[: max(1, maximum - 1)], f"  … {hidden} more {continuation}"]


def _elapsed_seconds(detail: Mapping[str, Any]) -> float | None:
    """Return terminal duration, or live elapsed time for an active Run."""
    duration = detail.get("duration_seconds")
    if isinstance(duration, int | float):
        return max(0.0, float(duration))
    started = detail.get("started_at_utc")
    if not isinstance(started, str) or not started:
        return None
    finished = detail.get("finished_at_utc")
    try:
        start_time = datetime.fromisoformat(started.replace("Z", "+00:00"))
        end_time = (
            datetime.fromisoformat(finished.replace("Z", "+00:00"))
            if isinstance(finished, str) and finished
            else datetime.now(timezone.utc)
        )
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
        return max(0.0, (end_time - start_time).total_seconds())
    except (TypeError, ValueError):
        return None


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
                "gpu_memory": _gpu_memory(observed),
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


def _is_study_work(item: Mapping[str, Any]) -> bool:
    """Recognize a declared parameter study before its live telemetry exists."""
    return item.get("study_expected") is True


def _should_open_study(item: Mapping[str, Any]) -> bool:
    """Use the study dashboard unless a terminal Work has no study evidence."""
    if not _is_study_work(item):
        return False
    if isinstance(item.get("study"), Mapping):
        return True
    return str(item.get("state", "")) not in {
        "planned",
        "succeeded",
        "failed",
        "cancelled",
        "timeout",
    }


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
            (
                "  CLUSTER           STATUS   CPU                 RAM                 "
                "GPU                 GPUS JOBS"
                if width >= 118
                else "  CLUSTER           STATUS   CPU    RAM    GPU    GPUS JOBS"
            ),
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
            status = "online" if cluster.get("online") else "offline"
            if width >= 118:
                lines.append(
                    f"{marker} {name:<17.17} {status:<7} "
                    f"{cls._metric(observed.get('cpu_load'), history, name, 'cpu'):<19.19} "
                    f"{cls._metric(_ram(observed), history, name, 'ram'):<19.19} "
                    f"{cls._metric(_gpu(observed), history, name, 'gpu'):<19.19} "
                    f"{len(gpus):<4} {str(personal.get('active_jobs', 0)):<4}"
                )
            else:
                lines.append(
                    f"{marker} {name:<17.17} {status:<7} "
                    f"{cls._number(observed.get('cpu_load')):<6} "
                    f"{cls._number(_ram(observed)):<6} {cls._number(_gpu(observed)):<6} "
                    f"{len(gpus):<4} {str(personal.get('active_jobs', 0)):<4}"
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
                if runtime is None and attempt.get("state") in {"preparing", "staging", "queued"}
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


class StudyRenderer:
    """Render adaptive/exhaustive candidates and their isolated seed Runs."""

    @staticmethod
    def study(
        payload: Mapping[str, Any], work_index: int
    ) -> tuple[Mapping[str, Any] | None, list[Mapping[str, Any]]]:
        work = payload.get("work", {})
        work = work if isinstance(work, Mapping) else {}
        items = [item for item in work.get("items", ()) if isinstance(item, Mapping)]
        if not items:
            return None, []
        selected = items[min(work_index, len(items) - 1)]
        study = selected.get("study")
        study = study if isinstance(study, Mapping) else None
        candidates = (
            [value for value in study.get("candidates", ()) if isinstance(value, Mapping)]
            if study is not None
            else []
        )
        return study, candidates

    @classmethod
    def render_candidates(
        cls,
        payload: Mapping[str, Any],
        work_index: int,
        *,
        selected_candidate: int,
        message: str,
        width: int,
        height: int,
    ) -> str:
        study, candidates = cls.study(payload, work_index)
        if study is None:
            return (
                "LambdaForge study\n\n"
                "Run telemetry is not available yet. The Work may still be preparing, or it "
                "may have been launched with an older worker runtime.\n\n"
                "The view refreshes automatically · a scheduler Attempts · ←/b/Esc/q back"
            )
        counts = study.get("counts", {})
        counts = counts if isinstance(counts, Mapping) else {}
        objective = study.get("objective", {})
        objective = objective if isinstance(objective, Mapping) else {}
        composite = isinstance(objective.get("metrics"), Mapping)
        metric = "composite utility" if composite else str(objective.get("metric", ""))
        mode = str(objective.get("mode", "-"))
        direction = "↑" if mode == "max" else "↓" if mode == "min" else ""
        raw_analysis = study.get("hpo_analysis")
        analysis = raw_analysis if isinstance(raw_analysis, Mapping) else {}
        raw_question = analysis.get("next_question")
        question = raw_question if isinstance(raw_question, Mapping) else {}
        lines = [
            f"LambdaForge study · {study.get('name', '-')} · {study.get('strategy', '-')}",
            (
                f"proposed={counts.get('candidates', len(candidates))}/"
                f"{study.get('planned_candidates', counts.get('candidates', len(candidates)))}  "
                f"scheduled={counts.get('scheduled_runs', 0)}  "
                f"active={counts.get('active_runs', 0)}  "
                f"completed={counts.get('completed_runs', 0)}  "
                f"pruned={counts.get('pruned_runs', 0)}  failed={counts.get('failed_runs', 0)}"
            ),
            (
                f"objective: {metric or '-'} ({mode})  |  "
                + (
                    f"{objective.get('aggregation', 'weighted_mean')} over "
                    f"{', '.join(str(name) for name in objective.get('metrics', {}))}  |  "
                    if composite
                    else ""
                )
                + "each row is one hyperparameter combination"
            ),
            *(
                [
                    f"HPO insight: {analysis.get('status', 'preparing')} · next question: "
                    f"{question.get('parameter', '-')} · press i for evidence and confidence"
                ]
                if str(study.get("strategy", "")) == "adaptive"
                else []
            ),
            "",
            (
                "  CANDIDATE   STATE       RUNS     BEST OBJECTIVE  "
                "BEST SEED@EPOCH       CURRENT OBJECTIVE"
            ),
        ]
        selected_candidate = min(selected_candidate, max(0, len(candidates) - 1))
        selected_parameters = (
            candidates[selected_candidate].get("parameters", {}) if candidates else {}
        )
        selected_parameters = (
            selected_parameters if isinstance(selected_parameters, Mapping) else {}
        )
        parameter_lines = _key_value_lines(
            selected_parameters,
            width=width,
            heading="SELECTED TRIAL PARAMETERS",
        )
        parameter_lines = _bounded_panel(
            parameter_lines,
            maximum=max(3, min(6, height // 4)),
            continuation="parameter row(s); Enter opens the Trial.",
        )
        if composite and candidates:
            selected_value = candidates[selected_candidate]
            selected_utility = selected_value.get("selection_objective")
            selected_error = selected_value.get("selection_standard_error")
            parameter_lines.extend(
                (
                    "SELECTED TRIAL UTILITY",
                    (
                        f"  mean={_display_value(selected_utility)}  "
                        f"standard error={_display_value(selected_error)}  "
                        f"aggregation={objective.get('aggregation', 'geometric')}  "
                        f"best step={selected_value.get('best_step', '-')}  "
                        f"Pareto={'yes' if selected_value.get('pareto_optimal') else 'no'}"
                    ),
                )
            )
            raw_components = selected_value.get("objective_components", {})
            components = raw_components if isinstance(raw_components, Mapping) else {}
            if components:
                parameter_lines.extend(
                    (
                        "COMPONENTS",
                        "  METRIC                    RAW       NORMALIZED  WEIGHT   CONTRIBUTION",
                    )
                )
                for name, raw_detail in components.items():
                    detail = raw_detail if isinstance(raw_detail, Mapping) else {}
                    parameter_lines.append(
                        f"  {str(name):<25.25} "
                        f"{_display_value(detail.get('raw')):<9.9} "
                        f"{_display_value(detail.get('normalized')):<11.11} "
                        f"{_display_value(detail.get('weight')):<8.8} "
                        f"{_display_value(detail.get('contribution')):<12.12}"
                    )
            feasibility = selected_value.get("feasibility", {})
            feasibility = feasibility if isinstance(feasibility, Mapping) else {}
            constraints = feasibility.get("constraints", {})
            constraints = constraints if isinstance(constraints, Mapping) else {}
            if constraints:
                parameter_lines.append(
                    "CONSTRAINTS · "
                    + "; ".join(
                        f"{name}={_display_value(detail.get('mean'))} "
                        f"({'ok' if detail.get('satisfied') else 'not satisfied'})"
                        for name, raw_detail in constraints.items()
                        for detail in [raw_detail if isinstance(raw_detail, Mapping) else {}]
                    )
                )
        capacity = max(1, height - len(lines) - len(parameter_lines) - 4)
        start = max(0, selected_candidate - capacity + 1)
        for offset, candidate in enumerate(candidates[start : start + capacity]):
            index = start + offset
            runs = [value for value in candidate.get("runs", ()) if isinstance(value, Mapping)]
            terminal = sum(
                value.get("state") in {"succeeded", "failed", "pruned"} for value in runs
            )
            best = candidate.get("best_objective")
            current = candidate.get("current_objective")
            best_text = f"{float(best):.5g}" if isinstance(best, int | float) else "-"
            current_text = f"{float(current):.5g}" if isinstance(current, int | float) else "-"
            best_seed = candidate.get("best_seed")
            best_step = candidate.get("best_step")
            best_location = (
                f"seed {best_seed if best_seed is not None else 'none'} @ {best_step}"
                if best_step is not None
                else "-"
            )
            lines.append(
                f"{'▶' if index == selected_candidate else ' '} "
                f"{'◆' if candidate.get('pareto_optimal') else ' '}"
                f"Trial {int(candidate.get('trial', index + 1)):<4} "
                f"{str(candidate.get('state', 'pending')):<11.11} "
                f"{terminal:>2}/{len(runs):<4} {best_text:<15.15} "
                f"{best_location:<21.21} {current_text:<17.17}"
            )
        if metric:
            lines.append(
                f"  HPO uses the mean of each completed seed's best {metric} {direction}; "
                "current is the latest mean."
            )
        if composite:
            lines.append("  ◆ marks the current non-dominated Pareto diagnostic in raw metrics.")
        lines.extend(("", *parameter_lines))
        lines.extend(
            (
                "",
                message
                or (
                    "↑/↓ candidates · Enter/→ seed Runs · i HPO insights · "
                    "a scheduler Attempts · ←/b/Esc/q back · x cancel"
                    if str(study.get("strategy", "")) == "adaptive"
                    else "↑/↓ candidates · Enter/→ seed Runs · a scheduler Attempts · "
                    "←/b/Esc/q back · x cancel"
                ),
            )
        )
        return _screen(lines, width=width, height=height)

    @classmethod
    def candidate_runs(
        cls, payload: Mapping[str, Any], work_index: int, candidate_index: int
    ) -> tuple[Mapping[str, Any] | None, list[Mapping[str, Any]]]:
        _study, candidates = cls.study(payload, work_index)
        if not candidates:
            return None, []
        candidate = candidates[min(candidate_index, len(candidates) - 1)]
        return candidate, [
            value for value in candidate.get("runs", ()) if isinstance(value, Mapping)
        ]

    @classmethod
    def render_runs(
        cls,
        payload: Mapping[str, Any],
        work_index: int,
        candidate_index: int,
        *,
        selected_run: int,
        message: str,
        width: int,
        height: int,
    ) -> str:
        candidate, runs = cls.candidate_runs(payload, work_index, candidate_index)
        if candidate is None:
            return "LambdaForge candidate\n\nNo candidate is available.\n\n←/b/Esc/q back"
        study, _candidates = cls.study(payload, work_index)
        objective = study.get("objective", {}) if isinstance(study, Mapping) else {}
        objective = objective if isinstance(objective, Mapping) else {}
        objective_metric = str(objective.get("metric", ""))
        parameters = candidate.get("parameters", {})
        parameters = parameters if isinstance(parameters, Mapping) else {}
        parameter_lines = _key_value_lines(
            parameters,
            width=width,
            heading="TRIAL PARAMETERS",
        )
        parameter_lines = _bounded_panel(
            parameter_lines,
            maximum=max(3, min(5, height // 5)),
            continuation="parameter row(s); see the previous Trial panel for all values.",
        )
        selected_run = min(selected_run, max(0, len(runs) - 1))
        selected_metrics = runs[selected_run].get("latest_metrics", {}) if runs else {}
        selected_metrics = selected_metrics if isinstance(selected_metrics, Mapping) else {}
        metric_lines = _key_value_lines(
            selected_metrics,
            width=width,
            heading="SELECTED SEED/RUN METRICS",
        )
        metric_lines = _bounded_panel(
            metric_lines,
            maximum=max(3, min(6, height // 4)),
            continuation="metric row(s); Enter opens the complete Run dashboard.",
        )
        lines = [
            f"LambdaForge candidate · Trial {candidate.get('trial', '-')} · "
            f"{candidate.get('state', '-')}",
            *parameter_lines,
            "",
            (
                "  RUN / SEED           GPU        STATE       LAST EPOCH  BEST EPOCH  "
                "BEST OBJECTIVE  CURRENT OBJECTIVE  TIME"
            ),
        ]
        capacity = max(3, height - len(lines) - len(metric_lines) - 5)
        start = max(0, selected_run - capacity + 1)
        for offset, run in enumerate(runs[start : start + capacity]):
            index = start + offset
            epoch = cls._latest_step(run)
            best_epoch = cls._best_step(run)
            best_objective = run.get("best_objective")
            latest_metrics = run.get("latest_metrics", {})
            latest_metrics = latest_metrics if isinstance(latest_metrics, Mapping) else {}
            current_objective = latest_metrics.get(objective_metric)
            best_text = (
                f"{float(best_objective):.5g}" if isinstance(best_objective, int | float) else "-"
            )
            current_text = (
                f"{float(current_objective):.5g}"
                if isinstance(current_objective, int | float)
                else "-"
            )
            seed = run.get("seed")
            lines.append(
                f"{'▶' if index == selected_run else ' '} "
                f"Seed {str(seed if seed is not None else 'none'):<14.14} "
                f"{cls._gpu_label(run):<10.10} "
                f"{str(run.get('state', 'scheduled')):<11.11} "
                f"{str(epoch if epoch is not None else '-'):<11.11} "
                f"{str(best_epoch if best_epoch is not None else '-'):<11.11} "
                f"{best_text:<15.15} "
                f"{current_text:<17.17} "
                f"{_seconds(_elapsed_seconds(run)):<10.10}"
            )
        if objective_metric:
            lines.append(
                f"  Best epoch is selected by objective {objective_metric} "
                f"({objective.get('mode', '-')}); pruned means terminal early stop, not pause."
            )
        if runs and runs[selected_run].get("prune_reason"):
            termination = runs[selected_run].get("termination", {})
            termination = termination if isinstance(termination, Mapping) else {}
            lines.extend(
                (
                    "PERFORMANCE PRUNED · candidate-level censored evidence",
                    (
                        f"  candidate={termination.get('candidate', candidate.get('trial', '-'))}  "
                        f"seed={runs[selected_run].get('seed', '-')}  "
                        f"step={termination.get('common_step', '-')}  "
                        f"fidelity={_display_value(termination.get('fidelity_targets'))}"
                    ),
                    (
                        f"  utility current={_display_value(termination.get('current_utility'))}  "
                        f"predicted={_display_value(termination.get('predicted_utility'))} ± "
                        f"{_display_value(termination.get('predicted_standard_error'))}  "
                        f"reference Trial={termination.get('reference_candidate', '-')}"
                    ),
                    (
                        f"  P(competitive)="
                        f"{_display_value(termination.get('probability_competitive'))}  "
                        f"threshold={_display_value(termination.get('threshold'))}  "
                        f"margin={_display_value(termination.get('equivalence_margin'))}  "
                        f"confirmations={termination.get('confirmations', '-')}/"
                        f"{termination.get('required_confirmations', '-')}"
                    ),
                    (
                        f"  model={termination.get('curve_model', '-')}  "
                        f"method={termination.get('comparison_method', '-')}  "
                        f"reason={runs[selected_run]['prune_reason']}"
                    ),
                )
            )
        if runs and runs[selected_run].get("termination_type") == "scheduler_preempted":
            termination = runs[selected_run].get("termination", {})
            termination = termination if isinstance(termination, Mapping) else {}
            lines.extend(
                (
                    "SCHEDULER PAUSED · neutral evidence, resumable from owned checkpoint",
                    (
                        f"  step={termination.get('observed_step', '-')}  "
                        f"old={termination.get('old_action', 'CONTINUE')} "
                        f"({_display_value(termination.get('old_priority'))})  →  "
                        f"new={termination.get('new_action', '-')} "
                        f"({_display_value(termination.get('new_priority'))})"
                    ),
                    (
                        f"  hysteresis={_display_value(termination.get('hysteresis_ratio'))}  "
                        f"runtime={_seconds(termination.get('elapsed_seconds'))}  "
                        f"reason={termination.get('reason', '-')}"
                    ),
                )
            )
        lines.extend(("", *metric_lines))
        lines.extend(
            (
                "",
                message
                or "↑/↓ Runs · Enter/→ curves, statistics and isolated log · "
                "←/b/Esc/q candidates · x cancel",
            )
        )
        return _screen(lines, width=width, height=height)

    @staticmethod
    def _latest_step(run: Mapping[str, Any]) -> int | None:
        value = run.get("latest_step")
        return int(value) if isinstance(value, int) else None

    @staticmethod
    def _best_step(run: Mapping[str, Any]) -> int | None:
        value = run.get("best_step")
        return int(value) if isinstance(value, int) else None

    @staticmethod
    def _gpu_label(run: Mapping[str, Any]) -> str:
        token = run.get("gpu_token")
        index = run.get("gpu_index")
        if isinstance(token, str) and token.isdigit():
            return f"GPU {token}"
        if isinstance(index, int):
            return f"GPU {index}"
        return "-"


class StudyInsightRenderer:
    """Render interpretable, explicitly non-causal adaptive-search diagnostics."""

    @staticmethod
    def analysis(
        payload: Mapping[str, Any], work_index: int
    ) -> tuple[Mapping[str, Any] | None, list[Mapping[str, Any]]]:
        study, candidates = StudyRenderer.study(payload, work_index)
        raw = study.get("hpo_analysis") if isinstance(study, Mapping) else None
        analysis = raw if isinstance(raw, Mapping) else None
        objective = study.get("objective") if isinstance(study, Mapping) else None
        parameters = (
            [value for value in analysis.get("parameters", ()) if isinstance(value, Mapping)]
            if analysis is not None
            else []
        )
        current_shape = all(
            "response" in value and "joint_relationships" in value and "pruning_signal" in value
            for value in parameters
        )
        if (
            candidates
            and isinstance(objective, Mapping)
            and (
                analysis is None
                or int(analysis.get("analysis_version", 0) or 0) < 4
                or not current_shape
            )
        ):
            # The observer may be newer than the already-running remote worker. Recompute this
            # bounded read model locally so a Work does not need restarting to gain newer views.
            analysis = StudyInsightAnalyzer.analyze(candidates, objective)
        parameters = (
            [value for value in analysis.get("parameters", ()) if isinstance(value, Mapping)]
            if analysis is not None
            else []
        )
        return analysis, parameters

    @classmethod
    def render(
        cls,
        payload: Mapping[str, Any],
        work_index: int,
        *,
        selected_parameter: int,
        message: str,
        width: int,
        height: int,
    ) -> str:
        study, _candidates = StudyRenderer.study(payload, work_index)
        analysis, parameters = cls.analysis(payload, work_index)
        if not isinstance(study, Mapping) or str(study.get("strategy", "")) != "adaptive":
            return _screen(
                [
                    "LambdaForge HPO insights",
                    "",
                    "This study is exhaustive; no adaptive controller is active.",
                    "",
                    "←/b/Esc/q back",
                ],
                width=width,
                height=height,
            )
        if analysis is None:
            return _screen(
                [
                    "LambdaForge HPO insights",
                    "",
                    "The adaptive controller is preparing its first analysis snapshot.",
                    "",
                    "The view refreshes automatically · r refresh · ←/b/Esc/q back",
                ],
                width=width,
                height=height,
            )
        objective = analysis.get("objective", {})
        objective = objective if isinstance(objective, Mapping) else {}
        controller = study.get("controller", {})
        controller = controller if isinstance(controller, Mapping) else {}
        last = controller.get("last", {})
        last = last if isinstance(last, Mapping) else {}
        scheduler = controller.get("scheduler", {})
        scheduler = scheduler if isinstance(scheduler, Mapping) else {}
        next_question = analysis.get("next_question")
        next_question = next_question if isinstance(next_question, Mapping) else {}
        constraints = objective.get("constraints", {})
        constraints = constraints if isinstance(constraints, Mapping) else {}
        guardrails = ", ".join(
            f"{name} "
            + " ".join(
                f"{bound}={float(limit):.5g}"
                for bound, limit in rule.items()
                if bound in {"min", "max"} and isinstance(limit, int | float)
            )
            for name, rule in constraints.items()
            if isinstance(rule, Mapping)
        )
        lines = [
            f"LambdaForge HPO insights · {study.get('name', '-')} · "
            f"{objective.get('metric', '-')} ({objective.get('mode', '-')})",
            (
                f"comparable={analysis.get('candidate_observations', 0)}  "
                f"terminal={analysis.get('terminal_candidate_observations', 0)}  "
                f"running={analysis.get('provisional_candidate_observations', 0)}  "
                f"pruned/censored={analysis.get('censored_pruned_candidates', 0)}  "
                f"infeasible={analysis.get('infeasible_candidates', 0)}  "
                f"analysis={analysis.get('status', 'insufficient')}"
            ),
            f"outcome guardrails: {guardrails or 'none (the declared objective alone decides)'}",
            f"decision model: {analysis.get('decision_model') or 'preparing joint model'}",
            f"controller: {cls._decision(last)}",
            (
                "scheduler: "
                f"slots={scheduler.get('slots_active', '-')} active/"
                f"{scheduler.get('slots_total', '-')} total  "
                f"available={scheduler.get('slots_available', '-')}  "
                f"queued={scheduler.get('queued', 0)}  paused={scheduler.get('paused', 0)}  "
                f"preempted={scheduler.get('preempted', 0)}"
            ),
            (
                "scheduler evidence: "
                f"score={_display_value(last.get('score'))}  "
                f"information={_display_value(last.get('expected_information'))}  "
                f"cost={_seconds(last.get('expected_cost_seconds'))}  "
                f"reason={last.get('reason', 'collecting evidence')}"
            ),
            (
                f"next question: {next_question.get('parameter', '-')} · "
                f"{next_question.get('suggestion', 'More evidence is required.')}"
            ),
            "",
            "  PARAMETER                 KIND         N/VALUES  CONFIDENCE  CURRENT SIGNAL",
        ]
        selected_parameter = min(selected_parameter, max(0, len(parameters) - 1))
        selected = parameters[selected_parameter] if parameters else {}
        detail = cls._detail(selected, width=width)
        capacity = max(1, height - len(lines) - len(detail) - 4)
        start = max(0, selected_parameter - capacity + 1)
        for offset, parameter in enumerate(parameters[start : start + capacity]):
            index = start + offset
            conclusion = str(parameter.get("conclusion", "-"))
            lines.append(
                f"{'▶' if index == selected_parameter else ' '} "
                f"{str(parameter.get('parameter', '-')):<25.25} "
                f"{str(parameter.get('kind', '-')):<12.12} "
                f"{int(parameter.get('observations', 0)):>3}/"
                f"{int(parameter.get('distinct_values', 0)):<3} "
                f"{str(parameter.get('confidence_label', 'low')):<10.10}  "
                f"{conclusion:<{max(8, width - 68)}.{max(8, width - 68)}}"
            )
        if not parameters:
            lines.append("  No hyperparameter has enough objective observations yet.")
        lines.extend(("", *detail, ""))
        lines.append(
            message
            or "↑/↓ hyperparameters · Enter/→ visual detail · r refresh · ←/b/Esc/q candidates · "
            "associations are exploratory, not causal"
        )
        return _screen(lines, width=width, height=height)

    @classmethod
    def _detail(cls, parameter: Mapping[str, Any], *, width: int) -> list[str]:
        if not parameter:
            return ["SELECTED HYPERPARAMETER", "  Waiting for evidence."]
        lines = [
            f"SELECTED HYPERPARAMETER · {parameter.get('parameter', '-')}",
            (
                f"  coverage: {parameter.get('observations', 0)} candidates / "
                f"{parameter.get('distinct_values', 0)} values  |  confidence: "
                f"{parameter.get('confidence', 0):.0%} "
                f"({parameter.get('confidence_label', 'low')})"
            ),
        ]
        if parameter.get("kind") == "numeric":
            observed = parameter.get("observed_range", ())
            observed = observed if isinstance(observed, Sequence) else ()
            threshold = parameter.get("possible_threshold")
            threshold = threshold if isinstance(threshold, Mapping) else {}
            lines.append(
                "  numeric evidence: "
                f"range={_display_value(list(observed))}  "
                f"best observed={_display_value(parameter.get('best_observed_value'))}  "
                f"rank correlation={float(parameter.get('rank_correlation', 0)):+.2f}"
            )
            if threshold:
                lines.append(
                    f"  possible threshold: {_display_value(threshold.get('value'))} · "
                    f"{threshold.get('better_side', '-')} side appears better · "
                    f"effect={float(threshold.get('standardized_effect', 0)):.2f} SD"
                )
        else:
            groups = [value for value in parameter.get("groups", ()) if isinstance(value, Mapping)]
            if groups:
                preview = " · ".join(
                    f"{value.get('label', '-')}: n={value.get('observations', 0)}, "
                    f"mean={float(value.get('objective_mean', 0)):.5g}"
                    for value in groups[:4]
                )
                lines.extend(cls._wrapped("  levels: " + preview, width))
        lines.extend(cls._wrapped("  observed: " + str(parameter.get("conclusion", "-")), width))
        lines.extend(
            cls._wrapped(
                "  suggested evidence: " + str(parameter.get("recommendation", "-")),
                width,
            )
        )
        return lines

    @staticmethod
    def _decision(value: Mapping[str, Any]) -> str:
        if not value:
            return "no adaptive decision has been published yet"
        action = str(value.get("action", "unknown"))
        trial = value.get("trial")
        seed = value.get("seed")
        suffix = ""
        if trial is not None:
            suffix += f" trial={trial}"
        if seed is not None:
            suffix += f" seed={seed}"
        if action == "PROPOSE":
            suffix += f" via {value.get('backend', 'sampler')}"
        if action in {"RESUME", "RESUME_PREEMPTED", "PROMOTE_FIDELITY"}:
            suffix += f" budget={value.get('current', '?')}→{value.get('target', '?')}"
        if action == "SURROGATE_FALLBACK":
            suffix += f" to {value.get('fallback', 'safe fallback')}"
        return action + suffix

    @staticmethod
    def _wrapped(value: str, width: int) -> list[str]:
        return textwrap.wrap(value, width=max(24, width), subsequent_indent="    ") or [value]


class StudyParameterInsightRenderer:
    """Expand one parameter into a bounded response chart and relationship panel."""

    @classmethod
    def render(
        cls,
        payload: Mapping[str, Any],
        work_index: int,
        *,
        selected_parameter: int,
        message: str,
        width: int,
        height: int,
    ) -> str:
        analysis, parameters = StudyInsightRenderer.analysis(payload, work_index)
        if analysis is None or not parameters:
            return _screen(
                [
                    "LambdaForge HPO parameter evidence",
                    "",
                    "No comparable parameter evidence is available yet.",
                    "",
                    "←/b/Esc/q back",
                ],
                width=width,
                height=height,
            )
        selected_parameter = min(selected_parameter, len(parameters) - 1)
        parameter = parameters[selected_parameter]
        objective = analysis.get("objective", {})
        objective = objective if isinstance(objective, Mapping) else {}
        name = str(parameter.get("parameter", "-"))
        metric = str(objective.get("metric", "objective"))
        lines = [
            f"LambdaForge HPO parameter · {name} → {metric}",
            (
                f"evidence={parameter.get('observations', 0)} candidates  "
                f"values={parameter.get('distinct_values', 0)}  "
                f"confidence={float(parameter.get('confidence', 0)):.0%} "
                f"({parameter.get('confidence_label', 'low')})"
            ),
            "",
            "MARGINAL RESPONSE · descriptive, with other parameters uncontrolled",
        ]
        response = parameter.get("response", {})
        response = response if isinstance(response, Mapping) else {}
        points = [value for value in response.get("points", ()) if isinstance(value, Mapping)]
        if response.get("kind") == "numeric-binned" and points:
            x_values = [float(value.get("parameter", 0)) for value in points]
            y_values = [float(value.get("objective", 0)) for value in points]
            lines.extend(
                TerminalTimeSeriesChart.render(
                    f"mean {metric}",
                    y_values,
                    width=width,
                    rows=min(5, max(3, (height - 15) // 2)),
                    x_label=(
                        f"{name}: {min(x_values):.5g} → {max(x_values):.5g} "
                        f"({len(points)} equal-count bins)"
                    ),
                )
            )
            lines.append(
                "  bins: "
                + "  ".join(
                    f"{float(value.get('parameter', 0)):.4g}→"
                    f"{float(value.get('objective', 0)):.4g} (n={value.get('samples', 0)})"
                    for value in points
                )
            )
        elif points:
            lines.extend(cls._categorical_response(points, metric=metric, width=width))
        else:
            lines.append("  More distinct completed candidates are required for a response view.")

        pruning = parameter.get("pruning_signal", {})
        pruning = pruning if isinstance(pruning, Mapping) else {}
        pruning_groups = [
            value for value in pruning.get("groups", ()) if isinstance(value, Mapping)
        ]
        lines.extend(("", "CENSORED PRUNING EVIDENCE · informative, never a completed objective"))
        if pruning.get("observations"):
            lines.append(
                f"  total: {int(pruning.get('pruned', 0))}/"
                f"{int(pruning.get('observations', 0))} terminal candidates pruned "
                f"({float(pruning.get('pruned_rate', 0)):.0%}; smoothed 90% interval "
                f"{float(pruning.get('rate_lower', 0)):.0%}–"
                f"{float(pruning.get('rate_upper', 1)):.0%})"
            )
            for value in pruning_groups[: min(4, max(1, height // 10))]:
                label = value.get("label", value.get("value", "-"))
                lines.append(
                    f"  {_display_value(label):<22.22} "
                    f"{int(value.get('pruned', 0)):>2}/"
                    f"{int(value.get('observations', 0)):<2} pruned candidates  "
                    f"{float(value.get('pruned_rate', 0)):>4.0%}  "
                    f"[{float(value.get('rate_lower', 0)):.0%}, "
                    f"{float(value.get('rate_upper', 1)):.0%}]"
                )
        else:
            lines.append("  No succeeded/pruned terminal candidate evidence is available yet.")

        relationships = [
            value
            for value in parameter.get("joint_relationships", ())
            if isinstance(value, Mapping)
        ]
        lines.extend(
            (
                "",
                "JOINT SURROGATE CONTEXT · controller uses all parameters simultaneously",
                "  Pairwise held-out predictive gain follows; higher-order effects may exist.",
            )
        )
        if parameter.get("interaction_context"):
            lines.extend(
                textwrap.wrap(
                    f"  {parameter['interaction_context']}",
                    width=max(24, width),
                    subsequent_indent="    ",
                )
            )
        if relationships:
            predictive = [
                float(value.get("gain", 0))
                for value in relationships
                if value.get("status") == "predictive"
            ]
            maximum = max(predictive, default=0.0) or 1.0
            lines.append("  OTHER PARAMETER          JOINT CELLS   GAIN (objective SD)   CONF.   N")
            available = max(1, height - len(lines) - 4)
            for value in relationships[:available]:
                gain = float(value.get("gain", 0))
                filled = min(12, round(12 * gain / maximum))
                predictive_status = value.get("status") == "predictive"
                bar = "█" * filled + "░" * (12 - filled) if predictive_status else "coverage only"
                lines.append(
                    f"  {str(value.get('parameter', '-')):<24.24} "
                    f"{int(value.get('joint_cells', 0)):>3}          {bar:<12.12} "
                    f"{gain:>6.3f}  {str(value.get('confidence_label', 'low')):<6.6} "
                    f"{int(value.get('observations', 0)):>3}"
                )
        else:
            lines.append("  Two joint observations are required; predictive gain starts at three.")
        lines.extend(
            (
                "",
                message
                or "live refresh · r refresh · ←/b/Esc/q HPO parameters · "
                "gain is predictive association, not causation",
            )
        )
        return _screen(lines, width=width, height=height)

    @staticmethod
    def _categorical_response(
        points: Sequence[Mapping[str, Any]], *, metric: str, width: int
    ) -> list[str]:
        values = [float(value.get("objective", 0)) for value in points]
        low, high = min(values), max(values)
        span = max(high - low, 1e-12)
        output = [f"  VALUE                     MEAN {metric:<18.18} N"]
        bar_width = min(24, max(8, width - 58))
        for value in points[:12]:
            objective = float(value.get("objective", 0))
            filled = round(bar_width * (objective - low) / span)
            output.append(
                f"  {str(value.get('label', '-')):<25.25} "
                f"{'█' * filled}{'░' * (bar_width - filled)} "
                f"{objective:>10.5g} {int(value.get('samples', 0)):>3}"
            )
        return output


class StudyRunRenderer:
    """Render paged learning curves and a readable epoch-by-epoch dashboard."""

    @classmethod
    def render(
        cls,
        detail: Mapping[str, Any] | None,
        *,
        scroll: int,
        horizontal: int = 0,
        chart_page: int = 0,
        selected_epoch: int | None = None,
        show_raw_log: bool = False,
        show_failure: bool = False,
        message: str,
        width: int,
        height: int,
    ) -> str:
        if detail is None:
            return "LambdaForge study Run\n\nLoading isolated telemetry…\n\n←/b/Esc/q back"
        curves = detail.get("curves", {})
        curves = curves if isinstance(curves, Mapping) else {}
        latest = detail.get("latest_metrics", {})
        latest = latest if isinstance(latest, Mapping) else {}
        parameters = detail.get("parameters", {})
        parameters = parameters if isinstance(parameters, Mapping) else {}
        failure = detail.get("failure")
        failure = failure if isinstance(failure, Mapping) else {}
        chart_filter = detail.get("chart_filter", {})
        chart_filter = chart_filter if isinstance(chart_filter, Mapping) else {}
        objective = detail.get("objective", {})
        objective = objective if isinstance(objective, Mapping) else {}
        epoch_rows = cls.epoch_rows(curves)
        selected_epoch = cls.clamp_epoch_selection(epoch_rows, selected_epoch)
        selected_step = (
            epoch_rows[selected_epoch][0] if epoch_rows and selected_epoch is not None else None
        )
        latest_step = epoch_rows[-1][0] if epoch_rows else None
        best_step, best_objective = cls.best_epoch(
            epoch_rows,
            objective=str(objective.get("metric", "")),
            mode=str(objective.get("mode", "max")),
        )
        recorded_best_step = detail.get("best_step")
        if isinstance(recorded_best_step, int):
            best_step = recorded_best_step
        recorded_best = detail.get("best_objective")
        if isinstance(recorded_best, int | float) and not isinstance(recorded_best, bool):
            best_objective = float(recorded_best)
        elapsed = _elapsed_seconds(detail)
        latest_epoch_time = cls._numeric(latest.get("epoch_time_s"))
        epoch_time_label = (
            f"{latest_epoch_time:.3g}s latest"
            if latest_epoch_time is not None
            else (
                f"~{elapsed / latest_step:.3g}s avg"
                if elapsed is not None and latest_step
                else "collecting"
            )
        )
        validation_time = cls._numeric(latest.get("validation_time_s"))
        validation_label = (
            f"{validation_time:.3g}s latest" if validation_time is not None else "collecting"
        )
        lines = [
            f"LambdaForge Run · Trial {detail.get('trial', '-')} · Seed {detail.get('seed', '-')} "
            f"· {StudyRenderer._gpu_label(detail)} · {detail.get('state', '-')}",
            *_key_value_lines(parameters, width=width, heading="PARAMETERS"),
            (
                f"duration: {_seconds(elapsed)}  |  latest epoch: {latest_step or '-'}  |  "
                f"best epoch: {best_step or '-'}"
                + (
                    f" ({objective.get('metric')}={best_objective:.5g})"
                    if best_objective is not None and objective.get("metric")
                    else ""
                )
                + "  |  "
                f"epoch time: {epoch_time_label}  |  validation: "
                f"{validation_label}"
            ),
        ]
        if failure:
            lines.append(f"failure: {failure.get('type', 'Error')}: {failure.get('message', '')}")
            if show_failure:
                return cls._failure_view(
                    detail,
                    scroll=scroll,
                    horizontal=horizontal,
                    message=message,
                    width=width,
                    height=height,
                )
        preferred = cls._curve_order(
            curves,
            chart_filter=chart_filter,
            objective=str(objective.get("metric", "")),
        )
        chart_columns = 2 if width >= 100 else 1
        chart_capacity = 4 if chart_columns == 2 and height >= 34 else chart_columns
        pages = max(1, (len(preferred) + chart_capacity - 1) // chart_capacity)
        chart_page = min(max(0, chart_page), pages - 1)
        page_start = chart_page * chart_capacity
        visible_names = preferred[page_start : page_start + chart_capacity]
        chart_width = max(24, (width - 2) // chart_columns)
        charts: list[list[str]] = []
        for name in visible_names:
            values = curves.get(name, ())
            values = values if isinstance(values, (list, tuple)) else ()
            observations = [
                value
                for value in values
                if isinstance(value, Mapping)
                and isinstance(value.get("value"), int | float)
                and isinstance(value.get("step"), int)
            ]
            points = [float(value["value"]) for value in observations]
            steps = [int(value["step"]) for value in observations]
            marker_index = (
                steps.index(selected_step)
                if selected_step is not None and selected_step in steps
                else None
            )
            best_index = (
                steps.index(best_step) if best_step is not None and best_step in steps else None
            )
            suffix = "s" if name.endswith("_time_s") else "MiB" if name.endswith("_mb") else ""
            x_label = f"epoch {steps[0]} → {steps[-1]}" if steps else "older ← epochs → latest"
            charts.append(
                TerminalTimeSeriesChart.render(
                    name,
                    points,
                    width=chart_width,
                    rows=4,
                    suffix=suffix,
                    x_label=x_label,
                    selected_index=marker_index,
                    best_index=best_index,
                )
            )
        if charts:
            lines.extend(
                (
                    "",
                    f"LEARNING CURVES · page {chart_page + 1}/{pages} · "
                    f"showing {page_start + 1}-{page_start + len(visible_names)} of "
                    f"{len(preferred)}: {', '.join(visible_names)}  |  "
                    "● selected  ◆ best objective epoch",
                )
            )
            lines.extend(_side_by_side(charts, width=width, row_gap=1))

        if show_raw_log:
            return cls._with_raw_log(
                lines,
                detail,
                scroll=scroll,
                horizontal=horizontal,
                chart_page=chart_page,
                message=message,
                width=width,
                height=height,
            )

        lines.extend(("", f"EPOCH METRICS ({len(epoch_rows)} observed)"))
        metric_names = cls._table_metrics(
            epoch_rows,
            preferred,
            objective=str(objective.get("metric", "")),
            width=width,
        )
        lines.append("    EPOCH  " + "  ".join(f"{name:<16.16}" for name in metric_names))
        selected_values = (
            epoch_rows[selected_epoch][1] if epoch_rows and selected_epoch is not None else {}
        )
        detail_preview = _key_value_lines(
            selected_values,
            width=width,
            heading=f"SELECTED EPOCH {selected_step or '-'} · Enter/→ opens every metric",
        )[:3]
        capacity = max(1, height - len(lines) - len(detail_preview) - 3)
        start = max(0, (selected_epoch or 0) - capacity + 1)
        for index, (step, values) in enumerate(epoch_rows[start : start + capacity], start):
            cells = [cls._format_metric(values.get(name)) for name in metric_names]
            lines.append(
                f"{'▶' if index == selected_epoch else ' '}"
                f"{'★' if step == best_step else ' '} {step:<5}  "
                + "  ".join(f"{value:<16.16}" for value in cells)
            )
        if not epoch_rows:
            lines.append("  Waiting for the first structured epoch observation…")
        lines.extend(("", *detail_preview))
        suffix = " · source truncated safely" if detail.get("log_truncated") else ""
        lines.extend(
            (
                "",
                message
                or "★ best objective epoch · live refresh · ↑/↓ epoch · "
                "Enter/→ details · n/p curve page "
                "· o raw output"
                + (" · e failure details" if failure else "")
                + f" · ←/b/Esc/q back{suffix}",
            )
        )
        return _screen(lines, width=width, height=height)

    @staticmethod
    def failure_lines(detail: Mapping[str, Any] | None) -> list[str]:
        """Return the complete persisted failure as one scrollable document."""
        failure = detail.get("failure") if isinstance(detail, Mapping) else None
        if not isinstance(failure, Mapping):
            return []
        traceback = str(failure.get("traceback") or "").strip()
        message = str(failure.get("message") or "Scientific execution failed.")
        lines = [
            "SCIENTIFIC FAILURE",
            f"Type: {failure.get('type') or 'Exception'}",
            f"Phase: {failure.get('phase') or 'Work.run'}",
            f"Persisted result: {failure.get('result_path') or '-'}",
            "",
            "Message:",
            *(message.splitlines() or [message]),
        ]
        if traceback:
            lines.extend(("", "Traceback:", *traceback.splitlines()))
        else:
            lines.extend(("", "No persisted traceback is available for this Run."))
        return lines

    @classmethod
    def _failure_view(
        cls,
        detail: Mapping[str, Any],
        *,
        scroll: int,
        horizontal: int,
        message: str,
        width: int,
        height: int,
    ) -> str:
        lines = cls.failure_lines(detail)
        capacity = max(1, height - 4)
        maximum = max(0, len(lines) - capacity)
        start = min(maximum, max(0, scroll))
        return "\n".join(
            [
                (
                    f"LambdaForge Run failure · Trial {detail.get('trial', '-')} · "
                    f"Seed {detail.get('seed', '-')} · lines "
                    f"{start + 1 if lines else 0}-{min(len(lines), start + capacity)} "
                    f"of {len(lines)}"
                )[:width],
                "",
                *(
                    _horizontal_view(line, offset=horizontal, width=width)
                    for line in lines[start : start + capacity]
                ),
                "",
                (
                    message
                    or "failure details · ↑/↓ scroll · PgUp/PgDn · Shift+←/→ columns "
                    "· e back to metrics · ←/b/Esc/q back"
                )[:width],
            ]
        )

    @classmethod
    def _with_raw_log(
        cls,
        lines: list[str],
        detail: Mapping[str, Any],
        *,
        scroll: int,
        horizontal: int,
        chart_page: int,
        message: str,
        width: int,
        height: int,
    ) -> str:
        lines.extend(("", "RAW ISOLATED RUN LOG"))
        log_lines = str(detail.get("log", "")).splitlines()
        capacity = max(1, height - len(lines) - 3)
        maximum = max(0, len(log_lines) - capacity)
        start = min(maximum, max(0, scroll))
        lines.extend(
            [
                _horizontal_view(line, offset=horizontal, width=width)
                for line in log_lines[start : start + capacity]
            ]
            or ["No Run output has been emitted yet."]
        )
        suffix = " · source truncated safely" if detail.get("log_truncated") else ""
        failure = detail.get("failure")
        lines.extend(
            (
                "",
                message
                or f"raw output · ↑/↓ scroll · PgUp/PgDn · Shift+←/→ columns "
                f"({horizontal}) · o epoch metrics · n/p curve page {chart_page + 1} "
                + ("· e failure details " if isinstance(failure, Mapping) else "")
                + f"· ←/b/Esc/q back{suffix}",
            )
        )
        return _screen(lines, width=width, height=height)

    @classmethod
    def epoch_rows(cls, curves: Mapping[str, Any]) -> list[tuple[int, dict[str, float]]]:
        """Pivot scalar curves into deterministic epoch records for the TUI."""
        rows: dict[int, dict[str, float]] = {}
        for raw_name, raw_values in curves.items():
            if not isinstance(raw_values, (list, tuple)):
                continue
            for observation in raw_values:
                if not isinstance(observation, Mapping):
                    continue
                step, value = observation.get("step"), observation.get("value")
                if isinstance(step, int) and isinstance(value, int | float):
                    rows.setdefault(step, {})[str(raw_name)] = float(value)
        return sorted(rows.items())

    @staticmethod
    def clamp_epoch_selection(
        rows: Sequence[tuple[int, Mapping[str, float]]], selected: int | None
    ) -> int | None:
        if not rows:
            return None
        return len(rows) - 1 if selected is None else min(max(0, selected), len(rows) - 1)

    @staticmethod
    def best_epoch(
        rows: Sequence[tuple[int, Mapping[str, float]]],
        *,
        objective: str,
        mode: str,
    ) -> tuple[int | None, float | None]:
        observed = [
            (step, float(values[objective]))
            for step, values in rows
            if objective and objective in values
        ]
        if not observed:
            return None, None
        return (min if mode == "min" else max)(observed, key=lambda value: value[1])

    @classmethod
    def _table_metrics(
        cls,
        rows: Sequence[tuple[int, Mapping[str, float]]],
        preferred: Sequence[str],
        *,
        objective: str,
        width: int,
    ) -> list[str]:
        available = {name for _, values in rows for name in values}
        order = [objective, "epoch_time_s", "validation_time_s", *preferred, *sorted(available)]
        unique: list[str] = []
        for name in order:
            if name and name in available and name not in unique:
                unique.append(name)
        return unique[: max(1, min(6, (width - 10) // 18))]

    @staticmethod
    def _format_metric(value: Any) -> str:
        return f"{float(value):.6g}" if isinstance(value, int | float) else "-"

    @staticmethod
    def _numeric(value: Any) -> float | None:
        return float(value) if isinstance(value, int | float) else None

    @classmethod
    def chart_page_count(cls, detail: Mapping[str, Any] | None, *, width: int, height: int) -> int:
        if detail is None:
            return 1
        curves = detail.get("curves", {})
        curves = curves if isinstance(curves, Mapping) else {}
        chart_filter = detail.get("chart_filter", {})
        chart_filter = chart_filter if isinstance(chart_filter, Mapping) else {}
        objective = detail.get("objective", {})
        objective = objective if isinstance(objective, Mapping) else {}
        count = len(
            cls._curve_order(
                curves,
                chart_filter=chart_filter,
                objective=str(objective.get("metric", "")),
            )
        )
        columns = 2 if width >= 100 else 1
        capacity = 4 if columns == 2 and height >= 34 else columns
        return max(1, (count + capacity - 1) // capacity)

    @staticmethod
    def _curve_order(
        curves: Mapping[str, Any],
        *,
        chart_filter: Mapping[str, Any] | None = None,
        objective: str = "",
    ) -> list[str]:
        names = [str(value) for value in curves]
        selected = chart_filter or {}
        include = selected.get("include")
        include = (
            tuple(str(value) for value in include) if isinstance(include, (list, tuple)) else ()
        )
        exclude = selected.get("exclude")
        exclude = (
            tuple(str(value) for value in exclude) if isinstance(exclude, (list, tuple)) else ()
        )
        if include:
            names = [
                name for name in names if any(fnmatchcase(name, pattern) for pattern in include)
            ]
        if exclude:
            names = [
                name for name in names if not any(fnmatchcase(name, pattern) for pattern in exclude)
            ]
        priority = (
            objective,
            "epoch_time_s",
            "validation_time_s",
            "gpu_mem_mb",
            "gpu_peak_reserved_mb",
            "val_",
            "train_",
        )
        return sorted(
            names,
            key=lambda name: next(
                (
                    index
                    for index, prefix in enumerate(priority)
                    if prefix and (name == prefix or name.startswith(prefix))
                ),
                len(priority),
            ),
        )


class StudyEpochRenderer:
    """Expand every scalar for one selected epoch without horizontal panning."""

    @staticmethod
    def render(
        detail: Mapping[str, Any] | None,
        *,
        selected_epoch: int | None,
        message: str,
        width: int,
        height: int,
    ) -> str:
        if detail is None:
            return "LambdaForge epoch details\n\nLoading telemetry…\n\n←/b/Esc/q back"
        curves = detail.get("curves", {})
        curves = curves if isinstance(curves, Mapping) else {}
        rows = StudyRunRenderer.epoch_rows(curves)
        selection = StudyRunRenderer.clamp_epoch_selection(rows, selected_epoch)
        if selection is None:
            return (
                "LambdaForge epoch details\n\nNo structured epoch metrics are available yet."
                "\n\nlive refresh · ←/b/Esc/q back"
            )
        step, values = rows[selection]
        elapsed = _elapsed_seconds(detail)
        lines = [
            f"LambdaForge epoch · Trial {detail.get('trial', '-')} · "
            f"Seed {detail.get('seed', '-')} · epoch {step}",
            (
                f"epoch {selection + 1}/{len(rows)} observed  |  "
                f"Run duration {_seconds(elapsed)}  |  {len(values)} scalar metrics"
            ),
            "",
            *_key_value_lines(values, width=width, heading="ALL EPOCH METRICS"),
            "",
            message or "live refresh · ↑/↓ previous/next epoch · ←/b/Esc/q back",
        ]
        return _screen(lines, width=width, height=height)


def json_compact(value: Any, *, width: int) -> str:
    """Render deterministic compact JSON without allowing one row to flood the screen."""
    import json

    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return text if len(text) <= width else text[: max(1, width - 1)] + "…"


class HistoryChart:
    """Render one clear whole-cluster curve plus the current personal share."""

    @staticmethod
    def render(
        label: str,
        total: tuple[float | None, ...],
        personal: tuple[float | None, ...],
        *,
        width: int,
        rows: int = 4,
        window_seconds: float = 60,
    ) -> list[str]:
        current_mine = next((value for value in reversed(personal) if value is not None), None)
        mine = HistoryChart._current(current_mine)
        title = f"{label} · whole cluster"
        if personal:
            title += f" · mine now={mine}"
        lines = TerminalTimeSeriesChart.render(
            title,
            total,
            width=width,
            rows=rows,
            minimum=0,
            maximum=100,
            suffix="%",
            x_label=f"← {window_seconds:g}s history · now",
        )
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
        chart_width = max(24, (width - 2) // 2) if width >= 100 else width
        charts = [
            HistoryChart.render(
                "CPU load",
                history.series(name, "cpu"),
                history.series(name, "my_cpu"),
                width=chart_width,
                window_seconds=history.window_seconds,
            ),
            HistoryChart.render(
                "RAM used",
                history.series(name, "ram"),
                history.series(name, "my_ram"),
                width=chart_width,
                window_seconds=history.window_seconds,
            ),
            HistoryChart.render(
                "GPU utilization",
                history.series(name, "gpu"),
                (),
                width=chart_width,
                window_seconds=history.window_seconds,
            ),
            HistoryChart.render(
                "GPU memory",
                history.series(name, "gpu_memory"),
                history.series(name, "my_gpu"),
                width=chart_width,
                window_seconds=history.window_seconds,
            ),
        ]
        lines.extend(_side_by_side(charts, width=width))
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
        job_id: str,
        text: str,
        *,
        scroll: int,
        horizontal: int = 0,
        include_traceback: bool = False,
        message: str,
        width: int,
        height: int,
    ) -> str:
        lines = text.splitlines()
        capacity = max(1, height - 4)
        start = min(max(0, len(lines) - capacity), max(0, scroll))
        end = min(len(lines), start + capacity)
        header = (
            f"LambdaForge log · {job_id} · lines {start + 1 if lines else 0}-{end} of "
            f"{len(lines)} · column {horizontal + 1}"
        )
        return "\n".join(
            [
                header[:width],
                "",
                *(
                    _horizontal_view(line, offset=horizontal, width=width)
                    for line in lines[start:end]
                ),
                "",
                (
                    message
                    or "live refresh · ↑/↓ line · PgUp/PgDn page · "
                    "Shift+←/→ columns · e failure details "
                    f"({'shown' if include_traceback else 'hidden'}) · ←/b/Esc/q back"
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


def _collect_logs(
    jobs: JobService,
    job_id: str,
    include_traceback: bool,
    connection: Connection,
) -> None:
    try:
        connection.send((jobs.logs(job_id, include_traceback=include_traceback), None))
    except BaseException as error:
        connection.send((None, f"{error.__class__.__name__}: {error}"))
    finally:
        connection.close()


def _collect_study_run(
    jobs: JobService,
    job_id: str,
    run_key: str,
    connection: Connection,
) -> None:
    """Load bounded per-Run curves and logs without blocking terminal input."""
    try:
        connection.send((jobs.study_run(job_id, run_key, tail=2_000, curve_points=80), None))
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

    def __init__(self, jobs: JobService, job_id: str, *, include_traceback: bool = False) -> None:
        super().__init__(
            _collect_logs,
            (jobs, job_id, include_traceback),
            "lambdaforge-top-logs",
        )


class StudyRunProcess(BackgroundProcess):
    """Load one isolated scientific Run dashboard in the background."""

    def __init__(self, jobs: JobService, job_id: str, run_key: str) -> None:
        super().__init__(
            _collect_study_run,
            (jobs, job_id, run_key),
            "lambdaforge-top-study-run",
        )


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
        selected = selected_cluster = detail_selected = log_scroll = log_horizontal = 0
        selected_candidate = selected_study_run = study_log_scroll = study_horizontal = 0
        selected_hpo_parameter = 0
        study_chart_page = 0
        study_epoch: int | None = None
        study_raw_log = False
        study_failure_expanded = False
        log_include_traceback = False
        focus, mode, log_job, log_label, log_text, message = (
            "jobs",
            "overview",
            "",
            "",
            "",
            "",
        )
        pending_cancel: tuple[str, str] | None = None
        pending_history: tuple[str, str] | None = None
        return_mode = "overview"
        payload: Mapping[str, Any] = {}
        history, dirty = ResourceHistory(self.history_seconds), True
        poller = SnapshotProcess(self.overview)
        log_poller: LogProcess | None = None
        study_poller: StudyRunProcess | None = None
        study_detail: Mapping[str, Any] | None = None
        study_job = study_key = ""
        action_poller: HistoryActionProcess | None = None
        poller.start()
        next_refresh = time.monotonic() + self.interval
        next_log_refresh = float("inf")
        next_study_refresh = float("inf")
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
                        capacity = max(1, shutil.get_terminal_size((120, 30)).lines - 4)
                        old_maximum = max(0, len(log_text.splitlines()) - capacity)
                        new_maximum = max(0, len(new_text.splitlines()) - capacity)
                        following = not log_text or log_scroll >= old_maximum
                        log_text = new_text
                        log_scroll = new_maximum if following else min(log_scroll, new_maximum)
                        message, log_poller, dirty = (
                            (f"Log load failed: {error}" if error else ""),
                            None,
                            True,
                        )
                        next_log_refresh = time.monotonic() + self.interval
                    if study_poller is not None and (loaded := study_poller.take()):
                        value, error = loaded
                        if isinstance(value, Mapping):
                            study_detail = value
                        message = f"Run telemetry failed: {error}" if error else ""
                        study_poller, dirty = None, True
                        next_study_refresh = time.monotonic() + self.interval
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
                        log_poller = LogProcess(
                            self.jobs,
                            log_job,
                            include_traceback=log_include_traceback,
                        )
                        log_poller.start()
                        next_log_refresh = time.monotonic() + self.interval
                    if (
                        mode in {"study-run", "study-epoch"}
                        and study_job
                        and study_key
                        and study_poller is None
                        and time.monotonic() >= next_study_refresh
                    ):
                        study_poller = StudyRunProcess(self.jobs, study_job, study_key)
                        study_poller.start()
                        next_study_refresh = time.monotonic() + self.interval
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
                    _, candidate_items = StudyRenderer.study(payload, selected)
                    selected_candidate = min(selected_candidate, max(0, len(candidate_items) - 1))
                    _, study_run_items = StudyRenderer.candidate_runs(
                        payload, selected, selected_candidate
                    )
                    _, hpo_parameter_items = StudyInsightRenderer.analysis(payload, selected)
                    selected_hpo_parameter = min(
                        selected_hpo_parameter,
                        max(0, len(hpo_parameter_items) - 1),
                    )
                    selected_study_run = min(selected_study_run, max(0, len(study_run_items) - 1))
                    if mode == "work":
                        detail_selected = min(detail_selected, max(0, len(attempt_items) - 1))
                    size = shutil.get_terminal_size((120, 30))
                    if dirty:
                        if mode == "logs":
                            rendered = LogViewerRenderer.render(
                                log_label or log_job,
                                log_text,
                                scroll=log_scroll,
                                horizontal=log_horizontal,
                                include_traceback=log_include_traceback,
                                message=message or ("Loading complete log…" if log_poller else ""),
                                width=size.columns,
                                height=size.lines,
                            )
                        elif mode == "study":
                            rendered = StudyRenderer.render_candidates(
                                payload,
                                selected,
                                selected_candidate=selected_candidate,
                                message=message,
                                width=size.columns,
                                height=size.lines,
                            )
                        elif mode == "study-candidate":
                            rendered = StudyRenderer.render_runs(
                                payload,
                                selected,
                                selected_candidate,
                                selected_run=selected_study_run,
                                message=message,
                                width=size.columns,
                                height=size.lines,
                            )
                        elif mode == "study-hpo":
                            rendered = StudyInsightRenderer.render(
                                payload,
                                selected,
                                selected_parameter=selected_hpo_parameter,
                                message=message,
                                width=size.columns,
                                height=size.lines,
                            )
                        elif mode == "study-hpo-parameter":
                            rendered = StudyParameterInsightRenderer.render(
                                payload,
                                selected,
                                selected_parameter=selected_hpo_parameter,
                                message=message,
                                width=size.columns,
                                height=size.lines,
                            )
                        elif mode == "study-run":
                            rendered = StudyRunRenderer.render(
                                study_detail,
                                scroll=study_log_scroll,
                                horizontal=study_horizontal,
                                chart_page=study_chart_page,
                                selected_epoch=study_epoch,
                                show_raw_log=study_raw_log,
                                show_failure=study_failure_expanded,
                                message=message
                                or ("Refreshing Run telemetry…" if study_poller else ""),
                                width=size.columns,
                                height=size.lines,
                            )
                        elif mode == "study-epoch":
                            rendered = StudyEpochRenderer.render(
                                study_detail,
                                selected_epoch=study_epoch,
                                message=message
                                or ("Refreshing Run telemetry…" if study_poller else ""),
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
                        rendered = TerminalTheme.apply(
                            rendered,
                            enabled=TerminalTheme.enabled(self.stream),
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
                        elif key in {"l", "]", "\x1b[1;2C"}:
                            maximum_column = max(
                                0,
                                max((len(line) for line in log_text.splitlines()), default=0)
                                - size.columns,
                            )
                            log_horizontal, dirty = min(maximum_column, log_horizontal + 8), True
                        elif key in {"h", "[", "\x1b[1;2D"}:
                            log_horizontal, dirty = max(0, log_horizontal - 8), True
                        elif key in {"e", "E"}:
                            log_include_traceback = not log_include_traceback
                            if log_poller is not None:
                                log_poller.close()
                            log_poller = LogProcess(
                                self.jobs,
                                log_job,
                                include_traceback=log_include_traceback,
                            )
                            log_poller.start()
                            message = (
                                "Loading persisted failure details…"
                                if log_include_traceback
                                else "Hiding persisted tracebacks…"
                            )
                            dirty = True
                        continue
                    if mode == "study-epoch":
                        curves = (
                            study_detail.get("curves", {})
                            if isinstance(study_detail, Mapping)
                            else {}
                        )
                        curves = curves if isinstance(curves, Mapping) else {}
                        epochs = StudyRunRenderer.epoch_rows(curves)
                        current_epoch = StudyRunRenderer.clamp_epoch_selection(epochs, study_epoch)
                        if key in {"q", "Q", "b", "B", "\x1b", "\x1b[D"}:
                            mode, message, dirty = "study-run", "", True
                        elif key in {"j", "\x1b[B"} and current_epoch is not None:
                            study_epoch = min(len(epochs) - 1, current_epoch + 1)
                            dirty = True
                        elif key in {"k", "\x1b[A"} and current_epoch is not None:
                            study_epoch = max(0, current_epoch - 1)
                            dirty = True
                        elif key == "r" and study_poller is None:
                            study_poller = StudyRunProcess(self.jobs, study_job, study_key)
                            study_poller.start()
                            message, dirty = "Refreshing Run telemetry…", True
                        continue
                    if mode == "study-run":
                        log_lines = (
                            StudyRunRenderer.failure_lines(study_detail)
                            if study_failure_expanded
                            else (
                                str(study_detail.get("log", "")).splitlines()
                                if isinstance(study_detail, Mapping)
                                else []
                            )
                        )
                        page = max(1, size.lines - 12)
                        maximum = max(0, len(log_lines) - page)
                        curves = (
                            study_detail.get("curves", {})
                            if isinstance(study_detail, Mapping)
                            else {}
                        )
                        curves = curves if isinstance(curves, Mapping) else {}
                        epochs = StudyRunRenderer.epoch_rows(curves)
                        current_epoch = StudyRunRenderer.clamp_epoch_selection(epochs, study_epoch)
                        chart_pages = StudyRunRenderer.chart_page_count(
                            study_detail,
                            width=size.columns,
                            height=size.lines,
                        )
                        study_chart_page = min(study_chart_page, chart_pages - 1)
                        if key in {"q", "Q", "b", "B", "\x1b", "\x1b[D"}:
                            if study_poller is not None:
                                study_poller.close()
                                study_poller = None
                            mode, message, dirty = "study-candidate", "", True
                            next_study_refresh = float("inf")
                        elif key in {"j", "\x1b[B"}:
                            if study_raw_log or study_failure_expanded:
                                study_log_scroll = min(maximum, study_log_scroll + 1)
                            elif current_epoch is not None:
                                study_epoch = (
                                    None
                                    if study_epoch is None or current_epoch == len(epochs) - 1
                                    else current_epoch + 1
                                )
                            dirty = True
                        elif key in {"k", "\x1b[A"}:
                            if study_raw_log or study_failure_expanded:
                                study_log_scroll = max(0, study_log_scroll - 1)
                            elif current_epoch is not None:
                                study_epoch = max(0, current_epoch - 1)
                            dirty = True
                        elif key == "\x1b[6~":
                            if study_raw_log or study_failure_expanded:
                                study_log_scroll = min(maximum, study_log_scroll + page)
                            elif current_epoch is not None:
                                study_epoch = min(len(epochs) - 1, current_epoch + 10)
                            dirty = True
                        elif key == "\x1b[5~":
                            if study_raw_log or study_failure_expanded:
                                study_log_scroll = max(0, study_log_scroll - page)
                            elif current_epoch is not None:
                                study_epoch = max(0, current_epoch - 10)
                            dirty = True
                        elif key in {"g", "\x1b[H", "\x1b[1~"}:
                            if study_raw_log or study_failure_expanded:
                                study_log_scroll = 0
                            elif epochs:
                                study_epoch = 0
                            dirty = True
                        elif key in {"G", "\x1b[F", "\x1b[4~"}:
                            if study_raw_log or study_failure_expanded:
                                study_log_scroll = maximum
                            else:
                                study_epoch = None
                            dirty = True
                        elif key in {"n", "N"}:
                            study_chart_page = min(chart_pages - 1, study_chart_page + 1)
                            dirty = True
                        elif key in {"p", "P"}:
                            study_chart_page = max(0, study_chart_page - 1)
                            dirty = True
                        elif key in {"l", "\x1b[1;2C"} and (
                            study_raw_log or study_failure_expanded
                        ):
                            maximum_column = max(
                                0,
                                max((len(line) for line in log_lines), default=0) - size.columns,
                            )
                            study_horizontal, dirty = (
                                min(maximum_column, study_horizontal + 8),
                                True,
                            )
                        elif key in {"h", "\x1b[1;2D"} and (
                            study_raw_log or study_failure_expanded
                        ):
                            study_horizontal, dirty = max(0, study_horizontal - 8), True
                        elif key in {"o", "O"}:
                            study_failure_expanded = False
                            study_raw_log = not study_raw_log
                            message, dirty = "", True
                        elif (
                            key in {"e", "E"}
                            and isinstance(study_detail, Mapping)
                            and isinstance(study_detail.get("failure"), Mapping)
                        ):
                            study_failure_expanded = not study_failure_expanded
                            study_raw_log = False
                            study_log_scroll = study_horizontal = 0
                            message, dirty = "", True
                        elif (
                            key in {"\r", "\n", "\x1b[C"}
                            and not study_raw_log
                            and not study_failure_expanded
                            and epochs
                        ):
                            study_epoch = current_epoch
                            mode, message, dirty = "study-epoch", "", True
                        elif key == "r" and study_poller is None:
                            study_poller = StudyRunProcess(self.jobs, study_job, study_key)
                            study_poller.start()
                            message, dirty = "Refreshing Run telemetry…", True
                        continue
                    if pending_cancel is not None:
                        cancel_scope, cancel_selector = pending_cancel
                        if key in {"x", "X", "y", "Y", "\r", "\n"}:
                            try:
                                if cancel_scope == "work":
                                    result = self._work_service().cancel(cancel_selector)
                                    stopped_count = len(result["cancelled_jobs"])
                                    reconciled_count = len(
                                        result.get("reconciled_cancelled_jobs", ())
                                    )
                                    message = (
                                        f"Work cancelled: {stopped_count} active Job(s) stopped; "
                                        f"{reconciled_count} prior cancellation(s) reconciled."
                                    )
                                else:
                                    record = self.jobs.cancel(cancel_selector)
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
                                f"Cancel {cancel_scope} {cancel_selector}? Press x again, y or "
                                "Enter to confirm; "
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
                    if mode == "study-hpo":
                        if key in {"q", "Q", "b", "B", "\x1b", "\x1b[D"}:
                            mode, message, dirty = "study", "", True
                        elif key in {"j", "\x1b[B"}:
                            selected_hpo_parameter = min(
                                max(0, len(hpo_parameter_items) - 1),
                                selected_hpo_parameter + 1,
                            )
                            dirty = True
                        elif key in {"k", "\x1b[A"}:
                            selected_hpo_parameter = max(0, selected_hpo_parameter - 1)
                            dirty = True
                        elif key in {"\r", "\n", "\x1b[C"} and hpo_parameter_items:
                            mode, message, dirty = "study-hpo-parameter", "", True
                        elif key == "r":
                            if not poller.running:
                                poller.start()
                            message, dirty = "Refreshing HPO evidence…", True
                        continue
                    if mode == "study-hpo-parameter":
                        if key in {"q", "Q", "b", "B", "\x1b", "\x1b[D"}:
                            mode, message, dirty = "study-hpo", "", True
                        elif key == "r":
                            if not poller.running:
                                poller.start()
                            message, dirty = "Refreshing parameter evidence…", True
                        continue
                    if mode == "study":
                        if key in {"q", "Q", "b", "B", "\x1b", "\x1b[D"}:
                            mode, message, dirty = "overview", "", True
                        elif key in {"j", "\x1b[B"}:
                            selected_candidate = min(
                                max(0, len(candidate_items) - 1),
                                selected_candidate + 1,
                            )
                            selected_study_run, dirty = 0, True
                        elif key in {"k", "\x1b[A"}:
                            selected_candidate = max(0, selected_candidate - 1)
                            selected_study_run, dirty = 0, True
                        elif key in {"\r", "\n", "\x1b[C"} and candidate_items:
                            mode, selected_study_run, message, dirty = (
                                "study-candidate",
                                0,
                                "",
                                True,
                            )
                        elif key in {"a", "A"}:
                            mode, detail_selected, message, dirty = "work", 0, "", True
                        elif key in {"i", "I"}:
                            mode, selected_hpo_parameter, message, dirty = (
                                "study-hpo",
                                0,
                                "",
                                True,
                            )
                        elif key == "r":
                            if not poller.running:
                                poller.start()
                            message, dirty = "Refreshing study state…", True
                        elif key in {"x", "X"} and items:
                            selected_work = items[selected]
                            pending_cancel = (
                                "work",
                                str(
                                    selected_work.get("work_id") or _selected_job_id(selected_work)
                                ),
                            )
                            message = (
                                "Cancel this Work and every active Job/process it owns? Press x "
                                "again, y or Enter to confirm; n/Esc keeps it running."
                            )
                            dirty = True
                        continue
                    if mode == "study-candidate":
                        if key in {"q", "Q", "b", "B", "\x1b", "\x1b[D"}:
                            mode, message, dirty = "study", "", True
                        elif key in {"j", "\x1b[B"}:
                            selected_study_run = min(
                                max(0, len(study_run_items) - 1),
                                selected_study_run + 1,
                            )
                            dirty = True
                        elif key in {"k", "\x1b[A"}:
                            selected_study_run = max(0, selected_study_run - 1)
                            dirty = True
                        elif key in {"\r", "\n", "\x1b[C"} and study_run_items:
                            selected_run = study_run_items[selected_study_run]
                            study_job = _selected_job_id(items[selected])
                            study_key = str(selected_run.get("key", ""))
                            study_detail, study_log_scroll, study_horizontal, message = (
                                None,
                                0,
                                0,
                                "",
                            )
                            study_chart_page, study_epoch, study_raw_log = 0, None, False
                            study_failure_expanded = False
                            mode, dirty = "study-run", True
                            study_poller = StudyRunProcess(self.jobs, study_job, study_key)
                            study_poller.start()
                            next_study_refresh = time.monotonic() + self.interval
                        elif key in {"i", "I"}:
                            mode, selected_hpo_parameter, message, dirty = (
                                "study-hpo",
                                0,
                                "",
                                True,
                            )
                        elif key in {"x", "X"} and items:
                            selected_work = items[selected]
                            pending_cancel = (
                                "work",
                                str(
                                    selected_work.get("work_id") or _selected_job_id(selected_work)
                                ),
                            )
                            message = (
                                "Cancel this Work and every active Job/process it owns? Press x "
                                "again, y or Enter to confirm; n/Esc keeps it running."
                            )
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
                            log_text, log_scroll, log_horizontal, message = "", 0, 0, ""
                            return_mode, mode = "work", "logs"
                            log_include_traceback = False
                            log_poller = LogProcess(self.jobs, log_job)
                            log_poller.start()
                            next_log_refresh = time.monotonic() + self.interval
                            dirty = True
                        elif key in {"x", "X"} and attempt_items:
                            pending_cancel = (
                                "job",
                                str(attempt_items[detail_selected].get("job_id", "")),
                            )
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
                            log_text, log_scroll, log_horizontal, message = "", 0, 0, ""
                            return_mode, mode = "cluster", "logs"
                            log_include_traceback = False
                            log_poller = LogProcess(self.jobs, log_job)
                            log_poller.start()
                            next_log_refresh = time.monotonic() + self.interval
                            dirty = True
                        elif key in {"x", "X"} and detail_items:
                            pending_cancel = (
                                "job",
                                str(detail_items[detail_selected]["job_id"]),
                            )
                            message = (
                                f"Cancel Job {pending_cancel[1]}? Press x again, y or Enter to "
                                "confirm; "
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
                            selected_work = items[selected]
                            mode = "study" if _should_open_study(selected_work) else "work"
                            selected_candidate = selected_study_run = 0
                            message, dirty = "", True
                        elif items:
                            return_mode = "overview"
                            log_job = _selected_job_id(items[selected])
                            log_label = _job_label(payload, items[selected])
                            log_text, log_scroll, log_horizontal, message, mode = (
                                "",
                                0,
                                0,
                                "",
                                "logs",
                            )
                            log_include_traceback = False
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
                        selected_item = items[selected]
                        pending_cancel = (
                            (
                                "work",
                                str(
                                    selected_item.get("work_id") or _selected_job_id(selected_item)
                                ),
                            )
                            if work_items
                            else ("job", _selected_job_id(selected_item))
                        )
                        message = (
                            f"Cancel {pending_cancel[0]} {pending_cancel[1]}? Press x again, y or "
                            "Enter to confirm; "
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
            if study_poller is not None:
                study_poller.close()
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
