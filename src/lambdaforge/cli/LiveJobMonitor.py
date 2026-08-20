"""Dependency-light interactive renderer for global control-plane snapshots."""

from __future__ import annotations

import select
import shutil
import sys
import time
from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Any, TextIO

from lambdaforge.cli.common import age, job_resources
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.OverviewService import OverviewService


class MonitorRenderer:
    """Render one immutable snapshot; terminal input and services remain outside this class."""

    _SPARK = "▁▂▃▄▅▆▇█"

    @classmethod
    def render(
        cls,
        payload: Mapping[str, Any],
        *,
        selected: int = 0,
        log_text: str = "",
        message: str = "",
        width: int = 120,
    ) -> str:
        jobs = payload.get("jobs", {})
        items = jobs.get("items", ()) if isinstance(jobs, Mapping) else ()
        items = items if isinstance(items, list) else []
        states = jobs.get("by_state", {}) if isinstance(jobs, Mapping) else {}
        generated = str(payload.get("generated_at_utc", ""))
        lines = [
            "LambdaForge live jobs",
            (
                f"updated {generated[11:19] or '-'}  "
                f"preparing={states.get('preparing', 0)} staging={states.get('staging', 0)} "
                f"queued={states.get('queued', 0)} running={states.get('running', 0)} "
                f"failed={states.get('failed', 0)} total={jobs.get('total', 0)}"
            ),
            "",
            "CLUSTER           STATUS    CPU             RAM             GPU",
        ]
        for cluster in payload.get("clusters", ()):
            if not isinstance(cluster, Mapping):
                continue
            observed = cluster.get("observed", {})
            observed = observed if isinstance(observed, Mapping) else {}
            cpu = observed.get("cpu_load")
            ram_total = observed.get("ram_total_bytes")
            ram_available = observed.get("ram_available_bytes")
            ram = (
                100.0 * (1.0 - float(ram_available) / float(ram_total))
                if isinstance(ram_total, (int, float))
                and ram_total
                and isinstance(ram_available, (int, float))
                else None
            )
            gpus = observed.get("gpus", ())
            gpu_values = [
                float(value.get("utilization_percent", 0))
                for value in gpus
                if isinstance(value, Mapping)
            ] if isinstance(gpus, (list, tuple)) else []
            gpu = sum(gpu_values) / len(gpu_values) if gpu_values else None
            lines.append(
                f"{str(cluster.get('cluster', '-')):<17.17} "
                f"{'online' if cluster.get('online') else 'offline':<9} "
                f"{cls._bar(cpu):<15} {cls._bar(ram):<15} {cls._bar(gpu):<15}"
            )
        lines.extend(
            (
                "",
                "  JOB                             NAME              TYPE           "
                "STATE       CLUSTER      AGE      RESOURCES",
            )
        )
        visible = items[: max(1, shutil.get_terminal_size((width, 30)).lines - 16)]
        for index, job in enumerate(visible):
            metadata = job.get("metadata", {}) if isinstance(job, Mapping) else {}
            metadata = metadata if isinstance(metadata, Mapping) else {}
            marker = "▶" if index == selected else " "
            lines.append(
                f"{marker} {str(job.get('job_id', '-')):<32.32} "
                f"{str(metadata.get('name', '-')):<17.17} "
                f"{str(job.get('job_type', '-')):<14.14} "
                f"{str(job.get('state', '-')):<11.11} "
                f"{str(job.get('cluster', '-')):<12.12} "
                f"{age(str(job.get('created_at_utc', ''))):<8} "
                f"{job_resources(job.get('resources', {}))}"
            )
        if visible:
            selected_job = visible[min(selected, len(visible) - 1)]
            metadata = selected_job.get("metadata", {})
            metadata = metadata if isinstance(metadata, Mapping) else {}
            lines.extend(
                (
                    "",
                    f"Selected: {selected_job.get('job_id')}  "
                    f"scheduler={selected_job.get('scheduler_id') or 'not acknowledged'}  "
                    f"phase={metadata.get('submission_phase', '-')}",
                )
            )
        if log_text:
            lines.append("Logs (last 12 lines):")
            lines.extend(log_text.splitlines()[-12:])
        lines.extend(("", message or "↑/↓ or j/k select · l logs · x cancel · r refresh · q quit"))
        return "\n".join(line[:width] for line in lines)

    @classmethod
    def _bar(cls, value: object) -> str:
        if not isinstance(value, (int, float)):
            return "unknown"
        bounded = max(0.0, min(100.0, float(value)))
        cells = round(bounded / 10)
        return f"{'█' * cells}{'░' * (10 - cells)} {bounded:3.0f}%"


class _TerminalSession(AbstractContextManager["_TerminalSession"]):
    """Restore terminal flags and screen even when monitoring is interrupted."""

    def __init__(self, stream: TextIO = sys.stdout) -> None:
        self.stream = stream
        self.fd = sys.stdin.fileno()
        self.previous: list[Any] | None = None
        self.termios: Any = None

    def __enter__(self) -> _TerminalSession:
        import termios
        import tty

        self.termios = termios
        self.previous = termios.tcgetattr(self.fd)
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
        ready, _, _ = select.select([self.fd], [], [], timeout)
        if not ready:
            return None
        value = sys.stdin.read(1)
        if value == "\x1b" and select.select([self.fd], [], [], 0.01)[0]:
            value += sys.stdin.read(2)
        return value


class LiveJobMonitor:
    """Coordinate snapshots and explicit job actions without owning control-plane state."""

    def __init__(
        self,
        overview: OverviewService,
        jobs: JobService,
        *,
        interval: float = 2.0,
        stream: TextIO = sys.stdout,
    ) -> None:
        if interval < 0.2:
            raise ValueError("Live monitor interval must be at least 0.2 seconds.")
        self.overview = overview
        self.jobs = jobs
        self.interval = interval
        self.stream = stream

    def run(self) -> int:
        """Refresh until q, supporting selection, logs and confirmed cancellation."""
        selected = 0
        logs = ""
        message = ""
        payload: Mapping[str, Any] = {}
        refresh = True
        with _TerminalSession(self.stream) as terminal:
            while True:
                if refresh:
                    payload = self.overview.snapshot()
                    refresh = False
                items = payload.get("jobs", {}).get("items", [])
                selected = min(selected, max(0, len(items) - 1))
                width = shutil.get_terminal_size((120, 30)).columns
                rendered = MonitorRenderer.render(
                    payload,
                    selected=selected,
                    log_text=logs,
                    message=message,
                    width=width,
                )
                self.stream.write("\x1b[H\x1b[2J" + rendered)
                self.stream.flush()
                started = time.monotonic()
                key = terminal.key(self.interval)
                if key in {"q", "Q"}:
                    return 0
                if key in {"j", "\x1b[B"}:
                    selected = min(max(0, len(items) - 1), selected + 1)
                elif key in {"k", "\x1b[A"}:
                    selected = max(0, selected - 1)
                elif key == "r" or key is None:
                    refresh = True
                elif key == "l" and items:
                    logs = self.jobs.logs(str(items[selected]["job_id"]), tail=12)
                    message = ""
                elif key == "x" and items:
                    job_id = str(items[selected]["job_id"])
                    message = f"Press y to cancel {job_id}; any other key aborts."
                    self.stream.write("\x1b[H\x1b[2J" + MonitorRenderer.render(
                        payload, selected=selected, log_text=logs, message=message, width=width
                    ))
                    self.stream.flush()
                    if terminal.key(10.0) == "y":
                        record = self.jobs.cancel(job_id)
                        message = f"{record.job_id}: {record.state.value}"
                        refresh = True
                    else:
                        message = "Cancellation aborted."
                if time.monotonic() - started >= self.interval:
                    refresh = True
