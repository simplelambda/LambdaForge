"""Bounded external scientific-process execution for Work."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Thread
from typing import Any


@dataclass(frozen=True, slots=True)
class Tool(os.PathLike[str]):
    """One resolved external executable and its optional version evidence."""

    name: str
    path: Path
    version: str | None = None

    def __fspath__(self) -> str:
        return str(self.path)

    def __str__(self) -> str:
        return str(self.path)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Immutable result of one completed external command."""

    name: str
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float


class ToolExecutionError(RuntimeError):
    """An external command failed after LambdaForge captured bounded diagnostics."""

    def __init__(self, result: ToolResult) -> None:
        self.result = result
        diagnostic = (result.stderr or result.stdout or "no output")[-8000:]
        super().__init__(
            f"External tool {result.name!r} exited with code {result.returncode}.\n{diagnostic}"
        )


class ToolService:
    """Resolve tools, run them without a shell and record used-tool provenance."""

    THREAD_VARIABLES = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )

    def __init__(self, log: Any) -> None:
        self._log = log
        self._tools: dict[str, Tool] = {}
        self._lock = Lock()
        environment_bin = str(Path(sys.executable).resolve().parent)
        inherited = os.environ.get("PATH", "")
        self._search_path = os.pathsep.join(
            value for value in (environment_bin, inherited) if value
        )

    def require(
        self,
        executable: str,
        *,
        version_args: Sequence[str] | None = None,
        version_timeout: float = 10.0,
    ) -> Tool:
        """Resolve an executable and optionally capture an explicit version probe."""
        selected = str(executable).strip()
        if not selected:
            raise ValueError("Executable names cannot be empty.")
        resolved = shutil.which(selected, path=self._search_path)
        if resolved is None:
            raise FileNotFoundError(
                f"Required external tool {selected!r} was not found on PATH. "
                "Install it in the selected LambdaForge environment or configure PATH explicitly."
            )
        path = Path(resolved).resolve()
        version: str | None = None
        if version_args is not None:
            try:
                completed = subprocess.run(
                    [str(path), *(str(value) for value in version_args)],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=version_timeout,
                    shell=False,
                )
            except (OSError, subprocess.SubprocessError) as error:
                raise RuntimeError(
                    f"Version probe for external tool {selected!r} failed: {error}"
                ) from error
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "no output")[-2000:]
                raise RuntimeError(
                    f"Version probe for {selected!r} exited with {completed.returncode}: {detail}"
                )
            version = (completed.stdout or completed.stderr).strip()[:2000] or None
        tool = Tool(selected, path, version)
        with self._lock:
            existing = self._tools.get(str(path))
            self._tools[str(path)] = (
                existing if existing and existing.version and not version else tool
            )
        return tool

    def run(
        self,
        command: Sequence[str | os.PathLike[str]],
        *,
        name: str | None = None,
        threads: int | None = None,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        check: bool = True,
    ) -> ToolResult:
        """Run one argv command with scoped environment and captured Work logs."""
        if not command:
            raise ValueError("External command cannot be empty.")
        if threads is not None and (
            isinstance(threads, bool) or not isinstance(threads, int) or threads < 1
        ):
            raise ValueError("threads must be an integer >= 1 or null.")
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, int | float)
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive finite number or null.")
        arguments = tuple(os.fspath(value) for value in command)
        executable = shutil.which(arguments[0], path=self._search_path)
        if executable is None:
            raise FileNotFoundError(f"External executable {arguments[0]!r} was not found on PATH.")
        arguments = (str(Path(executable).resolve()), *arguments[1:])
        selected_name = str(name or Path(arguments[0]).name)
        environment = dict(os.environ)
        environment["PATH"] = self._search_path
        if threads is not None:
            environment.update({key: str(threads) for key in self.THREAD_VARIABLES})
        if env is not None:
            reserved = sorted(
                str(key) for key in env if str(key).upper().startswith("LAMBDAFORGE_")
            )
            if reserved:
                raise ValueError(
                    "External tool environment cannot override framework-owned variables: "
                    + ", ".join(reserved)
                )
            environment.update({str(key): str(value) for key, value in env.items()})
        self._log.emit(f"Starting external tool {selected_name}: {arguments!r}")
        started = time.perf_counter()
        try:
            process = subprocess.Popen(
                arguments,
                cwd=Path(cwd) if cwd is not None else None,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                shell=False,
            )
        except OSError as error:
            raise RuntimeError(
                f"Could not start external tool {selected_name!r}: {error}"
            ) from error
        stdout = _BoundedCapture()
        stderr = _BoundedCapture()
        pumps = (
            Thread(
                target=self._pump,
                args=(process.stdout, stdout, "info"),
                daemon=True,
            ),
            Thread(
                target=self._pump,
                args=(process.stderr, stderr, "warning"),
                daemon=True,
            ),
        )
        for pump in pumps:
            pump.start()
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.wait()
            for pump in pumps:
                pump.join(timeout=2)
            raise RuntimeError(
                f"External tool {selected_name!r} exceeded its {timeout}-second timeout."
            ) from error
        for pump in pumps:
            pump.join()
        duration = time.perf_counter() - started
        result = ToolResult(
            selected_name,
            arguments,
            returncode,
            stdout.text,
            stderr.text,
            duration,
        )
        with self._lock:
            if arguments[0] not in self._tools:
                self._tools[arguments[0]] = Tool(Path(arguments[0]).name, Path(arguments[0]))
        if check and returncode != 0:
            raise ToolExecutionError(result)
        return result

    def _pump(self, stream: Any, capture: _BoundedCapture, level: str) -> None:
        if stream is None:
            return
        try:
            for line in stream:
                capture.append(line)
                self._log.emit(line.rstrip("\r\n"), level=level)
        finally:
            stream.close()

    @property
    def provenance(self) -> tuple[dict[str, str | None], ...]:
        """Return deterministic evidence for tools actually required or executed."""
        with self._lock:
            return tuple(
                {
                    "name": tool.name,
                    "path": str(tool.path),
                    "version": tool.version,
                }
                for tool in sorted(self._tools.values(), key=lambda value: str(value.path))
            )


__all__ = ["Tool", "ToolExecutionError", "ToolResult", "ToolService"]


class _BoundedCapture:
    """Retain only the most recent bounded process output while logs receive every line."""

    LIMIT = 1024 * 1024

    def __init__(self) -> None:
        self._parts: deque[str] = deque()
        self._size = 0
        self._lock = Lock()

    def append(self, value: str) -> None:
        selected = value[-self.LIMIT :]
        with self._lock:
            self._parts.append(selected)
            self._size += len(selected)
            while self._size > self.LIMIT and self._parts:
                removed = self._parts.popleft()
                self._size -= len(removed)

    @property
    def text(self) -> str:
        with self._lock:
            return "".join(self._parts)[-self.LIMIT :]
