"""Persistent metadata for an externally detached experiment process."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lambdaforge.experiments.DetachedStatus import DetachedStatus
from lambdaforge.experiments.ExperimentConfig import ExperimentConfig


class DetachedRunState:
    """Read and atomically update PID metadata for a detached suite launcher."""

    def __init__(
        self,
        config_path: str | Path,
        pid_path: str | Path | None = None,
        state_path: str | Path | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        config = ExperimentConfig.from_yaml(self.config_path)
        self.pid_path = (
            Path(pid_path) if pid_path is not None else config.suite_dir / "experiment.pid"
        )
        self.state_path = (
            Path(state_path)
            if state_path is not None
            else config.suite_dir / "experiment.detached.json"
        )

    def write(
        self,
        *,
        pid: int | None = None,
        pgid: int | None = None,
        log_path: str | Path | None = None,
        status: DetachedStatus | str = DetachedStatus.RUNNING,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist state and update the shell-friendly plain PID file."""
        resolved_status = DetachedStatus(status)
        self.pid_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        previous = self.read()
        now = datetime.now(timezone.utc).isoformat()
        payload: dict[str, Any] = {
            "status": resolved_status.value,
            "pid": int(pid) if pid is not None else os.getpid(),
            "pgid": int(pgid) if pgid is not None else None,
            "config": str(self.config_path),
            "log_path": str(log_path) if log_path is not None else previous.get("log_path"),
            "pid_file": str(self.pid_path),
            "state_file": str(self.state_path),
            "launched_at": previous.get("launched_at", now),
            "updated_at": now,
        }
        if resolved_status in {DetachedStatus.FINISHED, DetachedStatus.FAILED}:
            payload["finished_at"] = now
        if extra:
            payload.update(dict(extra))
        self._write_atomic(self.pid_path, f"{payload['pid']}\n")
        self._write_atomic(self.state_path, json.dumps(payload, indent=2) + "\n")
        return payload

    def read(self) -> dict[str, Any]:
        """Return the current JSON state, or an empty mapping if unavailable."""
        if not self.state_path.exists():
            return {}
        try:
            with self.state_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def is_alive(self) -> bool:
        """Check whether the recorded launcher PID still exists."""
        state = self.read()
        if state.get("status") not in {
            DetachedStatus.LAUNCHED.value,
            DetachedStatus.RUNNING.value,
        }:
            return False
        pid = state.get("pid")
        if pid in (None, ""):
            return False
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    @staticmethod
    def _write_atomic(path: Path, text: str) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
        temporary.replace(path)
