"""Spawn-safe descendant used by real process-lifecycle integration tests."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import psutil

from lambdaforge.training.orchestration.ProcessGuard import ProcessGuard


class LongRunningDescendant:
    """Remain alive until the worker or its process guard terminates this process."""

    def __init__(self, identity_path: str | Path, poll_seconds: float = 0.05) -> None:
        self.identity_path = Path(identity_path)
        self.poll_seconds = float(poll_seconds)

    def __call__(self) -> None:
        guard = ProcessGuard()
        guard.install_parent_death_guard(poll_seconds=self.poll_seconds)
        guard.install_child_process_cleanup(grace_seconds=0.2)
        self._write_identity()
        while True:
            time.sleep(self.poll_seconds)

    def _write_identity(self) -> None:
        self.identity_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "role": "descendant",
            "pid": os.getpid(),
            "create_time": psutil.Process(os.getpid()).create_time(),
        }
        temporary = self.identity_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        temporary.replace(self.identity_path)
