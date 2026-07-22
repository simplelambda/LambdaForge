"""Spawn-safe orchestrator job with one real nested descendant."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any

import psutil

from lambdaforge.training.orchestration.ProcessGuard import ProcessGuard
from tests.fixtures.LongRunningDescendant import LongRunningDescendant


class LongRunningProcessJob:
    """Run until stopped while exposing stable worker and descendant identities."""

    def __init__(
        self,
        state_dir: str | Path,
        *,
        cooperative: bool,
        poll_seconds: float = 0.05,
        startup_timeout_seconds: float = 10.0,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.cooperative = bool(cooperative)
        self.poll_seconds = float(poll_seconds)
        self.startup_timeout_seconds = float(startup_timeout_seconds)

    def __call__(self, stop_event: Any) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._write_identity(self.state_dir / "worker.json", "worker")
        context = mp.get_context("spawn")
        descendant = context.Process(
            target=LongRunningDescendant(
                self.state_dir / "descendant.json",
                poll_seconds=self.poll_seconds,
            ),
            name="lambdaforge-process-test-descendant",
        )
        descendant.start()
        try:
            self._wait_for_descendant(descendant)
            (self.state_dir / "ready").write_text("ready\n", encoding="utf-8")
            while True:
                if self.cooperative and stop_event.wait(self.poll_seconds):
                    return
                time.sleep(self.poll_seconds)
        finally:
            self._stop_descendant(descendant)

    def _wait_for_descendant(self, descendant: BaseProcess) -> None:
        identity_path = self.state_dir / "descendant.json"
        deadline = time.monotonic() + self.startup_timeout_seconds
        while time.monotonic() < deadline:
            if identity_path.exists():
                return
            if not descendant.is_alive():
                raise RuntimeError(
                    "Nested lifecycle-test descendant exited before publishing its identity."
                )
            time.sleep(self.poll_seconds)
        raise TimeoutError("Nested lifecycle-test descendant did not become ready in time.")

    def _stop_descendant(self, descendant: BaseProcess) -> None:
        ProcessGuard().terminate_process_tree(
            descendant.pid,
            grace_seconds=0.2,
            include_parent=True,
        )
        descendant.join(timeout=1.0)
        if descendant.is_alive():
            kill = getattr(descendant, "kill", None)
            if kill is not None:
                kill()
            else:
                descendant.terminate()
            descendant.join(timeout=1.0)

    @staticmethod
    def _write_identity(path: Path, role: str) -> None:
        payload = {
            "role": role,
            "pid": os.getpid(),
            "create_time": psutil.Process(os.getpid()).create_time(),
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
