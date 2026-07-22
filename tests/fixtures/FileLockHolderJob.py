"""Spawn-safe process job that holds or abandons a cross-process file lock."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from lambdaforge.runtime import CrossProcessFileLock


class FileLockHolderJob:
    """Acquire one lock, signal ownership and optionally terminate abruptly."""

    def __init__(
        self,
        path: str | Path,
        *,
        shared: bool,
        acquired_event: Any,
        release_event: Any,
        crash_exit_code: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.shared = shared
        self.acquired_event = acquired_event
        self.release_event = release_event
        self.crash_exit_code = crash_exit_code

    def __call__(self) -> None:
        with CrossProcessFileLock(
            self.path,
            shared=self.shared,
            timeout_seconds=10.0,
            poll_interval_seconds=0.01,
        ):
            self.acquired_event.set()
            if self.crash_exit_code is not None:
                os._exit(self.crash_exit_code)
            if not self.release_event.wait(10.0):
                raise TimeoutError("Timed out waiting to release the test file lock.")
