"""Pickle-safe synthetic job with a controlled duration."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class TimedWritingJob:
    """Wait for a short deterministic duration and publish one marker."""

    def __init__(self, path: Path, seconds: float) -> None:
        self.path = path
        self.seconds = float(seconds)

    def __call__(self, stop_event: Any) -> None:
        """Write only if the synthetic job was not cancelled."""
        if stop_event.wait(self.seconds):
            return
        self.path.write_text("done", encoding="utf-8")
