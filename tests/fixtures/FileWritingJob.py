"""Pickle-safe test job used by spawn-based orchestration tests."""

from pathlib import Path
from typing import Any


class FileWritingJob:
    """Write a marker file when invoked by a training subprocess."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __call__(self, stop_event: Any) -> None:
        if not stop_event.is_set():
            self.path.write_text("ok", encoding="utf-8")
