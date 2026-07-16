"""Context object that captures standard output and error in a run log."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import IO, Any

from lambdaforge.experiments.TeeStream import TeeStream


class StdIOCapture:
    """Append stdout/stderr to a UTF-8 log, optionally echoing to the console."""

    def __init__(self, log_path: str | Path, echo: bool = True) -> None:
        self.log_path = Path(log_path)
        self.echo = echo
        self._handle: IO[str] | None = None
        self._stdout: Any = None
        self._stderr: Any = None

    def __enter__(self) -> StdIOCapture:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stdout, self._stderr = sys.stdout, sys.stderr
        self._handle = self.log_path.open("a", encoding="utf-8", buffering=1)
        sys.stdout = TeeStream(self._handle, self._stdout if self.echo else None)
        sys.stderr = TeeStream(self._handle, self._stderr if self.echo else None)
        return self

    def __exit__(self, *_: Any) -> None:
        sys.stdout, sys.stderr = self._stdout, self._stderr
        if self._handle is not None:
            self._handle.close()
