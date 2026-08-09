"""Structured append-only adaptive optimization events."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock


class AdaptiveEventLog:
    """Append fsynced JSONL controller events under a cross-process lock."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.lock = CrossProcessFileLock(
            self.path.with_suffix(self.path.suffix + ".lock"),
            shared=False,
            timeout_seconds=30.0,
            poll_interval_seconds=0.05,
        )

    def append(self, event: str, payload: Mapping[str, Any] | None = None) -> None:
        """Persist one timestamped structured event."""
        record = {
            "event": str(event),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            **dict(payload or {}),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            with open(self.path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
