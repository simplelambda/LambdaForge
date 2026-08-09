"""Structured append-only event logging."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock


class EventLogger:
    """Append bounded JSONL events safely across local processes."""

    def __init__(self, path: str | Path, *, max_event_bytes: int = 65536) -> None:
        self.path = Path(path)
        self.max_event_bytes = max_event_bytes

    def write(self, event: str, fields: Mapping[str, Any] | None = None) -> None:
        """Append one timestamped JSON object after size/serialization validation."""
        if not event:
            raise ValueError("Event names cannot be empty.")
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **dict(fields or {}),
        }
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        if len(encoded.encode()) > self.max_event_bytes:
            raise ValueError("Structured event exceeds max_event_bytes.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with CrossProcessFileLock(
            self.path.with_suffix(self.path.suffix + ".lock"),
            shared=False,
            timeout_seconds=30,
            poll_interval_seconds=0.05,
        ):
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
