"""Spawn-safe writer used to verify mmap lease coordination."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lambdaforge.data import DiskCacheBackend


class CacheBlockedWriterJob:
    """Signal before opening a backend whose exclusive recovery lock may block."""

    def __init__(
        self,
        root: str | Path,
        namespace: str,
        max_bytes: int,
        max_entries: int,
        key: str,
        payload: bytes,
        started_event: Any,
        done_event: Any,
    ) -> None:
        self.root = Path(root)
        self.namespace = namespace
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.key = key
        self.payload = payload
        self.started_event = started_event
        self.done_event = done_event

    def __call__(self) -> None:
        self.started_event.set()
        backend = DiskCacheBackend(
            self.root,
            self.namespace,
            max_bytes=self.max_bytes,
            max_entries=self.max_entries,
            lock_timeout_seconds=10.0,
        )
        backend.write(self.key, self.payload)
        self.done_event.set()
