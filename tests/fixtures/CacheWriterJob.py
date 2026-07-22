"""Spawn-safe concurrent writer used by persistent-cache tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lambdaforge.data import DiskCacheBackend


class CacheWriterJob:
    """Wait on a shared gate, write one key and report the resulting usage."""

    def __init__(
        self,
        root: str | Path,
        namespace: str,
        max_bytes: int,
        max_entries: int,
        key: str,
        payload: bytes,
        ready_queue: Any,
        start_event: Any,
        result_queue: Any,
    ) -> None:
        self.root = Path(root)
        self.namespace = namespace
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.key = key
        self.payload = payload
        self.ready_queue = ready_queue
        self.start_event = start_event
        self.result_queue = result_queue

    def __call__(self) -> None:
        backend = DiskCacheBackend(
            self.root,
            self.namespace,
            max_bytes=self.max_bytes,
            max_entries=self.max_entries,
        )
        self.ready_queue.put(True)
        if not self.start_event.wait(10):
            raise TimeoutError("Concurrent cache writer start gate was not released.")
        written = backend.write(self.key, self.payload)
        usage = backend.usage()
        self.result_queue.put((written, usage.entries, usage.bytes))
