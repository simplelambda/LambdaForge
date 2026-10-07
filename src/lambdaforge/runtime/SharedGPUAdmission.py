"""Short host-wide check/start coordination, never a whole-GPU reservation."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.work.atomic import atomic_write_text

T = TypeVar("T")


def shared_gpu_launch(
    root: Path,
    token: str,
    launch: Callable[[], T | None],
    *,
    stagger_seconds: float,
) -> T | None:
    """Serialize a fresh physical check and launch; contention leaves the Run queued.

    All projects on a direct shared host use the same lease root and physical device token.
    Other GPUs remain independent. The OS releases the lock on death; the bounded last-start
    timestamp prevents sibling controllers racing while a new worker initializes CUDA.
    """
    key = hashlib.sha256(token.encode("utf-8")).hexdigest()
    lock = CrossProcessFileLock(
        root / f"admission-{key}.lock",
        shared=False,
        timeout_seconds=0.01,
        poll_interval_seconds=0.005,
    )
    try:
        lock.acquire()
    except TimeoutError:
        return None
    try:
        stamp = root / f"admission-{key}.json"
        if stamp.is_symlink():
            raise ValueError("Shared GPU admission metadata must not be a symlink.")
        if stamp.exists():
            previous = float(json.loads(stamp.read_text(encoding="utf-8"))["started_monotonic"])
            elapsed = time.monotonic() - previous
            if 0 <= elapsed < stagger_seconds:
                return None
        result = launch()
        if result is not None:
            atomic_write_text(
                stamp,
                json.dumps({"token": token, "started_monotonic": time.monotonic()}),
            )
        return result
    finally:
        lock.release()
