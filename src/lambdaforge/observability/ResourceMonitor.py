"""Bounded-overhead process and accelerator sampling."""

from __future__ import annotations

import os
import time
from typing import Any

import psutil
import torch


class ResourceMonitor:
    """Sample current-process CPU/RAM and optional CUDA memory on demand."""

    def __init__(self, *, min_interval_seconds: float = 1.0) -> None:
        if min_interval_seconds <= 0:
            raise ValueError("Resource monitor interval must be positive.")
        self.min_interval_seconds = min_interval_seconds
        self._last_sample = 0.0
        self._process = psutil.Process(os.getpid())

    def sample(
        self, *, processed_items: int | None = None, elapsed_seconds: float | None = None
    ) -> dict[str, Any] | None:
        """Return a sample at the configured maximum frequency, otherwise null."""
        now = time.monotonic()
        if now - self._last_sample < self.min_interval_seconds:
            return None
        self._last_sample = now
        memory = self._process.memory_info()
        payload: dict[str, Any] = {
            "cpu_percent": self._process.cpu_percent(None),
            "rss_bytes": memory.rss,
            "threads": self._process.num_threads(),
        }
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            payload.update(
                {
                    "cuda_device": device,
                    "cuda_allocated_bytes": torch.cuda.memory_allocated(device),
                    "cuda_reserved_bytes": torch.cuda.memory_reserved(device),
                    "cuda_max_allocated_bytes": torch.cuda.max_memory_allocated(device),
                }
            )
        if processed_items is not None and elapsed_seconds is not None and elapsed_seconds > 0:
            payload["items_per_second"] = processed_items / elapsed_seconds
        return payload
