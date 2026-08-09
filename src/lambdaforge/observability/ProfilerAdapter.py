"""Profiler provider boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any


class ProfilerAdapter(ABC):
    """Create a bounded profiling context without coupling runners to providers."""

    @abstractmethod
    def profile(self, output_dir: str | Path) -> AbstractContextManager[Any]:
        """Return a context manager that owns profiler lifecycle."""
