"""Serializable description of one orchestrated training job."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TrainingJob:
    """Name, callable and optional logical GPUs for one subprocess.

    ``run`` is called as ``run(stop_event)`` in a dedicated process. Device
    indices are logical positions within the parent ``CUDA_VISIBLE_DEVICES``.
    A ``None`` assignment inherits the complete parent-visible device set.
    """

    name: str
    run: Callable[[Any], None]
    devices: list[int] | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("TrainingJob.name cannot be empty.")
        if self.devices is not None and any(device < 0 for device in self.devices):
            raise ValueError("TrainingJob device indices must be non-negative.")
