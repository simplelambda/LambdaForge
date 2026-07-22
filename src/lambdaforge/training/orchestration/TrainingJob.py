"""Serializable description of one orchestrated training job."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from lambdaforge.training.orchestration.DeviceAssignment import DeviceAssignment


@dataclass(frozen=True, slots=True, init=False)
class TrainingJob:
    """Name, callable and optional logical GPUs for one subprocess.

    ``run`` is called as ``run(stop_event)`` in a dedicated process. Device
    indices are logical positions within the parent ``CUDA_VISIBLE_DEVICES``.
    A ``None`` assignment inherits the complete parent-visible device set.
    An empty sequence is frozen as ``()`` and explicitly hides CUDA from the
    worker, providing an unambiguous CPU-only assignment.
    """

    name: str
    run: Callable[[Any], None]
    devices: tuple[int, ...] | None

    def __init__(
        self,
        name: str,
        run: Callable[[Any], None],
        devices: Sequence[int] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("TrainingJob.name cannot be empty.")
        if not callable(run):
            raise TypeError("TrainingJob.run must be callable.")
        normalized_devices = DeviceAssignment.normalize(devices, label="TrainingJob.devices")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "run", run)
        object.__setattr__(self, "devices", normalized_devices)
