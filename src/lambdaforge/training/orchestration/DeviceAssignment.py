"""Validated immutable device assignment for one training process."""

from __future__ import annotations

from collections.abc import Sequence


class DeviceAssignment:
    """Normalize device sequences into immutable, unambiguous assignments.

    None inherits visible CUDA devices; an empty tuple explicitly hides CUDA.
    """

    @staticmethod
    def normalize(
        devices: Sequence[int] | None,
        *,
        label: str,
    ) -> tuple[int, ...] | None:
        if devices is None:
            return None
        if isinstance(devices, (str, bytes, bytearray)) or not isinstance(
            devices,
            Sequence,
        ):
            raise TypeError(f"{label} must be a sequence of integer indices.")
        normalized = tuple(devices)
        if any(not isinstance(device, int) or isinstance(device, bool) for device in normalized):
            raise TypeError(f"{label} indices must be integers, not booleans.")
        if any(device < 0 for device in normalized):
            raise ValueError(f"{label} indices must be non-negative.")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{label} indices cannot contain duplicates.")
        return normalized
