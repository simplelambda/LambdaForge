"""Spatial membership evidence, not a probability of a functional event."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class StableRegion:
    """Uniform-threshold membership on (lower, upper], bounded by branch lifetime."""

    branch_id: int
    interval: tuple[float, float]
    core: np.ndarray
    support: np.ndarray
    membership: np.ndarray
    mass: float
    support_mass: float
    persistence: float | None
    stability: float
    n_thresholds: int

    def __post_init__(self) -> None:
        for name in ("core", "support", "membership"):
            value = getattr(self, name).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
