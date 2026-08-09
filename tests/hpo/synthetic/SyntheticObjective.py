"""Deterministic multi-fidelity objective."""

from __future__ import annotations

import math


class SyntheticObjective:
    """Expose a known optimum with fidelity-correlated transient bias."""

    def __init__(self, *, optimum: float = 0.65, transient: float = 0.2, rate: float = 0.3) -> None:
        self.optimum = float(optimum)
        self.transient = float(transient)
        self.rate = float(rate)

    def evaluate(self, value: float, budget: int) -> float:
        """Return ``-(x-x*)² + A(x) exp(-k b)``."""
        amplitude = self.transient * (0.5 - value)
        return -((value - self.optimum) ** 2) + amplitude * math.exp(-self.rate * budget)
