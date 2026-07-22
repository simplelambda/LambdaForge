"""Implementation of the Sine object."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from lambdaforge.nn.activations.Activation import Activation


class Sine(Activation):
    """Configurable sinusoidal activation for periodic representations.

    Computes ``amplitude * sin(frequency * x + phase)``.

    Parameters
    ----------
    frequency : float
        Angular frequency applied to the input. Default: ``1.0``.
    amplitude : float
        Output amplitude. Default: ``1.0``.
    phase : float
        Phase offset in radians. Default: ``0.0``.
    name : str | None
        Optional name used to identify this activation instance.
    """

    def __init__(
        self,
        frequency: float = 1.0,
        amplitude: float = 1.0,
        phase: float = 0.0,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        values = {"frequency": frequency, "amplitude": amplitude, "phase": phase}
        for label, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{label} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{label} must be finite")
        self.frequency = float(frequency)
        self.amplitude = float(amplitude)
        self.phase = float(phase)

    def forward(self, x: Tensor) -> Tensor:
        return self.amplitude * torch.sin(self.frequency * x + self.phase)

    def extra_repr(self) -> str:
        return (
            f"frequency={self.frequency}, amplitude={self.amplitude}, "
            f"phase={self.phase}, name={self.name!r}"
        )
