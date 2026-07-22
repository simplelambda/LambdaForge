"""Radial-basis-function kernel."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from lambdaforge.nn.kernels.Kernel import Kernel


class RBFKernel(Kernel):
    r"""Gaussian RBF kernel ``exp(-||x-y||^2 / (2 l^2))``.

    ``length_scale`` may be fixed or learned. A softplus parameterization keeps
    a learned scale strictly above ``min_length_scale``.
    """

    def __init__(
        self,
        length_scale: float = 1.0,
        learnable: bool = False,
        min_length_scale: float = 1e-6,
        name: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(name=name)
        for label, value in {
            "length_scale": length_scale,
            "min_length_scale": min_length_scale,
        }.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{label} must be a real number.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{label} must be finite.")
        if length_scale <= 0:
            raise ValueError("length_scale must be positive.")
        if min_length_scale <= 0 or min_length_scale >= length_scale:
            raise ValueError("min_length_scale must be positive and smaller than length_scale.")
        if not isinstance(learnable, bool):
            raise TypeError("learnable must be a boolean.")
        initial = float(length_scale - min_length_scale)
        raw_value = initial if initial > 20.0 else math.log(math.expm1(initial))
        raw = torch.tensor(raw_value, device=device, dtype=dtype)
        if learnable:
            self.raw_length_scale = nn.Parameter(raw)
        else:
            self.register_buffer("raw_length_scale", raw)
        self.min_length_scale = float(min_length_scale)

    @property
    def length_scale(self) -> torch.Tensor:
        """Return the positive effective length scale."""
        return F.softplus(self.raw_length_scale) + self.min_length_scale

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.validate_inputs(x, y)
        if x.device != self.raw_length_scale.device or x.dtype != self.raw_length_scale.dtype:
            raise ValueError("Inputs and RBFKernel scale must share device and dtype.")
        squared_distance = torch.cdist(x, y, p=2.0).square()
        return torch.exp(-squared_distance / (2.0 * self.length_scale.square()))
