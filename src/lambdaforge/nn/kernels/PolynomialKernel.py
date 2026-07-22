"""Polynomial kernel."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from lambdaforge.nn.kernels.Kernel import Kernel


class PolynomialKernel(Kernel):
    r"""Compute ``(gamma * x @ y.T + offset) ** degree``.

    Gamma can be learned through a positive softplus parameterization. Offset
    is unconstrained when learned because negative offsets are valid for some
    experimental formulations.
    """

    def __init__(
        self,
        degree: int = 2,
        gamma: float = 1.0,
        offset: float = 1.0,
        learnable_gamma: bool = False,
        learnable_offset: bool = False,
        min_gamma: float = 1e-8,
        name: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(name=name)
        if isinstance(degree, bool) or not isinstance(degree, int):
            raise TypeError("degree must be an integer.")
        if degree < 1:
            raise ValueError("degree must be >= 1.")
        for label, value in {"gamma": gamma, "offset": offset, "min_gamma": min_gamma}.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{label} must be a real number.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{label} must be finite.")
        if gamma <= 0:
            raise ValueError("gamma must be positive.")
        if min_gamma <= 0 or min_gamma >= gamma:
            raise ValueError("min_gamma must be positive and smaller than gamma.")
        if not isinstance(learnable_gamma, bool) or not isinstance(learnable_offset, bool):
            raise TypeError("learnable_gamma and learnable_offset must be booleans.")
        initial_gamma = float(gamma - min_gamma)
        raw_gamma_value = (
            initial_gamma if initial_gamma > 20.0 else math.log(math.expm1(initial_gamma))
        )
        raw_gamma = torch.tensor(raw_gamma_value, device=device, dtype=dtype)
        offset_tensor = torch.tensor(float(offset), device=device, dtype=dtype)
        if learnable_gamma:
            self.raw_gamma = nn.Parameter(raw_gamma)
        else:
            self.register_buffer("raw_gamma", raw_gamma)
        if learnable_offset:
            self.offset = nn.Parameter(offset_tensor)
        else:
            self.register_buffer("offset", offset_tensor)
        self.degree = int(degree)
        self.min_gamma = float(min_gamma)

    @property
    def gamma(self) -> torch.Tensor:
        """Return the positive effective input scale."""
        return F.softplus(self.raw_gamma) + self.min_gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.validate_inputs(x, y)
        if x.device != self.raw_gamma.device or x.dtype != self.raw_gamma.dtype:
            raise ValueError("Inputs and PolynomialKernel parameters must share device and dtype.")
        dot_product = torch.matmul(x, y.transpose(-1, -2))
        return (self.gamma * dot_product + self.offset).pow(self.degree)
