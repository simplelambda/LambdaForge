"""Implementation of the CELU object."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from lambdaforge.nn.activations.Activation import Activation


class CELU(Activation):
    """Continuously Differentiable Exponential Linear Unit.

    Parameters
    ----------
    alpha : float
        Positive scale of the negative branch. Default: ``1.0``.
    inplace : bool
        Whether to modify the input tensor in place. Default: ``False``.
    name : str | None
        Optional name used to identify this activation instance.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        inplace: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be a real number")
        if not math.isfinite(float(alpha)) or float(alpha) <= 0.0:
            raise ValueError("alpha must be finite and greater than zero")
        if not isinstance(inplace, bool):
            raise TypeError("inplace must be a boolean")
        self.alpha = float(alpha)
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.celu(x, alpha=self.alpha, inplace=self.inplace)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, inplace={self.inplace}, name={self.name!r}"
