"""Implementation of the SquarePlus object."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from lambdaforge.nn.activations.Activation import Activation


class SquarePlus(Activation):
    r"""Algebraic smooth approximation of ReLU.

    Computes ``0.5 * (x + sqrt(x^2 + b))``.

    Parameters
    ----------
    b : float
        Positive smoothness constant. Default: ``4.0``.
    name : str | None
        Optional name used to identify this activation instance.
    """

    def __init__(self, b: float = 4.0, name: str | None = None) -> None:
        super().__init__(name=name)
        if isinstance(b, bool) or not isinstance(b, (int, float)):
            raise TypeError("b must be a real number")
        if not math.isfinite(float(b)) or float(b) <= 0.0:
            raise ValueError("b must be finite and greater than zero")
        self.b = float(b)

    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * (x + torch.sqrt(x.square() + self.b))

    def extra_repr(self) -> str:
        return f"b={self.b}, name={self.name!r}"
