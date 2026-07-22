"""Implementation of the Softplus object."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from lambdaforge.nn.activations.Activation import Activation


class Softplus(Activation):
    """Smooth approximation of ReLU with configurable sharpness.

    Parameters
    ----------
    beta : float
        Positive sharpness coefficient. Default: ``1.0``.
    threshold : float
        Input threshold above which the linear approximation is used for
        numerical stability. Default: ``20.0``.
    name : str | None
        Optional name used to identify this activation instance.
    """

    def __init__(
        self,
        beta: float = 1.0,
        threshold: float = 20.0,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if isinstance(beta, bool) or not isinstance(beta, (int, float)):
            raise TypeError("beta must be a real number")
        if not math.isfinite(float(beta)) or float(beta) <= 0.0:
            raise ValueError("beta must be finite and greater than zero")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be a real number")
        if not math.isfinite(float(threshold)):
            raise ValueError("threshold must be finite")
        self.beta = float(beta)
        self.threshold = float(threshold)

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.softplus(x, beta=self.beta, threshold=self.threshold)

    def extra_repr(self) -> str:
        return f"beta={self.beta}, threshold={self.threshold}, name={self.name!r}"
