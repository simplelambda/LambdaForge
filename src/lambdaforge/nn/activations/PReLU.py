"""Implementation of the PReLU object."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from lambdaforge.nn.activations.Activation import Activation


class PReLU(Activation):
    """Parametric Rectified Linear Unit with learnable negative slopes.

    Parameters
    ----------
    num_parameters : int
        Number of learned slopes. Use ``1`` to share one slope or the number
        of channels to learn one slope per channel. Default: ``1``.
    init : float
        Initial value of every negative slope. Default: ``0.25``.
    name : str | None
        Optional name used to identify this activation instance.
    """

    def __init__(
        self,
        num_parameters: int = 1,
        init: float = 0.25,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if isinstance(num_parameters, bool) or not isinstance(num_parameters, int):
            raise TypeError("num_parameters must be an integer")
        if num_parameters <= 0:
            raise ValueError("num_parameters must be greater than zero")
        if isinstance(init, bool) or not isinstance(init, (int, float)):
            raise TypeError("init must be a real number")
        if not math.isfinite(float(init)):
            raise ValueError("init must be finite")

        self.num_parameters = num_parameters
        self.init = float(init)
        self.weight = nn.Parameter(torch.full((num_parameters,), self.init))

    def forward(self, x: Tensor) -> Tensor:
        if self.num_parameters > 1 and (x.ndim < 2 or x.shape[1] != self.num_parameters):
            raise ValueError(
                "PReLU with multiple parameters expects input channel dimension 1 "
                f"to have size {self.num_parameters}"
            )
        return torch.nn.functional.prelu(x, self.weight)

    def extra_repr(self) -> str:
        return f"num_parameters={self.num_parameters}, init={self.init}, name={self.name!r}"
