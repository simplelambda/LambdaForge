"""Implementation of the ELU object."""

import torch
from torch import Tensor

from lambdaforge.nn.activations.Activation import Activation


class ELU(Activation):
    """Exponential Linear Unit.

    Formula:
        f(x) = x                if x > 0
        f(x) = alpha*(e^x - 1)  if x <= 0

    Smooths negative outputs via an exponential curve, keeping
    activations zero-centered and often converging faster than ReLU.

    Parameters
    ----------
    alpha : float
        Scale factor for the negative branch. Default: ``1.0``.
    inplace : bool
        If ``True``, apply the operation in-place. Default: ``False``.
    name : str | None
        Optional name to identify this activation instance.
    """

    def __init__(self, alpha: float = 1.0, inplace: bool = False, name: str | None = None) -> None:
        super().__init__(name=name)
        self.alpha = float(alpha)
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.elu(x, alpha=self.alpha, inplace=self.inplace)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, inplace={self.inplace}, name={self.name!r}"
