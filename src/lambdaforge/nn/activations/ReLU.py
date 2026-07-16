"""Implementation of the ReLU object."""

import torch
from torch import Tensor

from lambdaforge.nn.activations.Activation import Activation


class ReLU(Activation):
    """Rectified Linear Unit.

    Formula:
        f(x) = max(0, x)

    The most widely used activation in deep networks due to its
    simplicity, non-saturation for x > 0, and constant gradient.
    Its main drawback is the potential for dead neurons (zero
    gradient for x < 0).

    Parameters
    ----------
    inplace : bool
        If ``True``, apply the operation in-place. Default: ``False``.
    name : str | None
        Optional name to identify this activation instance.
    """

    def __init__(self, inplace: bool = False, name: str | None = None, **kwargs) -> None:
        super().__init__(name=name)
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return torch.relu_(x) if self.inplace else torch.relu(x)

    def extra_repr(self) -> str:
        return f"inplace={self.inplace}, name={self.name!r}"
