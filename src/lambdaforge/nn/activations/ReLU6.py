"""Implementation of the ReLU6 object."""

from __future__ import annotations

import torch
from torch import Tensor

from lambdaforge.nn.activations.Activation import Activation


class ReLU6(Activation):
    """Rectified Linear Unit capped at six.

    Parameters
    ----------
    inplace : bool
        Whether to modify the input tensor in place. Default: ``False``.
    name : str | None
        Optional name used to identify this activation instance.
    """

    def __init__(self, inplace: bool = False, name: str | None = None) -> None:
        super().__init__(name=name)
        if not isinstance(inplace, bool):
            raise TypeError("inplace must be a boolean")
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.relu6(x, inplace=self.inplace)

    def extra_repr(self) -> str:
        return f"inplace={self.inplace}, name={self.name!r}"
