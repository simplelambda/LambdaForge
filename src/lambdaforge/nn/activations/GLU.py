"""Implementation of the GLU object."""

from __future__ import annotations

import torch
from torch import Tensor

from lambdaforge.nn.activations.Activation import Activation


class GLU(Activation):
    """Gated Linear Unit over two equal halves of an input dimension.

    The first half is the value branch and the second half is passed through a
    sigmoid gate. The output therefore has half the size along ``dim``.

    Parameters
    ----------
    dim : int
        Dimension to split into value and gate branches. Default: ``-1``.
    name : str | None
        Optional name used to identify this activation instance.
    """

    def __init__(self, dim: int = -1, name: str | None = None) -> None:
        super().__init__(name=name)
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise TypeError("dim must be an integer")
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 0 or not -x.ndim <= self.dim < x.ndim:
            raise ValueError(f"dim={self.dim} is invalid for an input with {x.ndim} dimensions")
        if x.shape[self.dim] % 2 != 0:
            raise ValueError("GLU requires an even input size along dim")
        return torch.nn.functional.glu(x, dim=self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, name={self.name!r}"
