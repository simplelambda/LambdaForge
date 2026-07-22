"""Implementation of the GEGLU object."""

from __future__ import annotations

import torch
from torch import Tensor

from lambdaforge.nn.activations.Activation import Activation


class GEGLU(Activation):
    """GELU-gated linear unit over two equal input halves.

    Parameters
    ----------
    dim : int
        Dimension to split into value and gate branches. Default: ``-1``.
    approximate : str
        GELU implementation, either ``"none"`` or ``"tanh"``. Default:
        ``"none"``.
    name : str | None
        Optional name used to identify this activation instance.
    """

    def __init__(
        self,
        dim: int = -1,
        approximate: str = "none",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise TypeError("dim must be an integer")
        if approximate not in {"none", "tanh"}:
            raise ValueError("approximate must be either 'none' or 'tanh'")
        self.dim = dim
        self.approximate = approximate

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 0 or not -x.ndim <= self.dim < x.ndim:
            raise ValueError(f"dim={self.dim} is invalid for an input with {x.ndim} dimensions")
        if x.shape[self.dim] % 2 != 0:
            raise ValueError("GEGLU requires an even input size along dim")
        value, gate = x.chunk(2, dim=self.dim)
        return value * torch.nn.functional.gelu(gate, approximate=self.approximate)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, approximate={self.approximate!r}, name={self.name!r}"
