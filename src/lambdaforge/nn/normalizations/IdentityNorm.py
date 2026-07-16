"""Implementation of the IdentityNorm object."""

from __future__ import annotations

import torch
from torch import nn

from lambdaforge.nn.normalizations.Normalization import Normalization


class IdentityNorm(Normalization):
    r"""Identity normalization.

    This layer returns the input tensor unchanged. It is useful as a drop-in
    replacement when normalization is optional.

    Parameters
    ----------
    name : str | None
        Optional name used to identify the normalization layer.
    """

    def __init__(self, name: str | None = None, **kwargs) -> None:
        super().__init__(name=name)
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.identity(x)
