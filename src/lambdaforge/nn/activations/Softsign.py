"""Implementation of the Softsign object."""

from __future__ import annotations

import torch
from torch import Tensor

from lambdaforge.nn.activations.Activation import Activation


class Softsign(Activation):
    """Smooth bounded activation ``x / (1 + abs(x))``.

    Parameters
    ----------
    name : str | None
        Optional name used to identify this activation instance.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.softsign(x)
