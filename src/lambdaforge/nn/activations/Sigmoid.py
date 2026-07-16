"""Implementation of the Sigmoid object."""

import torch
from torch import Tensor

from lambdaforge.nn.activations.Activation import Activation


class Sigmoid(Activation):
    """Sigmoid (logistic) function.

    Formula:
        f(x) = sigma(x) = 1 / (1 + e^(-x))

    Squashes any real value into the interval (0, 1). Widely used in
    binary classification (final layer) and attention mechanisms
    (gating). Tends to saturate for extreme inputs, producing very
    small gradients.

    Parameters
    ----------
    name : str | None
        Optional name to identify this activation instance.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x)
