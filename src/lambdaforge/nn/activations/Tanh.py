"""Implementation of the Tanh object."""

import torch
from torch import Tensor

from lambdaforge.nn.activations.Activation import Activation


class Tanh(Activation):
    """Hyperbolic tangent.

    Formula:
        f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

    Equivalently:
        f(x) = (e^(2x) - 1) / (e^(2x) + 1)

    Squashes any real value into the interval (-1, 1). Unlike sigmoid,
    tanh is zero-centered, which helps keep activations balanced.
    Commonly used in RNNs, LSTMs, and as output activation when values
    between -1 and 1 are needed.

    Parameters
    ----------
    name : str | None
        Optional name to identify this activation instance.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(x)
