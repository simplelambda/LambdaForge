"""Implementation of the SiLU object."""

import torch
from torch import Tensor

from lambdaforge.nn.activations.Activation import Activation


class SiLU(Activation):
    """Sigmoid Linear Unit (also known as Swish).

    Formula:
        f(x) = x * sigma(x) = x / (1 + e^(-x))

    where sigma(x) is the sigmoid function.

    Combines the benefits of ReLU (non-saturation for x > 0) with
    smooth behaviour near zero and self-gating. Popularised in
    EfficientNet and modern architectures. Outperforms ReLU on
    deep image classification tasks.

    Parameters
    ----------
    name : str | None
        Optional name to identify this activation instance.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.silu(x)
