"""Implementation of the GELU object."""

import torch
from torch import Tensor

from lambdaforge.nn.activations.Activation import Activation


class GELU(Activation):
    """Gaussian Error Linear Unit.

    Exact formula:
        f(x) = x * Phi(x)

    where Phi(x) is the CDF of the standard normal distribution:
        Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))

    Tanh approximation (used in GPT, BERT):
        f(x) ~= 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Parameters
    ----------
    approximate : str
        ``"none"`` for the exact erf-based version,
        ``"tanh"`` for the fast approximation. Default: ``"none"``.
    name : str | None
        Optional name to identify this activation instance.
    """

    def __init__(self, approximate: str = "none", name: str | None = None) -> None:
        super().__init__(name=name)
        self.approximate = approximate

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.gelu(x, approximate=self.approximate)

    def extra_repr(self) -> str:
        return f"approximate={self.approximate!r}, name={self.name!r}"
