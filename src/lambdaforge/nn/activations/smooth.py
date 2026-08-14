"""Cohesive smooth activation contracts and implementations."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from lambdaforge.nn.activations.base import Activation


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


class Hardsigmoid(Activation):
    """Piecewise-linear approximation of the sigmoid activation.

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
        return torch.nn.functional.hardsigmoid(x, inplace=self.inplace)

    def extra_repr(self) -> str:
        return f"inplace={self.inplace}, name={self.name!r}"


class Hardswish(Activation):
    """Efficient piecewise-linear approximation of SiLU/Swish.

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
        return torch.nn.functional.hardswish(x, inplace=self.inplace)

    def extra_repr(self) -> str:
        return f"inplace={self.inplace}, name={self.name!r}"


class Identity(Activation):
    """Identity activation (passthrough / no-op).

    Formula:
        f(x) = x

    Leaves the input unchanged. Useful as a placeholder when an
    activation slot is required but no transformation is desired
    (e.g. final layer of a regressor, or when the model already
    includes its own activation).

    Parameters
    ----------
    name : str | None
        Optional name to identify this activation instance.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x


class Mish(Activation):
    """Smooth non-monotonic activation ``x * tanh(softplus(x))``.

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
        return torch.nn.functional.mish(x, inplace=self.inplace)

    def extra_repr(self) -> str:
        return f"inplace={self.inplace}, name={self.name!r}"


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


class Softplus(Activation):
    """Smooth approximation of ReLU with configurable sharpness.

    Parameters
    ----------
    beta : float
        Positive sharpness coefficient. Default: ``1.0``.
    threshold : float
        Input threshold above which the linear approximation is used for
        numerical stability. Default: ``20.0``.
    name : str | None
        Optional name used to identify this activation instance.
    """

    def __init__(
        self,
        beta: float = 1.0,
        threshold: float = 20.0,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if isinstance(beta, bool) or not isinstance(beta, (int, float)):
            raise TypeError("beta must be a real number")
        if not math.isfinite(float(beta)) or float(beta) <= 0.0:
            raise ValueError("beta must be finite and greater than zero")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be a real number")
        if not math.isfinite(float(threshold)):
            raise ValueError("threshold must be finite")
        self.beta = float(beta)
        self.threshold = float(threshold)

    def forward(self, x: Tensor) -> Tensor:
        parameters: dict[str, Any] = {"beta": self.beta, "threshold": self.threshold}
        return torch.nn.functional.softplus(x, **parameters)

    def extra_repr(self) -> str:
        return f"beta={self.beta}, threshold={self.threshold}, name={self.name!r}"


class Softsign(Activation):
    """Smooth bounded activation ``x / (1 + abs(x))``.

    Parameters
    ----------
    name : str | None
        Optional name used to identify this activation instance.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.softsign(x)


class SquarePlus(Activation):
    r"""Algebraic smooth approximation of ReLU.

    Computes ``0.5 * (x + sqrt(x^2 + b))``.

    Parameters
    ----------
    b : float
        Positive smoothness constant. Default: ``4.0``.
    name : str | None
        Optional name used to identify this activation instance.
    """

    def __init__(self, b: float = 4.0, name: str | None = None) -> None:
        super().__init__(name=name)
        if isinstance(b, bool) or not isinstance(b, (int, float)):
            raise TypeError("b must be a real number")
        if not math.isfinite(float(b)) or float(b) <= 0.0:
            raise ValueError("b must be finite and greater than zero")
        self.b = float(b)

    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * (x + torch.sqrt(x.square() + self.b))

    def extra_repr(self) -> str:
        return f"b={self.b}, name={self.name!r}"


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
