"""Cohesive rectifiers activation contracts and implementations."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from lambdaforge.nn.activations.base import Activation


class CELU(Activation):
    """Continuously Differentiable Exponential Linear Unit.

    Parameters
    ----------
    alpha : float
        Positive scale of the negative branch. Default: ``1.0``.
    inplace : bool
        Whether to modify the input tensor in place. Default: ``False``.
    name : str | None
        Optional name used to identify this activation instance.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        inplace: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be a real number")
        if not math.isfinite(float(alpha)) or float(alpha) <= 0.0:
            raise ValueError("alpha must be finite and greater than zero")
        if not isinstance(inplace, bool):
            raise TypeError("inplace must be a boolean")
        self.alpha = float(alpha)
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.celu(x, alpha=self.alpha, inplace=self.inplace)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, inplace={self.inplace}, name={self.name!r}"


class ELU(Activation):
    """Exponential Linear Unit.

    Formula:
        f(x) = x                if x > 0
        f(x) = alpha*(e^x - 1)  if x <= 0

    Smooths negative outputs via an exponential curve, keeping
    activations zero-centered and often converging faster than ReLU.

    Parameters
    ----------
    alpha : float
        Scale factor for the negative branch. Default: ``1.0``.
    inplace : bool
        If ``True``, apply the operation in-place. Default: ``False``.
    name : str | None
        Optional name to identify this activation instance.
    """

    def __init__(self, alpha: float = 1.0, inplace: bool = False, name: str | None = None) -> None:
        super().__init__(name=name)
        self.alpha = float(alpha)
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.elu(x, alpha=self.alpha, inplace=self.inplace)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, inplace={self.inplace}, name={self.name!r}"


class LeakyReLU(Activation):
    """Leaky Rectified Linear Unit.

    Fórmula:
    $$
    f(x) = \\begin{cases}
        x,                      & x > 0 \\\\\n        \\alpha \\, x,            & x \\le 0
    \\end{cases}
    $$

    A diferencia de ReLU, permite un pequeño gradiente ($\\alpha$) para
    valores negativos, lo que evita neuronas muertas y mejora el flujo
    de gradiente.

    Parameters
    ----------
    negative_slope : float
        Pendiente $\\alpha$ para la rama negativa (default ``0.01``).
    inplace : bool
        Si ``True``, aplica la operación in-place (default ``False``).
    name : str | None
        Nombre opcional para identificar la activación.
    """

    def __init__(
        self,
        negative_slope: float = 0.01,
        inplace: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.negative_slope = float(negative_slope)
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.leaky_relu(
            x,
            negative_slope=self.negative_slope,
            inplace=self.inplace,
        )

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}, inplace={self.inplace}, name={self.name!r}"


class PReLU(Activation):
    """Parametric Rectified Linear Unit with learnable negative slopes.

    Parameters
    ----------
    num_parameters : int
        Number of learned slopes. Use ``1`` to share one slope or the number
        of channels to learn one slope per channel. Default: ``1``.
    init : float
        Initial value of every negative slope. Default: ``0.25``.
    name : str | None
        Optional name used to identify this activation instance.
    """

    def __init__(
        self,
        num_parameters: int = 1,
        init: float = 0.25,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if isinstance(num_parameters, bool) or not isinstance(num_parameters, int):
            raise TypeError("num_parameters must be an integer")
        if num_parameters <= 0:
            raise ValueError("num_parameters must be greater than zero")
        if isinstance(init, bool) or not isinstance(init, (int, float)):
            raise TypeError("init must be a real number")
        if not math.isfinite(float(init)):
            raise ValueError("init must be finite")

        self.num_parameters = num_parameters
        self.init = float(init)
        self.weight = nn.Parameter(torch.full((num_parameters,), self.init))

    def forward(self, x: Tensor) -> Tensor:
        if self.num_parameters > 1 and (x.ndim < 2 or x.shape[1] != self.num_parameters):
            raise ValueError(
                "PReLU with multiple parameters expects input channel dimension 1 "
                f"to have size {self.num_parameters}"
            )
        return torch.nn.functional.prelu(x, self.weight)

    def extra_repr(self) -> str:
        return f"num_parameters={self.num_parameters}, init={self.init}, name={self.name!r}"


class ReLU(Activation):
    """Rectified Linear Unit.

    Formula:
        f(x) = max(0, x)

    The most widely used activation in deep networks due to its
    simplicity, non-saturation for x > 0, and constant gradient.
    Its main drawback is the potential for dead neurons (zero
    gradient for x < 0).

    Parameters
    ----------
    inplace : bool
        If ``True``, apply the operation in-place. Default: ``False``.
    name : str | None
        Optional name to identify this activation instance.
    """

    def __init__(self, inplace: bool = False, name: str | None = None, **kwargs) -> None:
        super().__init__(name=name)
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return torch.relu_(x) if self.inplace else torch.relu(x)

    def extra_repr(self) -> str:
        return f"inplace={self.inplace}, name={self.name!r}"


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


class SELU(Activation):
    """Scaled Exponential Linear Unit for self-normalizing networks.

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
        return torch.nn.functional.selu(x, inplace=self.inplace)

    def extra_repr(self) -> str:
        return f"inplace={self.inplace}, name={self.name!r}"
