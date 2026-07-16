"""Implementation of the LeakyReLU object."""

import torch
from torch import Tensor

from lambdaforge.nn.activations.Activation import Activation


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
