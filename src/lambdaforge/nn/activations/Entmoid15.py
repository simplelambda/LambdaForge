r"""Binary :math:`\alpha=1.5` entmax activation."""

from __future__ import annotations

import torch
from torch import Tensor

from lambdaforge.nn.activations.Activation import Activation


class Entmoid15(Activation):
    r"""Return the positive-class probability of two-class entmax 1.5.

    ``Entmoid15(x)`` is the efficient scalar equivalent of applying
    :class:`Entmax15` to the two logits ``[0, x]``. Unlike sigmoid, it can
    produce exact zeros and ones, which makes it useful for differentiable
    tree routing.

    Parameters
    ----------
    force_float32:
        Evaluate the square-root expression in float32 for half and bfloat16
        inputs, then restore the original dtype.
    name:
        Optional display name inherited from :class:`Activation`.
    """

    def __init__(self, force_float32: bool = True, name: str | None = None) -> None:
        super().__init__(name=name)
        self.force_float32 = force_float32

    def forward(self, x: Tensor) -> Tensor:
        """Return positive-class probabilities with the same shape as ``x``."""
        if not torch.is_floating_point(x):
            raise TypeError("Entmoid15 requires a floating-point tensor.")

        input_dtype = x.dtype
        work = x.float() if self.force_float32 and x.dtype in (torch.float16, torch.bfloat16) else x
        magnitude = work.abs()
        threshold = (magnitude + (8.0 - magnitude.square()).clamp_min(0.0).sqrt()) / 2.0
        threshold = torch.where(threshold <= magnitude, torch.full_like(threshold, 2.0), threshold)
        negative_probability = 0.25 * (threshold - magnitude).clamp_min(0.0).square()
        output = torch.where(work >= 0.0, 1.0 - negative_probability, negative_probability)
        return output.to(dtype=input_dtype)

    def extra_repr(self) -> str:
        """Return constructor-relevant state for module representations."""
        return f"force_float32={self.force_float32}, name={self.name!r}"
