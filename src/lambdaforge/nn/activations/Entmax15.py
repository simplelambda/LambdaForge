r"""Sparse :math:`\alpha=1.5` entmax activation."""

from __future__ import annotations

import torch
from torch import Tensor

from lambdaforge.nn.activations.Activation import Activation


class Entmax15(Activation):
    r"""Map logits to a sparse probability simplex with entmax 1.5.

    This is a PyTorch-native implementation of the closed-form
    :math:`\alpha=1.5` entmax transformation used by NODE and GradTree.
    The operation is differentiated through ordinary tensor operations; no
    custom autograd state or optional dependency is required.

    Parameters
    ----------
    dim:
        Dimension over which outputs sum to one.
    force_float32:
        Compute the sorting and support threshold in float32 for half and
        bfloat16 inputs, then restore the input dtype. This is safer under
        mixed precision without changing float32 or float64 inputs.
    name:
        Optional display name inherited from :class:`Activation`.
    """

    def __init__(
        self,
        dim: int = -1,
        force_float32: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.dim = dim
        self.force_float32 = force_float32

    def forward(self, x: Tensor) -> Tensor:
        """Return sparse simplex weights with the same shape as ``x``."""
        if not torch.is_floating_point(x):
            raise TypeError("Entmax15 requires a floating-point tensor.")
        if x.ndim == 0:
            raise ValueError("Entmax15 requires at least one tensor dimension.")

        dim = self.dim if self.dim >= 0 else x.ndim + self.dim
        if not 0 <= dim < x.ndim:
            raise ValueError(f"dim={self.dim} is invalid for a {x.ndim}-D tensor.")
        if x.shape[dim] == 0:
            raise ValueError("Entmax15 cannot reduce an empty dimension.")

        input_dtype = x.dtype
        work = x.float() if self.force_float32 and x.dtype in (torch.float16, torch.bfloat16) else x
        work = work / 2.0
        work = work - work.amax(dim=dim, keepdim=True)

        sorted_work = torch.sort(work, dim=dim, descending=True).values
        rank_shape = [1] * work.ndim
        rank_shape[dim] = work.shape[dim]
        ranks = torch.arange(
            1,
            work.shape[dim] + 1,
            dtype=work.dtype,
            device=work.device,
        ).view(rank_shape)

        mean = sorted_work.cumsum(dim=dim) / ranks
        mean_square = sorted_work.square().cumsum(dim=dim) / ranks
        variance_sum = ranks * (mean_square - mean.square())
        delta = (1.0 - variance_sum) / ranks
        thresholds = mean - delta.clamp_min(0.0).sqrt()
        support_size = (thresholds <= sorted_work).sum(dim=dim, keepdim=True).clamp_min(1)
        threshold = thresholds.gather(dim, support_size - 1)
        output = (work - threshold).clamp_min(0.0).square()
        return output.to(dtype=input_dtype)

    def extra_repr(self) -> str:
        """Return constructor-relevant state for module representations."""
        return f"dim={self.dim}, force_float32={self.force_float32}, name={self.name!r}"
