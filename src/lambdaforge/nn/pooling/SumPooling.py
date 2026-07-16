"""Implementation of the SumPooling object."""

from __future__ import annotations

import torch

from lambdaforge.nn.pooling.Pooling import Pooling


class SumPooling(Pooling):
    r"""Sum pooling over the set dimension.

    Computes the element-wise sum of valid feature vectors:

    .. math::

        z = \sum_{i \in \mathcal{V}} h_i

    Invalid positions contribute zero to the sum.  When no mask is
    supplied, all positions are treated as valid.

    Parameters
    ----------
    name : str | None
        Optional name used to identify the pooling layer.

    Shape convention
    ----------------
    B:
        Batch size.

    N:
        Number of elements per sample.  For example, vertices, atoms, nodes,
        tokens, or neighbors in a local patch.

    D:
        Feature dimension per element.

    x:
        Shape ``(B, N, D)``.
    mask:
        Shape ``(B, N)``.  Float or bool — ``1`` / ``True`` = valid.
    output:
        Shape ``(B, D)``.
    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name=name)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            mask = mask.to(dtype=x.dtype)
            x = x * mask.unsqueeze(-1)  # (B, N, D)

        return x.sum(dim=1)  # (B, D)
