"""Implementation of the MeanPooling object."""

from __future__ import annotations

import torch

from lambdaforge.nn.pooling.Pooling import Pooling


class MeanPooling(Pooling):
    r"""Mean pooling over the set dimension.

    Computes the arithmetic mean of valid feature vectors:

    .. math::

        z = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} h_i

    where :math:`\mathcal{V}` is the set of valid positions as indicated by
    the mask.  When no mask is supplied, all positions are treated as valid.

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
        if mask is None:
            return x.mean(dim=1)

        mask = mask.to(dtype=x.dtype)  # (B, N)
        x = x * mask.unsqueeze(-1)  # (B, N, D)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B, 1)
        return x.sum(dim=1) / denom  # (B, D)
