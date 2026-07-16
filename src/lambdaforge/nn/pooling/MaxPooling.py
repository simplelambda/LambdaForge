"""Implementation of the MaxPooling object."""

from __future__ import annotations

import torch

from lambdaforge.nn.pooling.Pooling import Pooling


class MaxPooling(Pooling):
    r"""Max pooling over the set dimension.

    Selects the maximum value along each feature dimension across the set
    of valid elements:

    .. math::

        z_d = \max_{i \in \mathcal{V}} h_{i,d}

    Invalid positions (mask = 0) are filled with :math:`-\infty` before the
    max, so they never win the competition.  When no mask is supplied, all
    positions are treated as valid.

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
            invalid = ~mask.bool()  # (B, N)
            x = x.masked_fill(invalid.unsqueeze(-1), float("-inf"))

        return x.max(dim=1).values  # (B, D)
