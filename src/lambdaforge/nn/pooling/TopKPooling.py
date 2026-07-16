"""Implementation of the TopKPooling object."""

from __future__ import annotations

import torch
import torch.nn as nn

from lambdaforge.nn.pooling.Pooling import Pooling


class TopKPooling(Pooling):
    r"""Top-K pooling with a learned element scoring function.

    Learns a scalar score for each element and retains the :math:`K`
    highest-scoring ones.  The final representation is the mean of those
    :math:`K` elements, weighted by their normalised scores:

    .. math::

        s_i = \sigma\!\bigl(\mathbf{v}^\top h_i\bigr)

    .. math::

        \mathcal{T} = \operatorname{top-}K\{s_i\}_{i=1}^{N}

    .. math::

        z = \frac{\sum_{i \in \mathcal{T}} s_i\, h_i}
                 {\sum_{i \in \mathcal{T}} s_i}

    Masked-out positions receive score :math:`-\infty` before the top-K
    selection so they are never chosen.  If the valid set has fewer than
    :math:`K` elements, all valid elements are selected.

    Parameters
    ----------
    in_features : int
        Dimensionality of each input feature vector ``D``.
    k : int
        Number of elements to retain.
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

    K:
        Number of retained elements (``k``).

    x:
        Shape ``(B, N, D)``.
    mask:
        Shape ``(B, N)``.  Float or bool — ``1`` / ``True`` = valid.
    output:
        Shape ``(B, D)``.
    """

    def __init__(
        self,
        in_features: int,
        k: int,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        self.k = k
        self.scorer = nn.Linear(in_features, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = x.shape
        k = min(self.k, N)

        scores = self.scorer(x).squeeze(-1)  # (B, N)

        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float("-inf"))

        topk_scores, topk_idx = scores.topk(k, dim=1)  # (B, K)

        # Gather selected features
        idx_expanded = topk_idx.unsqueeze(-1).expand(B, k, D)
        topk_feats = x.gather(1, idx_expanded)  # (B, K, D)

        # Weighted mean by sigmoid-normalised scores
        weights = torch.sigmoid(topk_scores)  # (B, K)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        return torch.einsum("bk,bkd->bd", weights, topk_feats)  # (B, D)
