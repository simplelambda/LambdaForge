"""Implementation of the MultiHeadGatedAttentionPooling object."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lambdaforge.nn.pooling.Pooling import Pooling


class MultiHeadGatedAttentionPooling(Pooling):
    r"""Multi-head gated attention pooling for sets.

    This extends gated attention pooling with several attention heads. Each
    head can focus on a different subset of instances; the head-wise pooled
    vectors are either averaged or concatenated and projected back to the
    input feature dimension.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = 128,
        num_heads: int = 4,
        merge: str = "mean",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if in_features < 1:
            raise ValueError("in_features must be >= 1.")
        if hidden_features < 1:
            raise ValueError("hidden_features must be >= 1.")
        if num_heads < 1:
            raise ValueError("num_heads must be >= 1.")
        if merge not in {"mean", "concat"}:
            raise ValueError("merge must be 'mean' or 'concat'.")

        self.in_features = int(in_features)
        self.hidden_features = int(hidden_features)
        self.num_heads = int(num_heads)
        self.merge = merge

        self.V = nn.Linear(in_features, num_heads * hidden_features, bias=False)
        self.U = nn.Linear(in_features, num_heads * hidden_features, bias=False)
        self.w = nn.Parameter(torch.empty(num_heads, hidden_features))
        nn.init.xavier_uniform_(self.w)

        self.output = (
            nn.Linear(num_heads * in_features, in_features, bias=False)
            if merge == "concat"
            else nn.Identity()
        )

    def attention_weights(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "MultiHeadGatedAttentionPooling expects x with shape (B, N, D), "
                f"got {tuple(x.shape)}."
            )

        batch_size, n_items, _ = x.shape
        v = torch.tanh(self.V(x)).view(batch_size, n_items, self.num_heads, self.hidden_features)
        u = torch.sigmoid(self.U(x)).view(batch_size, n_items, self.num_heads, self.hidden_features)
        gated = v * u
        scores = torch.einsum("bnhl,hl->bhn", gated, self.w)

        if mask is not None:
            scores = scores.masked_fill(~mask.bool().unsqueeze(1), float("-inf"))

        return F.softmax(scores, dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        weights = self.attention_weights(x, mask)
        if mask is not None:
            weights = weights.masked_fill(~mask.bool().unsqueeze(1), 0.0)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        pooled = torch.einsum("bhn,bnd->bhd", weights, x)
        if self.merge == "mean":
            return pooled.mean(dim=1)

        return self.output(pooled.flatten(start_dim=1))
