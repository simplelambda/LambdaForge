"""Learnable-query multi-head attention pooling."""

from __future__ import annotations

import torch
from torch import nn

from lambdaforge.nn.pooling.Pooling import Pooling


class MultiheadAttentionPooling(Pooling):
    """Summarize a set using configurable learned queries and attention heads.

    Parameters
    ----------
    in_features:
        Input and attention embedding size.
    num_heads:
        Number of attention heads; must divide ``in_features``.
    num_queries:
        Number of learned seed queries.
    merge:
        ``"mean"`` returns ``(B, O)``, ``"concat"`` returns ``(B, Q*O)``
        unless projected, and ``"none"`` preserves ``(B, Q, O)``.
    output_features:
        Optional projection size after attention. For concat mode the
        projection is applied after concatenation; otherwise per query.
    """

    def __init__(
        self,
        in_features: int,
        num_heads: int = 4,
        num_queries: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        merge: str = "mean",
        output_features: int | None = None,
        query_init_std: float = 0.02,
        name: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(name=name)
        if in_features < 1 or num_heads < 1 or num_queries < 1:
            raise ValueError("in_features, num_heads and num_queries must be positive.")
        if in_features % num_heads != 0:
            raise ValueError("in_features must be divisible by num_heads.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if merge not in {"mean", "concat", "none"}:
            raise ValueError("merge must be 'mean', 'concat' or 'none'.")
        if output_features is not None and output_features < 1:
            raise ValueError("output_features must be positive or None.")
        if query_init_std <= 0:
            raise ValueError("query_init_std must be positive.")
        factory_kwargs = {"device": device, "dtype": dtype}
        self.attention = nn.MultiheadAttention(
            embed_dim=in_features,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            batch_first=True,
            **factory_kwargs,
        )
        self.queries = nn.Parameter(
            torch.empty(num_queries, in_features, device=device, dtype=dtype)
        )
        nn.init.normal_(self.queries, mean=0.0, std=query_init_std)
        projection_input = num_queries * in_features if merge == "concat" else in_features
        self.output = (
            nn.Linear(projection_input, output_features, bias=bias, **factory_kwargs)
            if output_features is not None
            else nn.Identity()
        )
        self.in_features = int(in_features)
        self.num_queries = int(num_queries)
        self.merge = merge

    def attention_weights(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
    ) -> torch.Tensor:
        """Return query-to-element attention weights without pooled values."""
        _, weights, _ = self._attend(x, mask, average_attn_weights)
        return weights

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        pooled, _, has_values = self._attend(x, mask, True)
        if self.merge == "mean":
            output = self.output(pooled.mean(dim=1))
        elif self.merge == "concat":
            output = self.output(pooled.flatten(start_dim=1))
        else:
            output = self.output(pooled)
        expand_dims = output.ndim - 1
        return torch.where(
            has_values.view(has_values.shape[0], *([1] * expand_dims)),
            output,
            torch.zeros_like(output),
        )

    def _attend(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        average_attn_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] < 1 or x.shape[2] != self.in_features:
            raise ValueError(
                f"Expected non-empty x shaped (B, N, {self.in_features}), got {tuple(x.shape)}."
            )
        if mask is not None and tuple(mask.shape) != tuple(x.shape[:2]):
            raise ValueError("mask must have shape (B, N).")
        valid = (
            torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
            if mask is None
            else mask.bool()
        )
        has_values = valid.any(dim=1)
        safe_valid = valid.clone()
        safe_valid[~has_values, 0] = True
        safe_x = torch.where(safe_valid.unsqueeze(-1), x, torch.zeros_like(x))
        queries = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        pooled, weights = self.attention(
            queries,
            safe_x,
            safe_x,
            key_padding_mask=~safe_valid,
            need_weights=True,
            average_attn_weights=average_attn_weights,
        )
        weights = weights * has_values.view(has_values.shape[0], *([1] * (weights.ndim - 1)))
        return pooled, weights, has_values
