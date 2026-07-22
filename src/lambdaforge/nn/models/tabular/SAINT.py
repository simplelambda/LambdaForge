"""Row-and-feature attention model inspired by SAINT for tabular data."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import nn

from lambdaforge.nn.models.Model import Model


class SAINT(Model):
    """Alternate within-row feature attention and across-row attention blocks."""

    def __init__(
        self,
        num_continuous_features: int,
        categorical_cardinalities: Sequence[int],
        out_features: int,
        d_model: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        feedforward_dim: int | None = None,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        cardinalities = tuple(int(value) for value in categorical_cardinalities)
        if num_continuous_features < 0 or any(value < 1 for value in cardinalities):
            raise ValueError("Feature counts must be non-negative and cardinalities positive.")
        if num_continuous_features + len(cardinalities) < 1:
            raise ValueError("At least one feature is required.")
        if min(out_features, d_model, num_heads, num_layers) < 1:
            raise ValueError("Output size, d_model, num_heads and num_layers must be positive.")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.num_continuous_features = num_continuous_features
        self.categorical_cardinalities = cardinalities
        self.continuous_weight = (
            nn.Parameter(torch.empty(num_continuous_features, d_model))
            if num_continuous_features
            else None
        )
        self.continuous_bias = (
            nn.Parameter(torch.zeros(num_continuous_features, d_model))
            if num_continuous_features
            else None
        )
        if cardinalities:
            offsets = torch.tensor([0, *cardinalities[:-1]], dtype=torch.long).cumsum(0)
            self.register_buffer("categorical_offsets", offsets, persistent=True)
            self.category_embedding: nn.Embedding | None = nn.Embedding(sum(cardinalities), d_model)
        else:
            self.register_buffer("categorical_offsets", None, persistent=False)
            self.category_embedding = None
        self.feature_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=feedforward_dim or 4 * d_model,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                    bias=bias,
                )
                for _ in range(num_layers)
            ]
        )
        self.row_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    d_model,
                    num_heads,
                    dropout=dropout,
                    bias=bias,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.row_normalizations = nn.ModuleList(
            [nn.LayerNorm(d_model, bias=bias) for _ in range(num_layers)]
        )
        self.output_normalization = nn.LayerNorm(d_model, bias=bias)
        self.head = nn.Linear(d_model, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize continuous tokenizer parameters."""
        if self.continuous_weight is not None:
            nn.init.normal_(self.continuous_weight, std=0.02)

    def forward(
        self,
        continuous: torch.Tensor | None = None,
        categorical: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict from continuous floats and zero-based categorical indices."""
        tokens = self._tokens(continuous, categorical)
        for feature_layer, row_layer, row_norm in zip(
            self.feature_layers,
            self.row_layers,
            self.row_normalizations,
            strict=True,
        ):
            tokens = feature_layer(tokens)
            by_feature = tokens.transpose(0, 1)
            attended, _ = row_layer(by_feature, by_feature, by_feature, need_weights=False)
            tokens = row_norm((by_feature + self.dropout(attended)).transpose(0, 1))
        return self.head(self.output_normalization(tokens).mean(dim=1))

    def _tokens(
        self,
        continuous: torch.Tensor | None,
        categorical: torch.Tensor | None,
    ) -> torch.Tensor:
        tokens: list[torch.Tensor] = []
        batch_size: int | None = None
        if self.num_continuous_features:
            if continuous is None or continuous.ndim != 2:
                raise ValueError("continuous must have shape (batch, continuous_features).")
            if continuous.shape[1] != self.num_continuous_features:
                raise ValueError("continuous has an incompatible feature dimension.")
            if not torch.is_floating_point(continuous):
                raise TypeError("continuous must use a floating-point dtype.")
            assert self.continuous_weight is not None
            assert self.continuous_bias is not None
            tokens.append(continuous.unsqueeze(-1) * self.continuous_weight + self.continuous_bias)
            batch_size = continuous.shape[0]
        elif continuous is not None:
            raise ValueError("continuous was supplied but no continuous features are configured.")
        if self.categorical_cardinalities:
            if categorical is None or categorical.ndim != 2:
                raise ValueError("categorical must have shape (batch, categorical_features).")
            if categorical.shape[1] != len(self.categorical_cardinalities):
                raise ValueError("categorical has an incompatible feature dimension.")
            if categorical.dtype not in (torch.int32, torch.int64):
                raise TypeError("categorical must use an integer dtype.")
            if batch_size is not None and categorical.shape[0] != batch_size:
                raise ValueError("continuous and categorical batch sizes must match.")
            for index, cardinality in enumerate(self.categorical_cardinalities):
                invalid = (categorical[:, index] < 0) | (categorical[:, index] >= cardinality)
                if bool(invalid.any()):
                    raise ValueError(f"categorical feature {index} contains an invalid index.")
            embedding = cast(nn.Embedding, self.category_embedding)
            offsets = cast(torch.Tensor, self.categorical_offsets)
            tokens.append(embedding(categorical.long() + offsets.unsqueeze(0)))
        elif categorical is not None:
            raise ValueError("categorical was supplied but no categorical features are configured.")
        if not tokens:
            raise RuntimeError("No configured input tensor was supplied.")
        return torch.cat(tokens, dim=1)
