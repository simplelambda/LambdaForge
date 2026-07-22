"""Automatic feature-interaction network for heterogeneous tabular inputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import nn

from lambdaforge.nn.models.Model import Model


class AutoInt(Model):
    """Learn explicit high-order feature interactions through self-attention."""

    def __init__(
        self,
        num_continuous_features: int,
        categorical_cardinalities: Sequence[int],
        out_features: int,
        embedding_features: int = 32,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.0,
        residual: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        cardinalities = tuple(int(value) for value in categorical_cardinalities)
        num_fields = num_continuous_features + len(cardinalities)
        if num_continuous_features < 0 or any(value < 1 for value in cardinalities):
            raise ValueError("Feature counts must be non-negative and cardinalities positive.")
        if min(num_fields, out_features, embedding_features, num_heads, num_layers) < 1:
            raise ValueError("Configured dimensions and layer counts must be positive.")
        if embedding_features % num_heads:
            raise ValueError("embedding_features must be divisible by num_heads.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.num_continuous_features = num_continuous_features
        self.categorical_cardinalities = cardinalities
        self.num_fields = num_fields
        self.residual = residual
        self.continuous_weight = (
            nn.Parameter(torch.empty(num_continuous_features, embedding_features))
            if num_continuous_features
            else None
        )
        self.continuous_bias = (
            nn.Parameter(torch.zeros(num_continuous_features, embedding_features))
            if num_continuous_features
            else None
        )
        if cardinalities:
            offsets = torch.tensor([0, *cardinalities[:-1]], dtype=torch.long).cumsum(0)
            self.register_buffer("categorical_offsets", offsets, persistent=True)
            self.category_embedding: nn.Embedding | None = nn.Embedding(
                sum(cardinalities), embedding_features
            )
        else:
            self.register_buffer("categorical_offsets", None, persistent=False)
            self.category_embedding = None
        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embedding_features,
                    num_heads,
                    dropout=dropout,
                    bias=bias,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.normalizations = nn.ModuleList(
            [nn.LayerNorm(embedding_features, bias=bias) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(num_fields * embedding_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize continuous feature embeddings."""
        if self.continuous_weight is not None:
            nn.init.xavier_uniform_(self.continuous_weight)

    def forward(
        self,
        continuous: torch.Tensor | None = None,
        categorical: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return predictions after stacked interaction layers."""
        fields = self._fields(continuous, categorical)
        for attention, normalization in zip(
            self.attention_layers, self.normalizations, strict=True
        ):
            interacted, _ = attention(fields, fields, fields, need_weights=False)
            fields = normalization(
                fields + self.dropout(interacted) if self.residual else self.dropout(interacted)
            )
        return self.head(fields.flatten(start_dim=1))

    def _fields(
        self,
        continuous: torch.Tensor | None,
        categorical: torch.Tensor | None,
    ) -> torch.Tensor:
        fields: list[torch.Tensor] = []
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
            fields.append(continuous.unsqueeze(-1) * self.continuous_weight + self.continuous_bias)
            batch_size = continuous.shape[0]
        elif continuous is not None:
            raise ValueError("continuous was supplied but none are configured.")
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
            fields.append(embedding(categorical.long() + offsets.unsqueeze(0)))
        elif categorical is not None:
            raise ValueError("categorical was supplied but none are configured.")
        if not fields:
            raise RuntimeError("No configured input tensor was supplied.")
        return torch.cat(fields, dim=1)
