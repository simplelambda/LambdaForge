"""Joint factorization-machine and deep tabular prediction model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import nn

from lambdaforge.nn.models.Model import Model


class DeepFM(Model):
    """Combine first-order, pairwise factorization and nonlinear interactions."""

    def __init__(
        self,
        num_continuous_features: int,
        categorical_cardinalities: Sequence[int],
        out_features: int = 1,
        embedding_features: int = 16,
        hidden_features: Sequence[int] = (128, 64),
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        cardinalities = tuple(int(value) for value in categorical_cardinalities)
        hidden = tuple(int(value) for value in hidden_features)
        num_fields = num_continuous_features + len(cardinalities)
        if num_continuous_features < 0 or any(value < 1 for value in cardinalities):
            raise ValueError("Feature counts must be non-negative and cardinalities positive.")
        if min(num_fields, out_features, embedding_features) < 1 or any(
            value < 1 for value in hidden
        ):
            raise ValueError("Configured feature dimensions must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.num_continuous_features = num_continuous_features
        self.categorical_cardinalities = cardinalities
        self.continuous_embedding = (
            nn.Parameter(torch.empty(num_continuous_features, embedding_features))
            if num_continuous_features
            else None
        )
        self.continuous_first = (
            nn.Parameter(torch.empty(num_continuous_features, out_features))
            if num_continuous_features
            else None
        )
        if cardinalities:
            offsets = torch.tensor([0, *cardinalities[:-1]], dtype=torch.long).cumsum(0)
            self.register_buffer("categorical_offsets", offsets, persistent=True)
            self.category_embedding: nn.Embedding | None = nn.Embedding(
                sum(cardinalities), embedding_features
            )
            self.category_first: nn.Embedding | None = nn.Embedding(
                sum(cardinalities), out_features
            )
        else:
            self.register_buffer("categorical_offsets", None, persistent=False)
            self.category_embedding = None
            self.category_first = None
        layers: list[nn.Module] = []
        current = num_fields * embedding_features
        for width in hidden:
            layers.extend([nn.Linear(current, width, bias=bias), nn.ReLU(), nn.Dropout(dropout)])
            current = width
        layers.append(nn.Linear(current, out_features, bias=bias))
        self.deep = nn.Sequential(*layers)
        self.factorization_head = nn.Linear(1, out_features, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize continuous field parameters."""
        if self.continuous_embedding is not None:
            nn.init.xavier_uniform_(self.continuous_embedding)
        if self.continuous_first is not None:
            nn.init.zeros_(self.continuous_first)

    def forward(
        self,
        continuous: torch.Tensor | None = None,
        categorical: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict from continuous floats and zero-based categorical indices."""
        fields, first_order = self._terms(continuous, categorical)
        summed = fields.sum(dim=1)
        pairwise = 0.5 * (summed.square() - fields.square().sum(dim=1)).sum(dim=-1, keepdim=True)
        output = (
            first_order + self.factorization_head(pairwise) + self.deep(fields.flatten(start_dim=1))
        )
        return output + self.output_bias if self.output_bias is not None else output

    def _terms(
        self,
        continuous: torch.Tensor | None,
        categorical: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fields: list[torch.Tensor] = []
        first: list[torch.Tensor] = []
        batch_size: int | None = None
        if self.num_continuous_features:
            if continuous is None or continuous.ndim != 2:
                raise ValueError("continuous must have shape (batch, continuous_features).")
            if continuous.shape[1] != self.num_continuous_features:
                raise ValueError("continuous has an incompatible feature dimension.")
            if not torch.is_floating_point(continuous):
                raise TypeError("continuous must use a floating-point dtype.")
            assert self.continuous_embedding is not None
            assert self.continuous_first is not None
            fields.append(continuous.unsqueeze(-1) * self.continuous_embedding)
            first.append((continuous.unsqueeze(-1) * self.continuous_first).sum(dim=1))
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
            offsets = cast(torch.Tensor, self.categorical_offsets)
            indices = categorical.long() + offsets.unsqueeze(0)
            fields.append(cast(nn.Embedding, self.category_embedding)(indices))
            first.append(cast(nn.Embedding, self.category_first)(indices).sum(dim=1))
        elif categorical is not None:
            raise ValueError("categorical was supplied but none are configured.")
        if not fields:
            raise RuntimeError("No configured input tensor was supplied.")
        return torch.cat(fields, dim=1), torch.stack(first, dim=0).sum(dim=0)
