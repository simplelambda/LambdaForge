"""Learned attention pooling for sparse groups."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from lambdaforge.nn.activations.Activation import Activation
from lambdaforge.nn.activations.Tanh import Tanh
from lambdaforge.nn.ComponentRegistry import ComponentRegistry
from lambdaforge.nn.models.Scatter import Scatter
from lambdaforge.nn.pooling.sparse.SparsePooling import SparsePooling


class SparseAttentionPooling(SparsePooling):
    """Learn a normalized scalar importance for every row within its group."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        activation: type[Activation] | str = Tanh,
        activation_kwargs: dict[str, Any] | None = None,
        temperature: float = 1.0,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_features < 1:
            raise ValueError("in_features must be positive.")
        hidden = in_features if hidden_features is None else hidden_features
        if hidden < 1:
            raise ValueError("hidden_features must be positive.")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        activation_class = ComponentRegistry.resolve_activation(activation)
        self.scorer = nn.Sequential(
            nn.Linear(in_features, hidden, bias=bias),
            activation_class(**(activation_kwargs or {})),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(hidden, 1, bias=bias),
        )
        self.in_features = in_features
        self.temperature = float(temperature)

    def weights(
        self,
        x: torch.Tensor,
        group_index: torch.Tensor,
        num_groups: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Return normalized scalar row weights and the validated group count."""
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(f"x must have shape (N, {self.in_features}).")
        group_index, count = self.validate(x, group_index, num_groups)
        scores = self.scorer(x).squeeze(-1) / self.temperature
        return Scatter.segment_softmax(scores, group_index, count), count

    def forward(
        self,
        x: torch.Tensor,
        group_index: torch.Tensor,
        num_groups: int | None = None,
    ) -> torch.Tensor:
        """Return the learned weighted sum for every group."""
        group_index, count = self.validate(x, group_index, num_groups)
        weights = Scatter.segment_softmax(
            self.scorer(x).squeeze(-1) / self.temperature,
            group_index,
            count,
        )
        return Scatter.sum(x * weights.unsqueeze(-1), group_index, count)
