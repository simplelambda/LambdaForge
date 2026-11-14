"""GIN with edge attributes and an explicit per-message ReLU convention."""

from __future__ import annotations

import math

import torch
from torch import nn

from lambdaforge.nn.models.graph.GraphEdgeData import GraphEdgeData
from lambdaforge.nn.models.graph.validation import channels, graph


class GINELayer(nn.Module):
    r"""Compute MLP((1+epsilon)x_i + sum_j ReLU(x_j + E(e_ji))).

    This is the GINEConv convention, not a claim of literal equivalence to all
    Hu et al. pretraining details. No self-loops or residuals are added implicitly.
    Linear edge projections have no bias, so zero attributes add exactly zero.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_channels: int,
        hidden_channels: int | None = None,
        epsilon: float = 0.0,
        trainable_epsilon: bool = False,
    ) -> None:
        super().__init__()
        hidden = out_channels if hidden_channels is None else hidden_channels
        channels(in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden)
        if (
            isinstance(edge_channels, bool)
            or not isinstance(edge_channels, int)
            or edge_channels < 0
        ):
            raise ValueError("edge_channels must be a non-negative integer.")
        if not isinstance(trainable_epsilon, bool) or not math.isfinite(epsilon):
            raise ValueError("epsilon must be finite and trainable_epsilon Boolean.")
        self.in_channels, self.edge_channels = in_channels, edge_channels
        self.edge_projection = (
            nn.Linear(edge_channels, in_channels, bias=False) if edge_channels else None
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden), nn.ReLU(), nn.Linear(hidden, out_channels)
        )
        value = torch.tensor(float(epsilon))
        if trainable_epsilon:
            self.epsilon = nn.Parameter(value)
        else:
            self.register_buffer("epsilon", value)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return (N, out_channels); empty neighbourhoods retain the explicit root term."""
        return self.update(
            x, self.aggregate(self.messages(x, edge_index, edge_features), edge_index, len(x))
        )

    def messages(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return ReLU(x_sender+E(edge)), without root, reduction or final MLP."""
        edges, features = graph(x, edge_index, self.in_channels, edge_features, self.edge_channels)
        messages = x[edges[0]]
        if features is not None:
            assert self.edge_projection is not None
            messages = messages + self.edge_projection(features)
        return messages.relu()

    @staticmethod
    def aggregate(messages: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Sum aligned edge messages; relation scaling belongs after this reduction."""
        return GraphEdgeData.aggregate_messages(messages, edge_index, num_nodes)

    def update(self, x: torch.Tensor, aggregated: torch.Tensor) -> torch.Tensor:
        """Apply the root and node MLP once after composing arbitrary relation aggregates."""
        if x.ndim != 2 or x.shape[1] != self.in_channels:
            raise ValueError(f"x must have shape (N,{self.in_channels}).")
        neighbors = GraphEdgeData.normalize_features(
            aggregated,
            edge_channels=self.in_channels,
            edge_count=len(x),
            reference=x,
            name="aggregated",
        )
        return self.mlp((1 + self.epsilon) * x + neighbors)
