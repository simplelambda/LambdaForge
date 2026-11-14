"""Graph Isomorphism Network aggregation layer."""

from __future__ import annotations

import torch
from torch import nn

from lambdaforge.nn.models.graph.GraphEdgeData import GraphEdgeData
from lambdaforge.nn.models.graph.GraphEdgeIndex import GraphEdgeIndex
from lambdaforge.nn.models.MLP import MLP


class GINLayer(nn.Module):
    """Apply sum aggregation followed by a configurable MLP."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mlp_hidden_channels: int | list[int] | None = None,
        epsilon: float = 0.0,
        trainable_epsilon: bool = False,
        activation: str | type = "relu",
        normalization: str | type = "identity",
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError("in_channels and out_channels must be positive.")
        self.in_channels = in_channels
        epsilon_tensor = torch.tensor(float(epsilon))
        if trainable_epsilon:
            self.epsilon = nn.Parameter(epsilon_tensor)
        else:
            self.register_buffer("epsilon", epsilon_tensor, persistent=True)
        hidden = [out_channels] if mlp_hidden_channels is None else mlp_hidden_channels
        self.mlp = MLP(
            in_channels,
            out_channels,
            hidden=hidden,
            activation=activation,
            normalization=normalization,
            dropout=dropout,
            bias=bias,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Aggregate neighbors and transform each updated node."""
        return self.update(x, self.aggregate(self.messages(x, edge_index), edge_index, len(x)))

    def messages(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return aligned sender states (E,in_channels), without a root or update."""
        if x.ndim != 2:
            raise ValueError("x must have shape (N, F).")
        num_nodes = x.shape[0]
        edge_index = GraphEdgeIndex.normalize(
            edge_index,
            device=x.device,
            num_nodes=num_nodes,
        )
        source, destination = edge_index
        return x[source]

    @staticmethod
    def aggregate(messages: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Sum (E,C) messages per destination, retaining duplicate incidences."""
        return GraphEdgeData.aggregate_messages(messages, edge_index, num_nodes)

    def update(self, x: torch.Tensor, aggregated: torch.Tensor) -> torch.Tensor:
        """Apply one root and one MLP to a caller-composed aggregate (N,in_channels)."""
        if x.ndim != 2 or x.shape[1] != self.in_channels:
            raise ValueError(f"x must have shape (N,{self.in_channels}).")
        neighbors = GraphEdgeData.normalize_features(
            aggregated,
            edge_channels=self.in_channels,
            edge_count=len(x),
            reference=x,
            name="aggregated",
        )
        return self.mlp((1.0 + self.epsilon.to(dtype=x.dtype)) * x + neighbors)
