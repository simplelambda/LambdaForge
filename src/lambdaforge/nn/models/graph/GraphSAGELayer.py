"""Inductive GraphSAGE aggregation layer."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional

from lambdaforge.nn.models.Aggregation import Aggregation
from lambdaforge.nn.models.graph.GraphEdgeIndex import GraphEdgeIndex
from lambdaforge.nn.models.Scatter import Scatter


class GraphSAGELayer(nn.Module):
    """Aggregate neighbours and combine them with each destination node."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggregation: Aggregation | str = Aggregation.MEAN,
        root_weight: bool = True,
        project_neighbors: bool = False,
        normalize_output: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError("in_channels and out_channels must be positive.")
        self.aggregation = Aggregation(aggregation)
        self.neighbor_projection = (
            nn.Linear(in_channels, in_channels, bias=True) if project_neighbors else nn.Identity()
        )
        self.neighbor_linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.root_linear = nn.Linear(in_channels, out_channels, bias=False) if root_weight else None
        self.normalize_output = bool(normalize_output)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return inductive node embeddings."""
        if x.ndim != 2:
            raise ValueError("x must have shape (N, F).")
        num_nodes = x.shape[0]
        edge_index = GraphEdgeIndex.normalize(
            edge_index,
            device=x.device,
            num_nodes=num_nodes,
        )
        source, destination = edge_index
        projected = self.neighbor_projection(x)
        neighbors = Scatter.reduce(
            projected[source],
            destination,
            num_nodes,
            self.aggregation,
        )
        output = self.neighbor_linear(neighbors)
        if self.root_linear is not None:
            output = output + self.root_linear(x)
        return functional.normalize(output, p=2.0, dim=-1) if self.normalize_output else output
