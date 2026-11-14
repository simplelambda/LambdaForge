"""GVP message passing with persistent scalar/vector node states."""

from __future__ import annotations

import torch
from torch import nn

from lambdaforge.nn.models.graph.equivariant.GVP import GVP
from lambdaforge.nn.models.graph.equivariant.VectorDropout import VectorDropout
from lambdaforge.nn.models.graph.validation import aligned, graph
from lambdaforge.nn.models.Scatter import Scatter


class GVPGNNLayer(nn.Module):
    """Compose GVP messages and residual updates on a directed graph.

    edge_vectors (E, edge_vector_channels, 3) are authored polar vectors, not
    inferred coordinates. Scalars and vectors aggregate with the same sum/mean.
    This is a reusable residual GVP-GNN variant, not the original sequence decoder.
    """

    def __init__(
        self,
        scalar_channels: int,
        vector_channels: int,
        edge_channels: int = 0,
        edge_vector_channels: int = 1,
        aggregation: str = "mean",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if aggregation not in {"sum", "mean"}:
            raise ValueError("aggregation must be sum or mean.")
        if (
            isinstance(edge_vector_channels, bool)
            or not isinstance(edge_vector_channels, int)
            or edge_vector_channels < 0
        ):
            raise ValueError("edge_vector_channels must be non-negative integer.")
        # GraphEdgeData validates scalar edge width, including the optional zero case.
        if (
            isinstance(edge_channels, bool)
            or not isinstance(edge_channels, int)
            or edge_channels < 0
        ):
            raise ValueError("edge_channels must be non-negative integer.")
        self.scalar_channels, self.vector_channels = scalar_channels, vector_channels
        self.edge_channels, self.edge_vector_channels = edge_channels, edge_vector_channels
        self.aggregation = aggregation
        self.message = GVP(
            2 * scalar_channels + edge_channels,
            2 * vector_channels + edge_vector_channels,
            scalar_channels,
            vector_channels,
        )
        self.update = GVP(
            2 * scalar_channels, 2 * vector_channels, scalar_channels, vector_channels
        )
        self.scalar_dropout, self.vector_dropout = nn.Dropout(dropout), VectorDropout(dropout)

    def forward(
        self,
        scalars: torch.Tensor,
        vectors: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return residual (N,S), (N,C,3), without changing graph or geometry."""
        edges, features = graph(
            scalars, edge_index, self.scalar_channels, edge_features, self.edge_channels
        )
        vectors = aligned(vectors, (len(scalars), self.vector_channels, 3), scalars, "vectors")
        edge_vectors = aligned(
            edge_vectors, (edges.shape[1], self.edge_vector_channels, 3), scalars, "edge_vectors"
        )
        source, target = edges
        scalar_parts = [scalars[source], scalars[target]]
        if features is not None:
            scalar_parts.append(features)
        messages_s, messages_v = self.message(
            torch.cat(scalar_parts, -1),
            torch.cat((vectors[source], vectors[target], edge_vectors), -2),
        )
        aggregate_s = Scatter.reduce(messages_s, target, len(scalars), self.aggregation)
        aggregate_v = Scatter.reduce(messages_v, target, len(scalars), self.aggregation)
        delta_s, delta_v = self.update(
            torch.cat((scalars, aggregate_s), -1), torch.cat((vectors, aggregate_v), -2)
        )
        return scalars + self.scalar_dropout(delta_s), vectors + self.vector_dropout(delta_v)
