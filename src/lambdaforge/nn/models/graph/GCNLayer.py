"""Sparse graph-convolution layer."""

from __future__ import annotations

import torch
from torch import nn

from lambdaforge.nn.models.graph.GraphEdgeIndex import GraphEdgeIndex
from lambdaforge.nn.models.Scatter import Scatter


class GCNLayer(nn.Module):
    r"""Apply a degree-normalized graph convolution.

    For a directed sparse edge list, row zero is the source and row one the
    destination. Edge ``source -> destination`` is weighted by
    :math:``D_{out}(source)^{-1/2} D_{in}(destination)^{-1/2}``. Projection is
    bias-free during message passing; the optional bias is added exactly once
    after aggregation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        add_self_loops: bool = True,
        self_loop_weight: float = 1.0,
        degree_epsilon: float = 1e-12,
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError("in_channels and out_channels must be positive.")
        if self_loop_weight <= 0.0:
            raise ValueError("self_loop_weight must be positive.")
        if degree_epsilon <= 0.0:
            raise ValueError("degree_epsilon must be positive.")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.add_self_loops = bool(add_self_loops)
        self.self_loop_weight = float(self_loop_weight)
        self.degree_epsilon = float(degree_epsilon)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return node embeddings with shape ``(N, out_channels)``."""
        if x.ndim != 2 or x.shape[1] != self.in_channels:
            raise ValueError(f"x must have shape (N, {self.in_channels}).")
        num_nodes = x.shape[0]
        edge_index = GraphEdgeIndex.normalize(
            edge_index,
            device=x.device,
            num_nodes=num_nodes,
        )
        source, destination = edge_index
        weights = x.new_ones(source.shape[0])
        if self.add_self_loops:
            non_self = source != destination
            nodes = torch.arange(num_nodes, device=x.device)
            source = torch.cat((source[non_self], nodes))
            destination = torch.cat((destination[non_self], nodes))
            weights = torch.cat(
                (
                    weights[non_self],
                    x.new_full((num_nodes,), self.self_loop_weight),
                )
            )
        out_degree = (
            Scatter.sum(weights.unsqueeze(-1), source, num_nodes)
            .squeeze(-1)
            .clamp_min(self.degree_epsilon)
        )
        in_degree = (
            Scatter.sum(weights.unsqueeze(-1), destination, num_nodes)
            .squeeze(-1)
            .clamp_min(self.degree_epsilon)
        )
        normalization = weights * out_degree[source].rsqrt() * in_degree[destination].rsqrt()
        transformed = self.linear(x)
        messages = transformed[source] * normalization.unsqueeze(-1)
        output = Scatter.sum(messages, destination, num_nodes)
        return output + self.bias if self.bias is not None else output
