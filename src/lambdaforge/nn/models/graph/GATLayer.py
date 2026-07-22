"""Multi-head graph-attention layer."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional

from lambdaforge.nn.models.graph.GraphEdgeIndex import GraphEdgeIndex
from lambdaforge.nn.models.Scatter import Scatter


class GATLayer(nn.Module):
    """Attend over sparse incoming neighborhoods with configurable heads."""

    def __init__(
        self,
        in_channels: int,
        out_channels_per_head: int,
        num_heads: int = 1,
        concatenate_heads: bool = True,
        negative_slope: float = 0.2,
        attention_dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels_per_head < 1 or num_heads < 1:
            raise ValueError("Channel sizes and num_heads must be positive.")
        if negative_slope < 0.0:
            raise ValueError("negative_slope must be non-negative.")
        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1).")
        self.in_channels = in_channels
        self.out_channels_per_head = out_channels_per_head
        self.num_heads = num_heads
        self.concatenate_heads = bool(concatenate_heads)
        self.negative_slope = float(negative_slope)
        self.add_self_loops = bool(add_self_loops)
        self.projection = nn.Linear(in_channels, num_heads * out_channels_per_head, bias=False)
        self.source_attention = nn.Parameter(torch.empty(num_heads, out_channels_per_head))
        self.destination_attention = nn.Parameter(torch.empty(num_heads, out_channels_per_head))
        output_channels = (
            num_heads * out_channels_per_head if concatenate_heads else out_channels_per_head
        )
        self.bias = nn.Parameter(torch.zeros(output_channels)) if bias else None
        self.attention_dropout = nn.Dropout(attention_dropout)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.xavier_uniform_(self.source_attention)
        nn.init.xavier_uniform_(self.destination_attention)

    @property
    def out_channels(self) -> int:
        """Return the actual output width after head aggregation."""
        if self.concatenate_heads:
            return self.num_heads * self.out_channels_per_head
        return self.out_channels_per_head

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return attention-weighted node embeddings."""
        output, _, _ = self.forward_with_attention(x, edge_index)
        return output

    def forward_with_attention(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return embeddings, routed edge indices and normalized head weights."""
        if x.ndim != 2 or x.shape[-1] != self.in_channels:
            raise ValueError(f"x must have shape (N, {self.in_channels}).")
        num_nodes = x.shape[0]
        edge_index = GraphEdgeIndex.normalize(
            edge_index,
            device=x.device,
            num_nodes=num_nodes,
        )
        source, destination = edge_index
        if self.add_self_loops:
            non_self = source != destination
            nodes = torch.arange(num_nodes, device=x.device)
            source = torch.cat((source[non_self], nodes))
            destination = torch.cat((destination[non_self], nodes))
        projected = self.projection(x).view(num_nodes, self.num_heads, self.out_channels_per_head)
        scores = (projected[source] * self.source_attention).sum(dim=-1) + (
            projected[destination] * self.destination_attention
        ).sum(dim=-1)
        scores = functional.leaky_relu(scores, negative_slope=self.negative_slope)
        attention = Scatter.segment_softmax(scores, destination, num_nodes)
        dropped_attention = self.attention_dropout(attention)
        messages = projected[source] * dropped_attention.unsqueeze(-1)
        aggregated = Scatter.sum(messages, destination, num_nodes)
        output = aggregated.flatten(1) if self.concatenate_heads else aggregated.mean(dim=1)
        if self.bias is not None:
            output = output + self.bias
        routed_edges = torch.stack((source, destination))
        return output, routed_edges, attention
