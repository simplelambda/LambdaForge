"""Dynamic multi-head GATv2 attention for sparse directed graphs."""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as functional

from lambdaforge.nn.models.graph.GraphEdgeData import GraphEdgeData
from lambdaforge.nn.models.graph.GraphEdgeIndex import GraphEdgeIndex
from lambdaforge.nn.models.graph.GraphSelfLoopFill import GraphSelfLoopFill
from lambdaforge.nn.models.Scatter import Scatter


class GATv2Layer(nn.Module):
    """Apply edge-aware dynamic attention to incoming graph neighborhoods.

    For every directed `source -> destination` edge and attention head, the
    unnormalized score is
    `a^T LeakyReLU(W_source x_source + W_destination x_destination + W_edge e)`.
    The edge term is omitted when `edge_channels` is zero. Scores are
    normalized over edges with the same destination, and only the weights used
    for message aggregation receive attention dropout.

    Self-loops replace existing loops instead of duplicating them. Their edge
    features are synthesized through `self_loop_fill`, preserving exact
    alignment between the routed edges, edge features and returned attention.
    """

    in_channels: int
    out_channels_per_head: int
    num_heads: int
    concatenate_heads: bool
    share_weights: bool
    edge_channels: int
    negative_slope: float
    add_self_loops: bool
    self_loop_fill: GraphSelfLoopFill | float
    source_projection: nn.Linear
    destination_projection: nn.Linear | None
    edge_projection: nn.Linear | None
    attention: nn.Parameter
    bias: nn.Parameter | None
    attention_dropout: nn.Dropout

    def __init__(
        self,
        in_channels: int,
        out_channels_per_head: int,
        num_heads: int = 1,
        concatenate_heads: bool = True,
        share_weights: bool = False,
        edge_channels: int = 0,
        negative_slope: float = 0.2,
        attention_dropout: float = 0.0,
        add_self_loops: bool = True,
        self_loop_fill: GraphSelfLoopFill | str | float = GraphSelfLoopFill.ZERO,
        bias: bool = True,
    ) -> None:
        """Initialize one fully configurable GATv2 message-passing layer."""
        super().__init__()
        for name, value in (
            ("in_channels", in_channels),
            ("out_channels_per_head", out_channels_per_head),
            ("num_heads", num_heads),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 1:
                raise ValueError(f"{name} must be positive.")
        if isinstance(edge_channels, bool) or not isinstance(edge_channels, int):
            raise TypeError("edge_channels must be an integer.")
        if edge_channels < 0:
            raise ValueError("edge_channels must be non-negative.")
        for name, value in (
            ("concatenate_heads", concatenate_heads),
            ("share_weights", share_weights),
            ("add_self_loops", add_self_loops),
            ("bias", bias),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean.")
        if isinstance(negative_slope, bool) or not isinstance(negative_slope, Real):
            raise TypeError("negative_slope must be a real number.")
        if not math.isfinite(float(negative_slope)) or negative_slope < 0.0:
            raise ValueError("negative_slope must be finite and non-negative.")
        if isinstance(attention_dropout, bool) or not isinstance(attention_dropout, Real):
            raise TypeError("attention_dropout must be a real number.")
        if not math.isfinite(float(attention_dropout)) or not 0.0 <= attention_dropout < 1.0:
            raise ValueError("attention_dropout must be finite and in [0, 1).")

        self.in_channels = in_channels
        self.out_channels_per_head = out_channels_per_head
        self.num_heads = num_heads
        self.concatenate_heads = concatenate_heads
        self.share_weights = share_weights
        self.edge_channels = edge_channels
        self.negative_slope = float(negative_slope)
        self.add_self_loops = add_self_loops
        self.self_loop_fill = self._normalize_self_loop_fill(self_loop_fill)

        projection_channels = num_heads * out_channels_per_head
        self.source_projection = nn.Linear(in_channels, projection_channels, bias=False)
        self.destination_projection = (
            None if share_weights else nn.Linear(in_channels, projection_channels, bias=False)
        )
        self.edge_projection = (
            nn.Linear(edge_channels, projection_channels, bias=False) if edge_channels else None
        )
        self.attention = nn.Parameter(torch.empty(num_heads, out_channels_per_head))
        output_channels = projection_channels if concatenate_heads else out_channels_per_head
        self.bias = nn.Parameter(torch.zeros(output_channels)) if bias else None
        self.attention_dropout = nn.Dropout(float(attention_dropout))
        self.reset_parameters()

    @property
    def out_channels(self) -> int:
        """Return the feature width produced after combining attention heads."""
        if self.concatenate_heads:
            return self.num_heads * self.out_channels_per_head
        return self.out_channels_per_head

    def reset_parameters(self) -> None:
        """Reset every trainable projection and attention parameter."""
        nn.init.xavier_uniform_(self.source_projection.weight)
        if self.destination_projection is not None:
            nn.init.xavier_uniform_(self.destination_projection.weight)
        if self.edge_projection is not None:
            nn.init.xavier_uniform_(self.edge_projection.weight)
        nn.init.xavier_uniform_(self.attention)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return attention-weighted node embeddings."""
        output, _, _ = self.forward_with_attention(x, edge_index, edge_features)
        return output

    def forward_with_attention(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return embeddings, aligned routed edges and pre-dropout attention.

        The returned attention tensor has shape `(routed_edges, heads)` and is
        normalized independently for every destination and head. It is the
        probability before attention dropout, which makes inspection stable in
        training mode.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor.")
        if x.ndim != 2 or x.shape[1] != self.in_channels:
            raise ValueError(f"x must have shape (N, {self.in_channels}).")
        if not x.is_floating_point():
            raise TypeError("x must contain floating-point node features.")

        num_nodes = x.shape[0]
        routed_edges = GraphEdgeIndex.normalize(
            edge_index,
            device=x.device,
            num_nodes=num_nodes,
        )
        aligned_features = GraphEdgeData.normalize_features(
            edge_features,
            edge_channels=self.edge_channels,
            edge_count=routed_edges.shape[1],
            reference=x,
        )
        if self.add_self_loops:
            routed_edges, aligned_features = GraphEdgeData.replace_self_loops(
                routed_edges,
                device=x.device,
                num_nodes=num_nodes,
                edge_features=aligned_features,
                fill=self.self_loop_fill,
            )

        source, destination = routed_edges
        source_nodes = self.source_projection(x).reshape(
            num_nodes,
            self.num_heads,
            self.out_channels_per_head,
        )
        if self.destination_projection is None:
            destination_nodes = source_nodes
        else:
            destination_nodes = self.destination_projection(x).reshape(
                num_nodes,
                self.num_heads,
                self.out_channels_per_head,
            )

        attention_features = source_nodes[source] + destination_nodes[destination]
        if self.edge_projection is not None:
            if aligned_features is None:
                raise RuntimeError("Validated edge features unexpectedly became unavailable.")
            projected_edges = self.edge_projection(aligned_features).reshape(
                routed_edges.shape[1],
                self.num_heads,
                self.out_channels_per_head,
            )
            attention_features = attention_features + projected_edges
        activated = functional.leaky_relu(
            attention_features,
            negative_slope=self.negative_slope,
        )
        scores = (activated * self.attention).sum(dim=-1)
        attention = Scatter.segment_softmax(scores, destination, num_nodes)
        dropped_attention = self.attention_dropout(attention)
        messages = source_nodes[source] * dropped_attention.unsqueeze(-1)
        aggregated = Scatter.sum(messages, destination, num_nodes)
        output = aggregated.flatten(1) if self.concatenate_heads else aggregated.mean(dim=1)
        if self.bias is not None:
            output = output + self.bias
        return output, routed_edges, attention

    @staticmethod
    def _normalize_self_loop_fill(
        fill: GraphSelfLoopFill | str | float,
    ) -> GraphSelfLoopFill | float:
        if isinstance(fill, bool):
            raise TypeError("self_loop_fill must be 'zero', 'mean', or a real number.")
        if isinstance(fill, Real):
            value = float(fill)
            if not math.isfinite(value):
                raise ValueError("Numeric self_loop_fill must be finite.")
            return value
        return GraphSelfLoopFill(fill)
