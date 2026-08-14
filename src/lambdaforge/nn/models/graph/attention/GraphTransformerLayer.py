"""Sparse edge-aware graph Transformer block implemented with native PyTorch."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import torch
from torch import nn

from lambdaforge.nn.activations.base import Activation
from lambdaforge.nn.activations.smooth import GELU
from lambdaforge.nn.models.graph.GraphEdgeData import GraphEdgeData
from lambdaforge.nn.models.graph.GraphEdgeIndex import GraphEdgeIndex
from lambdaforge.nn.models.graph.GraphNormalization import GraphNormalization
from lambdaforge.nn.models.graph.GraphSelfLoopFill import GraphSelfLoopFill
from lambdaforge.nn.models.MLP import MLP
from lambdaforge.nn.models.Scatter import Scatter
from lambdaforge.nn.normalizations.LayerNorm import LayerNorm
from lambdaforge.nn.normalizations.Normalization import Normalization


class GraphTransformerLayer(nn.Module):
    """Apply sparse dot-product attention and a feed-forward graph block.

    Attention is restricted to the supplied directed edges, so computation and
    attention storage remain linear in the edge count rather than materializing
    an all-pairs node matrix. Edge features can modify both keys and values.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 1,
        concatenate_heads: bool = True,
        edge_channels: int = 0,
        feedforward_channels: int | None = None,
        activation: type[Activation] | str = GELU,
        activation_kwargs: dict[str, Any] | None = None,
        normalization: type[Normalization] | str = LayerNorm,
        normalization_kwargs: dict[str, Any] | None = None,
        feature_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        feedforward_dropout: float = 0.0,
        add_self_loops: bool = True,
        self_loop_edge_fill: GraphSelfLoopFill | str | float = GraphSelfLoopFill.ZERO,
        pre_norm: bool = True,
        residual: bool = True,
        beta: bool = False,
        bias: bool = True,
    ) -> None:
        """Build attention, normalization, gating and feed-forward objects."""
        super().__init__()
        for name, value in (
            ("in_channels", in_channels),
            ("out_channels", out_channels),
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
            ("add_self_loops", add_self_loops),
            ("pre_norm", pre_norm),
            ("residual", residual),
            ("beta", beta),
            ("bias", bias),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean.")
        if concatenate_heads and out_channels % num_heads:
            raise ValueError("out_channels must be divisible by num_heads when heads concatenate.")
        hidden = 4 * out_channels if feedforward_channels is None else feedforward_channels
        if isinstance(hidden, bool) or not isinstance(hidden, int):
            raise TypeError("feedforward_channels must be an integer or None.")
        if hidden < 1:
            raise ValueError("feedforward_channels must be positive.")
        for name, probability in (
            ("feature_dropout", feature_dropout),
            ("attention_dropout", attention_dropout),
            ("feedforward_dropout", feedforward_dropout),
        ):
            if isinstance(probability, bool) or not isinstance(probability, Real):
                raise TypeError(f"{name} must be a real number.")
            if not math.isfinite(float(probability)) or not 0.0 <= float(probability) < 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1).")
        if isinstance(self_loop_edge_fill, bool):
            raise TypeError("self_loop_edge_fill must be 'zero', 'mean', or a real number.")
        if isinstance(self_loop_edge_fill, Real):
            normalized_loop_fill: GraphSelfLoopFill | float = float(self_loop_edge_fill)
            if not math.isfinite(normalized_loop_fill):
                raise ValueError("Numeric self_loop_edge_fill must be finite.")
        else:
            normalized_loop_fill = GraphSelfLoopFill(self_loop_edge_fill)
        if beta and not residual:
            raise ValueError("beta gating requires residual=True.")
        feature_dropout_value = float(feature_dropout)
        attention_dropout_value = float(attention_dropout)
        feedforward_dropout_value = float(feedforward_dropout)

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.num_heads = int(num_heads)
        self.concatenate_heads = concatenate_heads
        self.edge_channels = int(edge_channels)
        self.add_self_loops = add_self_loops
        self.self_loop_edge_fill = normalized_loop_fill
        self.pre_norm = pre_norm
        self.residual = residual
        self.beta = beta
        self.head_channels = out_channels // num_heads if concatenate_heads else out_channels
        projected_channels = num_heads * self.head_channels
        self.query_projection = nn.Linear(in_channels, projected_channels, bias=False)
        self.key_projection = nn.Linear(in_channels, projected_channels, bias=False)
        self.value_projection = nn.Linear(in_channels, projected_channels, bias=False)
        self.edge_key_projection = (
            nn.Linear(edge_channels, projected_channels, bias=False) if edge_channels else None
        )
        self.edge_value_projection = (
            nn.Linear(edge_channels, projected_channels, bias=False) if edge_channels else None
        )
        self.output_projection = nn.Linear(out_channels, out_channels, bias=bias)
        self.root_projection = (
            nn.Identity()
            if not residual or in_channels == out_channels
            else nn.Linear(in_channels, out_channels, bias=False)
        )
        self.beta_projection = (
            nn.Linear(3 * out_channels, out_channels, bias=True) if beta else None
        )
        first_width = in_channels if pre_norm else out_channels
        self.first_normalization = GraphNormalization(
            normalization,
            first_width,
            normalization_kwargs,
        )
        self.second_normalization = GraphNormalization(
            normalization,
            out_channels,
            normalization_kwargs,
        )
        self.feedforward = MLP(
            out_channels,
            out_channels,
            hidden=[hidden],
            activation=activation,
            normalization="identity",
            dropout=feedforward_dropout_value,
            activation_kwargs=activation_kwargs,
            bias=bias,
        )
        self.feature_dropout = nn.Dropout(feature_dropout_value)
        self.attention_dropout = nn.Dropout(attention_dropout_value)
        self.feedforward_dropout = nn.Dropout(feedforward_dropout_value)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return transformed node features."""
        output, _, _ = self.forward_with_attention(x, edge_index, edge_features)
        return output

    def forward_with_attention(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return block output, routed edges and pre-dropout attention weights."""
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError("x must be a tensor with shape (N, in_channels).")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"x must have shape (N, {self.in_channels}).")
        if not x.is_floating_point():
            raise TypeError("x must use a floating-point dtype.")
        num_nodes = x.shape[0]
        routed = GraphEdgeIndex.normalize(
            edge_index,
            device=x.device,
            num_nodes=num_nodes,
        )
        features = GraphEdgeData.normalize_features(
            edge_features,
            edge_channels=self.edge_channels,
            edge_count=routed.shape[1],
            reference=x,
        )
        if self.add_self_loops:
            routed, features = GraphEdgeData.replace_self_loops(
                routed,
                device=x.device,
                num_nodes=num_nodes,
                edge_features=features,
                fill=self.self_loop_edge_fill,
            )
        attention_input = self.first_normalization(x) if self.pre_norm else x
        attention_input = self.feature_dropout(attention_input)
        source, destination = routed
        queries = self.query_projection(attention_input).view(
            num_nodes,
            self.num_heads,
            self.head_channels,
        )
        keys = self.key_projection(attention_input).view(
            num_nodes,
            self.num_heads,
            self.head_channels,
        )
        values = self.value_projection(attention_input).view(
            num_nodes,
            self.num_heads,
            self.head_channels,
        )
        routed_keys = keys[source]
        routed_values = values[source]
        if features is not None:
            assert self.edge_key_projection is not None
            assert self.edge_value_projection is not None
            routed_keys = routed_keys + self.edge_key_projection(features).view(
                -1,
                self.num_heads,
                self.head_channels,
            )
            routed_values = routed_values + self.edge_value_projection(features).view(
                -1,
                self.num_heads,
                self.head_channels,
            )
        scores = (queries[destination] * routed_keys).sum(dim=-1) / math.sqrt(self.head_channels)
        attention = Scatter.segment_softmax(scores, destination, num_nodes)
        messages = routed_values * self.attention_dropout(attention).unsqueeze(-1)
        aggregated = Scatter.sum(messages, destination, num_nodes)
        attended = aggregated.flatten(1) if self.concatenate_heads else aggregated.mean(dim=1)
        attended = self.output_projection(attended)
        root = self.root_projection(x)
        hidden = self._combine(attended, root)
        if not self.pre_norm:
            hidden = self.first_normalization(hidden)
        feedforward_input = self.second_normalization(hidden) if self.pre_norm else hidden
        transformed = self.feedforward_dropout(self.feedforward(feedforward_input))
        output = hidden + transformed if self.residual else transformed
        if not self.pre_norm:
            output = self.second_normalization(output)
        return output, routed, attention

    def _combine(self, attended: torch.Tensor, root: torch.Tensor) -> torch.Tensor:
        if not self.residual:
            return attended
        if self.beta_projection is None:
            return root + attended
        gate = torch.sigmoid(
            self.beta_projection(torch.cat((attended, root, attended - root), dim=-1))
        )
        return gate * root + (1.0 - gate) * attended
