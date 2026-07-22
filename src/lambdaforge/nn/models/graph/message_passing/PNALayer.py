"""Pure-PyTorch Principal Neighbourhood Aggregation layer."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from lambdaforge.nn.activations.Activation import Activation
from lambdaforge.nn.activations.ReLU import ReLU
from lambdaforge.nn.models.graph.GraphEdgeData import GraphEdgeData
from lambdaforge.nn.models.graph.GraphEdgeIndex import GraphEdgeIndex
from lambdaforge.nn.models.graph.message_passing.DegreeScaler import DegreeScaler
from lambdaforge.nn.models.graph.message_passing.PNAAggregator import PNAAggregator
from lambdaforge.nn.models.MLP import MLP


class PNALayer(nn.Module):
    r"""Apply degree-aware multi-aggregation message passing.

    Every directed edge `source -> destination` is transformed from
    `[destination_state, source_state, optional_edge_features]` by the
    configurable pre-MLP. Each requested aggregator is crossed with every
    requested degree scaler. The resulting vectors and the root node state
    are fused by the post-MLP. No lazy parameters are created in `forward`.
    By default both internal MLPs contain one hidden layer at their output
    width; pass an empty hidden-width sequence for a direct linear projection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggregators: (PNAAggregator | str | Sequence[PNAAggregator | str]) = (
            PNAAggregator.MEAN,
            PNAAggregator.MIN,
            PNAAggregator.MAX,
            PNAAggregator.STD,
        ),
        scalers: DegreeScaler | str | Sequence[DegreeScaler | str] = (DegreeScaler.IDENTITY,),
        edge_channels: int = 0,
        message_channels: int | None = None,
        pre_mlp_hidden_channels: int | list[int] | tuple[int, ...] | None = None,
        post_mlp_hidden_channels: int | list[int] | tuple[int, ...] | None = None,
        average_degree: float = 1.0,
        average_log_degree: float = 1.0,
        epsilon: float = 1e-12,
        dropout: float = 0.0,
        activation: type[Activation] | str = ReLU,
        activation_kwargs: dict[str, Any] | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = self._positive_integer(in_channels, "in_channels")
        self.out_channels = self._positive_integer(out_channels, "out_channels")
        if isinstance(edge_channels, bool) or not isinstance(edge_channels, int):
            raise TypeError("edge_channels must be an integer.")
        if edge_channels < 0:
            raise ValueError("edge_channels must be non-negative.")
        self.edge_channels = edge_channels
        self.message_channels = self._positive_integer(
            in_channels if message_channels is None else message_channels,
            "message_channels",
        )
        self.aggregators = PNAAggregator.normalize_many(aggregators)
        self.scalers = DegreeScaler.normalize_many(scalers)
        DegreeScaler.validate_statistics(average_degree, average_log_degree, epsilon)
        self.average_degree = float(average_degree)
        self.average_log_degree = float(average_log_degree)
        self.epsilon = float(epsilon)
        self.dropout_probability = self._dropout_probability(dropout)
        if not isinstance(activation_kwargs, dict | type(None)):
            raise TypeError("activation_kwargs must be a mapping or None.")
        if not isinstance(bias, bool):
            raise TypeError("bias must be a boolean.")

        pre_hidden = (
            [self.message_channels]
            if pre_mlp_hidden_channels is None
            else self._hidden_channels(pre_mlp_hidden_channels, "pre_mlp_hidden_channels")
        )
        post_hidden = (
            [self.out_channels]
            if post_mlp_hidden_channels is None
            else self._hidden_channels(
                post_mlp_hidden_channels,
                "post_mlp_hidden_channels",
            )
        )
        options = dict(activation_kwargs or {})
        self.pre_mlp = MLP(
            2 * self.in_channels + self.edge_channels,
            self.message_channels,
            hidden=pre_hidden,
            activation=activation,
            activation_kwargs=options,
            bias=bias,
        )
        combined_channels = self.message_channels * len(self.aggregators) * len(self.scalers)
        self.post_mlp = MLP(
            self.in_channels + combined_channels,
            self.out_channels,
            hidden=post_hidden,
            activation=activation,
            activation_kwargs=options,
            bias=bias,
        )
        self.dropout = (
            nn.Dropout(self.dropout_probability) if self.dropout_probability else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return node features with shape `(N, out_channels)`."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor.")
        if x.ndim != 2 or x.shape[1] != self.in_channels:
            raise ValueError(f"x must have shape (N, {self.in_channels}).")
        if not x.is_floating_point() or x.is_complex():
            raise TypeError("x must contain real floating-point values.")
        num_nodes = x.shape[0]
        routed_edges = GraphEdgeIndex.normalize(
            edge_index,
            device=x.device,
            num_nodes=num_nodes,
        )
        source, destination = routed_edges
        normalized_edges = GraphEdgeData.normalize_features(
            edge_features,
            edge_channels=self.edge_channels,
            edge_count=source.shape[0],
            reference=x,
        )

        message_parts = [x[destination], x[source]]
        if normalized_edges is not None:
            message_parts.append(normalized_edges)
        messages = self.pre_mlp(torch.cat(message_parts, dim=-1))
        degree = torch.bincount(destination, minlength=num_nodes).to(device=x.device, dtype=x.dtype)
        aggregated_values = [
            aggregator.reduce(messages, destination, num_nodes) for aggregator in self.aggregators
        ]
        combined = []
        for scaler in self.scalers:
            for aggregated in aggregated_values:
                combined.append(
                    scaler.scale(
                        aggregated,
                        degree,
                        average_degree=self.average_degree,
                        average_log_degree=self.average_log_degree,
                        epsilon=self.epsilon,
                    )
                )
        return self.dropout(self.post_mlp(torch.cat((x, *combined), dim=-1)))

    @staticmethod
    def _positive_integer(value: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if value < 1:
            raise ValueError(f"{name} must be positive.")
        return value

    @staticmethod
    def _hidden_channels(
        value: int | list[int] | tuple[int, ...] | None,
        name: str,
    ) -> int | list[int] | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise TypeError(f"{name} must be an integer, a sequence, or None.")
        if isinstance(value, int):
            if value < 0:
                raise ValueError(f"{name} must be non-negative when it is a count.")
            return value
        if not isinstance(value, list | tuple):
            raise TypeError(f"{name} must be an integer, a sequence, or None.")
        normalized = list(value)
        if any(isinstance(width, bool) or not isinstance(width, int) for width in normalized):
            raise TypeError(f"{name} widths must be integers.")
        if any(width < 1 for width in normalized):
            raise ValueError(f"{name} widths must be positive.")
        return normalized

    @staticmethod
    def _dropout_probability(value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError("dropout must be a real number.")
        probability = float(value)
        if not math.isfinite(probability) or not 0.0 <= probability < 1.0:
            raise ValueError("dropout must be finite and in [0, 1).")
        return probability
