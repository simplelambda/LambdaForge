"""Configurable stack of sparse edge-aware graph Transformer blocks."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, cast

import torch
from torch import nn

from lambdaforge.nn.activations.Activation import Activation
from lambdaforge.nn.activations.GELU import GELU
from lambdaforge.nn.models.graph.attention.GraphTransformerLayer import (
    GraphTransformerLayer,
)
from lambdaforge.nn.models.graph.GraphSelfLoopFill import GraphSelfLoopFill
from lambdaforge.nn.models.Model import Model
from lambdaforge.nn.normalizations.LayerNorm import LayerNorm
from lambdaforge.nn.normalizations.Normalization import Normalization


class GraphTransformer(Model):
    """Stack local sparse graph Transformer blocks without PyG or dense adjacency."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list[int] | tuple[int, ...] = (),
        heads: int | list[int] = 1,
        concatenate_heads: bool | list[bool] = True,
        edge_channels: int = 0,
        feedforward_channels: int | list[int] | None = None,
        activation: type[Activation] | str | list[type[Activation] | str] = GELU,
        normalization: (type[Normalization] | str | list[type[Normalization] | str]) = LayerNorm,
        feature_dropout: float | list[float] = 0.0,
        attention_dropout: float | list[float] = 0.0,
        feedforward_dropout: float | list[float] = 0.0,
        activation_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        normalization_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        add_self_loops: bool | list[bool] = True,
        self_loop_edge_fill: (
            GraphSelfLoopFill | str | float | list[GraphSelfLoopFill | str | float]
        ) = GraphSelfLoopFill.ZERO,
        pre_norm: bool | list[bool] = True,
        residual: bool | list[bool] = True,
        beta: bool | list[bool] = False,
        bias: bool | list[bool] = True,
    ) -> None:
        """Expand scalar options exactly once and build every block eagerly."""
        super().__init__()
        hidden = list(hidden_channels)
        widths = [in_channels, *hidden, out_channels]
        if any(isinstance(width, bool) or not isinstance(width, int) for width in widths):
            raise TypeError("All channel sizes must be integers.")
        if any(width < 1 for width in widths):
            raise ValueError("All channel sizes must be positive.")
        if isinstance(edge_channels, bool) or not isinstance(edge_channels, int):
            raise TypeError("edge_channels must be an integer.")
        if edge_channels < 0:
            raise ValueError("edge_channels must be non-negative.")
        layer_count = len(widths) - 1
        head_values = self._expand(heads, layer_count, "heads")
        concatenate_values = self._expand(
            concatenate_heads,
            layer_count,
            "concatenate_heads",
        )
        feedforward_values = (
            [None] * layer_count
            if feedforward_channels is None
            else self._expand(feedforward_channels, layer_count, "feedforward_channels")
        )
        activations = self._expand_component(activation, layer_count, "activation")
        normalizations = self._expand_component(
            normalization,
            layer_count,
            "normalization",
        )
        feature_dropouts = self._expand(feature_dropout, layer_count, "feature_dropout")
        attention_dropouts = self._expand(
            attention_dropout,
            layer_count,
            "attention_dropout",
        )
        feedforward_dropouts = self._expand(
            feedforward_dropout,
            layer_count,
            "feedforward_dropout",
        )
        activation_options = self._expand_options(
            activation_kwargs,
            layer_count,
            "activation_kwargs",
        )
        normalization_options = self._expand_options(
            normalization_kwargs,
            layer_count,
            "normalization_kwargs",
        )
        loop_values = self._expand(add_self_loops, layer_count, "add_self_loops")
        fill_values = self._expand(
            self_loop_edge_fill,
            layer_count,
            "self_loop_edge_fill",
        )
        pre_norm_values = self._expand(pre_norm, layer_count, "pre_norm")
        residual_values = self._expand(residual, layer_count, "residual")
        beta_values = self._expand(beta, layer_count, "beta")
        bias_values = self._expand(bias, layer_count, "bias")

        for name, values in (("heads", head_values), ("feedforward_channels", feedforward_values)):
            if any(
                value is not None and (isinstance(value, bool) or not isinstance(value, int))
                for value in values
            ):
                raise TypeError(f"{name} values must be integers.")
            if any(value is not None and value < 1 for value in values):
                raise ValueError(f"{name} values must be positive.")
        for name, values in (
            ("concatenate_heads", concatenate_values),
            ("add_self_loops", loop_values),
            ("pre_norm", pre_norm_values),
            ("residual", residual_values),
            ("beta", beta_values),
            ("bias", bias_values),
        ):
            if any(not isinstance(value, bool) for value in values):
                raise TypeError(f"{name} values must be Boolean.")
        for name, values in (
            ("feature_dropout", feature_dropouts),
            ("attention_dropout", attention_dropouts),
            ("feedforward_dropout", feedforward_dropouts),
        ):
            if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
                raise TypeError(f"{name} values must be real numbers.")
            if any(
                not math.isfinite(float(value)) or not 0.0 <= float(value) < 1.0 for value in values
            ):
                raise ValueError(f"{name} values must be finite and in [0, 1).")

        self.layers = nn.ModuleList(
            GraphTransformerLayer(
                widths[index],
                widths[index + 1],
                num_heads=int(head_values[index]),
                concatenate_heads=bool(concatenate_values[index]),
                edge_channels=edge_channels,
                feedforward_channels=(
                    None
                    if feedforward_values[index] is None
                    else int(cast(int, feedforward_values[index]))
                ),
                activation=activations[index],
                activation_kwargs=activation_options[index],
                normalization=normalizations[index],
                normalization_kwargs=normalization_options[index],
                feature_dropout=float(feature_dropouts[index]),
                attention_dropout=float(attention_dropouts[index]),
                feedforward_dropout=float(feedforward_dropouts[index]),
                add_self_loops=bool(loop_values[index]),
                self_loop_edge_fill=fill_values[index],
                pre_norm=bool(pre_norm_values[index]),
                residual=bool(residual_values[index]),
                beta=bool(beta_values[index]),
                bias=bool(bias_values[index]),
            )
            for index in range(layer_count)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply every sparse graph Transformer block."""
        for layer in self.layers:
            x = layer(x, edge_index, edge_features)
        return x

    def forward_with_attention(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Return final features and routed attention data for every block."""
        routed_edges: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        for layer in self.layers:
            if not isinstance(layer, GraphTransformerLayer):
                raise TypeError("GraphTransformer layers must be GraphTransformerLayer objects.")
            x, routed, attention = layer.forward_with_attention(
                x,
                edge_index,
                edge_features,
            )
            routed_edges.append(routed)
            weights.append(attention)
        return x, tuple(routed_edges), tuple(weights)

    @staticmethod
    def _expand(value: Any, expected: int, name: str) -> list[Any]:
        values = list(value) if isinstance(value, list | tuple) else [value] * expected
        if len(values) != expected:
            raise ValueError(f"{name} must contain exactly {expected} values.")
        return values

    @staticmethod
    def _expand_component(value: Any, expected: int, name: str) -> list[Any]:
        values = [value] * expected if isinstance(value, type | str) else list(value)
        if len(values) != expected:
            raise ValueError(f"{name} must contain exactly {expected} values.")
        return values

    @staticmethod
    def _expand_options(
        value: dict[str, Any] | list[dict[str, Any]] | None,
        expected: int,
        name: str,
    ) -> list[dict[str, Any]]:
        if value is None:
            return [{} for _ in range(expected)]
        values = [dict(value) for _ in range(expected)] if isinstance(value, dict) else list(value)
        if len(values) != expected or any(not isinstance(item, dict) for item in values):
            raise ValueError(f"{name} must contain exactly {expected} mappings.")
        return [dict(item) for item in values]
