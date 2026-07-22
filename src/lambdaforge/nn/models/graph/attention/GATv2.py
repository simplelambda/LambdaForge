"""Configurable multi-layer Graph Attention Network v2."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import torch
from torch import nn

from lambdaforge.nn.activations.Activation import Activation
from lambdaforge.nn.activations.ELU import ELU
from lambdaforge.nn.ComponentRegistry import ComponentRegistry
from lambdaforge.nn.models.graph.attention.GATv2Layer import GATv2Layer
from lambdaforge.nn.models.graph.GraphNormalization import GraphNormalization
from lambdaforge.nn.models.graph.GraphSelfLoopFill import GraphSelfLoopFill
from lambdaforge.nn.models.Model import Model
from lambdaforge.nn.normalizations.IdentityNorm import IdentityNorm
from lambdaforge.nn.normalizations.Normalization import Normalization


class GATv2(Model):
    """Stack pure-PyTorch dynamic graph-attention layers.

    Scalar layer options are broadcast across the complete stack; sequences
    must contain exactly one value per graph layer. Activations,
    normalizations and their keyword dictionaries describe hidden layers only,
    so the final representation always has exactly `out_channels` features.
    Optional edge features are forwarded unchanged to every layer, where
    self-loop replacement keeps them aligned with the topology.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list[int] | tuple[int, ...] = (),
        heads: int | list[int] | tuple[int, ...] = 1,
        concatenate_heads: bool | list[bool] | tuple[bool, ...] = True,
        share_weights: bool | list[bool] | tuple[bool, ...] = False,
        edge_channels: int = 0,
        activation: (
            type[Activation]
            | str
            | list[type[Activation] | str]
            | tuple[type[Activation] | str, ...]
        ) = ELU,
        normalization: (
            type[Normalization]
            | str
            | list[type[Normalization] | str]
            | tuple[type[Normalization] | str, ...]
        ) = IdentityNorm,
        feature_dropout: float | list[float] | tuple[float, ...] = 0.0,
        attention_dropout: float | list[float] | tuple[float, ...] = 0.0,
        activation_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        normalization_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        negative_slope: float | list[float] | tuple[float, ...] = 0.2,
        add_self_loops: bool | list[bool] | tuple[bool, ...] = True,
        self_loop_fill: (
            GraphSelfLoopFill
            | str
            | float
            | list[GraphSelfLoopFill | str | float]
            | tuple[GraphSelfLoopFill | str | float, ...]
        ) = GraphSelfLoopFill.ZERO,
        residual: bool | list[bool] | tuple[bool, ...] = False,
        bias: bool | list[bool] | tuple[bool, ...] = True,
    ) -> None:
        """Build a GATv2 stack from scalar or per-layer configuration."""
        super().__init__()
        hidden = list(hidden_channels)
        widths = [in_channels, *hidden, out_channels]
        for name, value in (("in_channels", in_channels), ("out_channels", out_channels)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
        if any(isinstance(width, bool) or not isinstance(width, int) for width in hidden):
            raise TypeError("hidden_channels must contain only integers.")
        if any(width < 1 for width in widths):
            raise ValueError("All channel sizes must be positive.")
        if isinstance(edge_channels, bool) or not isinstance(edge_channels, int):
            raise TypeError("edge_channels must be an integer.")
        if edge_channels < 0:
            raise ValueError("edge_channels must be non-negative.")

        layer_count = len(widths) - 1
        hidden_count = len(hidden)
        head_values = [heads] * layer_count if isinstance(heads, int) else list(heads)
        concatenate_values = (
            [concatenate_heads] * layer_count
            if isinstance(concatenate_heads, bool)
            else list(concatenate_heads)
        )
        share_values = (
            [share_weights] * layer_count
            if isinstance(share_weights, bool)
            else list(share_weights)
        )
        if isinstance(feature_dropout, (list, tuple)):
            feature_dropouts = list(feature_dropout)
        elif isinstance(feature_dropout, Real) and not isinstance(feature_dropout, bool):
            feature_dropouts = [float(feature_dropout)] * layer_count
        else:
            raise TypeError("feature_dropout must be a real number or a sequence.")
        if isinstance(attention_dropout, (list, tuple)):
            attention_dropouts = list(attention_dropout)
        elif isinstance(attention_dropout, Real) and not isinstance(attention_dropout, bool):
            attention_dropouts = [float(attention_dropout)] * layer_count
        else:
            raise TypeError("attention_dropout must be a real number or a sequence.")
        if isinstance(negative_slope, (list, tuple)):
            negative_slopes = list(negative_slope)
        elif isinstance(negative_slope, Real) and not isinstance(negative_slope, bool):
            negative_slopes = [float(negative_slope)] * layer_count
        else:
            raise TypeError("negative_slope must be a real number or a sequence.")
        self_loop_values = (
            [add_self_loops] * layer_count
            if isinstance(add_self_loops, bool)
            else list(add_self_loops)
        )
        fill_values = (
            list(self_loop_fill)
            if isinstance(self_loop_fill, (list, tuple))
            else [self_loop_fill] * layer_count
        )
        residual_values = [residual] * layer_count if isinstance(residual, bool) else list(residual)
        bias_values = [bias] * layer_count if isinstance(bias, bool) else list(bias)
        activations = (
            [activation] * hidden_count if isinstance(activation, type | str) else list(activation)
        )
        normalizations = (
            [normalization] * hidden_count
            if isinstance(normalization, type | str)
            else list(normalization)
        )
        activation_options = (
            [{} for _ in range(hidden_count)]
            if activation_kwargs is None
            else [dict(activation_kwargs) for _ in range(hidden_count)]
            if isinstance(activation_kwargs, dict)
            else list(activation_kwargs)
        )
        normalization_options = (
            [{} for _ in range(hidden_count)]
            if normalization_kwargs is None
            else [dict(normalization_kwargs) for _ in range(hidden_count)]
            if isinstance(normalization_kwargs, dict)
            else list(normalization_kwargs)
        )

        for name, values, expected in (
            ("heads", head_values, layer_count),
            ("concatenate_heads", concatenate_values, layer_count),
            ("share_weights", share_values, layer_count),
            ("feature_dropout", feature_dropouts, layer_count),
            ("attention_dropout", attention_dropouts, layer_count),
            ("negative_slope", negative_slopes, layer_count),
            ("add_self_loops", self_loop_values, layer_count),
            ("self_loop_fill", fill_values, layer_count),
            ("residual", residual_values, layer_count),
            ("bias", bias_values, layer_count),
            ("activation", activations, hidden_count),
            ("normalization", normalizations, hidden_count),
            ("activation_kwargs", activation_options, hidden_count),
            ("normalization_kwargs", normalization_options, hidden_count),
        ):
            if len(values) != expected:
                raise ValueError(f"{name} must contain exactly {expected} values.")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in head_values):
            raise TypeError("Every head count must be an integer.")
        if any(value < 1 for value in head_values):
            raise ValueError("Every head count must be positive.")
        for name, values in (
            ("concatenate_heads", concatenate_values),
            ("share_weights", share_values),
            ("add_self_loops", self_loop_values),
            ("residual", residual_values),
            ("bias", bias_values),
        ):
            if any(not isinstance(value, bool) for value in values):
                raise TypeError(f"{name} must contain only booleans.")
        for name, values in (
            ("feature_dropout", feature_dropouts),
            ("attention_dropout", attention_dropouts),
            ("negative_slope", negative_slopes),
        ):
            if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
                raise TypeError(f"{name} must contain only real numbers.")
        feature_dropouts = [float(value) for value in feature_dropouts]
        attention_dropouts = [float(value) for value in attention_dropouts]
        negative_slopes = [float(value) for value in negative_slopes]
        if any(
            not math.isfinite(value)
            for value in feature_dropouts + attention_dropouts + negative_slopes
        ):
            raise ValueError("Dropout probabilities and negative slopes must be finite.")
        if any(not 0.0 <= value < 1.0 for value in feature_dropouts + attention_dropouts):
            raise ValueError("Dropout probabilities must be in [0, 1).")
        if any(value < 0.0 for value in negative_slopes):
            raise ValueError("negative_slope values must be non-negative.")
        if any(not isinstance(options, dict) for options in activation_options):
            raise TypeError("activation_kwargs must contain only mappings.")
        if any(not isinstance(options, dict) for options in normalization_options):
            raise TypeError("normalization_kwargs must contain only mappings.")

        self.layers = nn.ModuleList()
        for index in range(layer_count):
            desired_width = widths[index + 1]
            if concatenate_values[index] and desired_width % head_values[index]:
                raise ValueError(f"Layer {index} output width must be divisible by its head count.")
            per_head = (
                desired_width // head_values[index] if concatenate_values[index] else desired_width
            )
            self.layers.append(
                GATv2Layer(
                    widths[index],
                    per_head,
                    num_heads=head_values[index],
                    concatenate_heads=concatenate_values[index],
                    share_weights=share_values[index],
                    edge_channels=edge_channels,
                    negative_slope=negative_slopes[index],
                    attention_dropout=attention_dropouts[index],
                    add_self_loops=self_loop_values[index],
                    self_loop_fill=fill_values[index],
                    bias=bias_values[index],
                )
            )
        self.feature_dropouts = nn.ModuleList(
            nn.Dropout(float(value)) if value else nn.Identity() for value in feature_dropouts
        )
        self.activations = nn.ModuleList(
            ComponentRegistry.resolve_activation(activations[index])(**activation_options[index])
            for index in range(hidden_count)
        )
        self.normalizations = nn.ModuleList(
            GraphNormalization(
                normalizations[index],
                hidden[index],
                normalization_options[index],
            )
            for index in range(hidden_count)
        )
        self.residual = tuple(residual_values)
        self.edge_channels = edge_channels

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply every graph-attention layer to one shared sparse topology."""
        for index, layer in enumerate(self.layers):
            identity = x
            x = layer(self.feature_dropouts[index](x), edge_index, edge_features)
            if index < len(self.activations):
                x = self.activations[index](self.normalizations[index](x))
            if self.residual[index] and x.shape == identity.shape:
                x = x + identity
        return x
