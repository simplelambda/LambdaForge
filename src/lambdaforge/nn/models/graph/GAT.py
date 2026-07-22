"""Configurable multi-layer Graph Attention Network."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from lambdaforge.nn.activations.Activation import Activation
from lambdaforge.nn.activations.ELU import ELU
from lambdaforge.nn.ComponentRegistry import ComponentRegistry
from lambdaforge.nn.models.graph.GATLayer import GATLayer
from lambdaforge.nn.models.graph.GraphNormalization import GraphNormalization
from lambdaforge.nn.models.Model import Model
from lambdaforge.nn.normalizations.IdentityNorm import IdentityNorm
from lambdaforge.nn.normalizations.Normalization import Normalization


class GAT(Model):
    """Stack sparse multi-head attention without requiring PyG."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list[int] | tuple[int, ...] = (),
        heads: int | list[int] = 1,
        concatenate_heads: bool | list[bool] = True,
        activation: type[Activation] | str | list[type[Activation] | str] = ELU,
        normalization: (type[Normalization] | str | list[type[Normalization] | str]) = IdentityNorm,
        feature_dropout: float | list[float] = 0.0,
        attention_dropout: float | list[float] = 0.0,
        activation_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        normalization_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        negative_slope: float = 0.2,
        add_self_loops: bool = True,
        residual: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        hidden = list(hidden_channels)
        widths = [in_channels, *hidden, out_channels]
        layer_count = len(widths) - 1
        hidden_count = len(hidden)
        if any(width < 1 for width in widths):
            raise ValueError("All channel sizes must be positive.")
        head_values = [heads] * layer_count if isinstance(heads, int) else list(heads)
        concatenate_values = (
            [concatenate_heads] * layer_count
            if isinstance(concatenate_heads, bool)
            else list(concatenate_heads)
        )
        feature_dropouts = (
            [float(feature_dropout)] * layer_count
            if isinstance(feature_dropout, int | float)
            else list(feature_dropout)
        )
        attention_dropouts = (
            [float(attention_dropout)] * layer_count
            if isinstance(attention_dropout, int | float)
            else list(attention_dropout)
        )
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
            else [activation_kwargs] * hidden_count
            if isinstance(activation_kwargs, dict)
            else list(activation_kwargs)
        )
        normalization_options = (
            [{} for _ in range(hidden_count)]
            if normalization_kwargs is None
            else [normalization_kwargs] * hidden_count
            if isinstance(normalization_kwargs, dict)
            else list(normalization_kwargs)
        )
        for name, values, expected in (
            ("heads", head_values, layer_count),
            ("concatenate_heads", concatenate_values, layer_count),
            ("feature_dropout", feature_dropouts, layer_count),
            ("attention_dropout", attention_dropouts, layer_count),
            ("activation", activations, hidden_count),
            ("normalization", normalizations, hidden_count),
            ("activation_kwargs", activation_options, hidden_count),
            ("normalization_kwargs", normalization_options, hidden_count),
        ):
            if len(values) != expected:
                raise ValueError(f"{name} must contain exactly {expected} values.")
        if any(value < 1 for value in head_values):
            raise ValueError("Every head count must be positive.")
        if any(not 0.0 <= value < 1.0 for value in feature_dropouts + attention_dropouts):
            raise ValueError("Dropout probabilities must be in [0, 1).")

        self.layers = nn.ModuleList()
        for index in range(layer_count):
            desired_width = widths[index + 1]
            if concatenate_values[index] and desired_width % head_values[index]:
                raise ValueError(f"Layer {index} output width must be divisible by its head count.")
            per_head = (
                desired_width // head_values[index] if concatenate_values[index] else desired_width
            )
            self.layers.append(
                GATLayer(
                    widths[index],
                    per_head,
                    num_heads=head_values[index],
                    concatenate_heads=concatenate_values[index],
                    negative_slope=negative_slope,
                    attention_dropout=attention_dropouts[index],
                    add_self_loops=add_self_loops,
                    bias=bias,
                )
            )
        self.feature_dropouts = nn.ModuleList(
            nn.Dropout(value) if value else nn.Identity() for value in feature_dropouts
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
        self.residual = bool(residual)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Attend over incoming neighbors at every layer."""
        for index, layer in enumerate(self.layers):
            identity = x
            x = layer(self.feature_dropouts[index](x), edge_index)
            if index < len(self.activations):
                x = self.activations[index](self.normalizations[index](x))
                if self.residual and x.shape == identity.shape:
                    x = x + identity
        return x
