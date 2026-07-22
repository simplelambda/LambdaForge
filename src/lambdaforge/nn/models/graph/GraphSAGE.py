"""Configurable GraphSAGE model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from lambdaforge.nn.activations.Activation import Activation
from lambdaforge.nn.activations.ReLU import ReLU
from lambdaforge.nn.ComponentRegistry import ComponentRegistry
from lambdaforge.nn.models.Aggregation import Aggregation
from lambdaforge.nn.models.graph.GraphNormalization import GraphNormalization
from lambdaforge.nn.models.graph.GraphSAGELayer import GraphSAGELayer
from lambdaforge.nn.models.Model import Model
from lambdaforge.nn.normalizations.IdentityNorm import IdentityNorm
from lambdaforge.nn.normalizations.Normalization import Normalization


class GraphSAGE(Model):
    """Inductive neighborhood-aggregation stack with per-layer components."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list[int] | tuple[int, ...] = (),
        aggregation: Aggregation | str = Aggregation.MEAN,
        activation: type[Activation] | str | list[type[Activation] | str] = ReLU,
        normalization: (type[Normalization] | str | list[type[Normalization] | str]) = IdentityNorm,
        dropout: float | list[float] = 0.0,
        activation_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        normalization_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        residual: bool = False,
        root_weight: bool = True,
        project_neighbors: bool = False,
        normalize_output: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        hidden = list(hidden_channels)
        if in_channels < 1 or out_channels < 1 or any(width < 1 for width in hidden):
            raise ValueError("All channel sizes must be positive.")
        count = len(hidden)
        activations = (
            [activation] * count if isinstance(activation, type | str) else list(activation)
        )
        normalizations = (
            [normalization] * count
            if isinstance(normalization, type | str)
            else list(normalization)
        )
        dropouts = [float(dropout)] * count if isinstance(dropout, int | float) else list(dropout)
        activation_options = (
            [{} for _ in range(count)]
            if activation_kwargs is None
            else [activation_kwargs] * count
            if isinstance(activation_kwargs, dict)
            else list(activation_kwargs)
        )
        normalization_options = (
            [{} for _ in range(count)]
            if normalization_kwargs is None
            else [normalization_kwargs] * count
            if isinstance(normalization_kwargs, dict)
            else list(normalization_kwargs)
        )
        for name, values in (
            ("activation", activations),
            ("normalization", normalizations),
            ("dropout", dropouts),
            ("activation_kwargs", activation_options),
            ("normalization_kwargs", normalization_options),
        ):
            if len(values) != count:
                raise ValueError(f"{name} must contain exactly {count} values.")
        if any(not 0.0 <= value < 1.0 for value in dropouts):
            raise ValueError("dropout probabilities must be in [0, 1).")
        widths = [in_channels, *hidden, out_channels]
        self.layers = nn.ModuleList(
            GraphSAGELayer(
                widths[index],
                widths[index + 1],
                aggregation=aggregation,
                root_weight=root_weight,
                project_neighbors=project_neighbors,
                normalize_output=normalize_output and index == len(widths) - 2,
                bias=bias,
            )
            for index in range(len(widths) - 1)
        )
        self.activations = nn.ModuleList(
            ComponentRegistry.resolve_activation(activations[index])(**activation_options[index])
            for index in range(count)
        )
        self.normalizations = nn.ModuleList(
            GraphNormalization(
                normalizations[index],
                hidden[index],
                normalization_options[index],
            )
            for index in range(count)
        )
        self.dropouts = nn.ModuleList(
            nn.Dropout(dropouts[index]) if dropouts[index] else nn.Identity()
            for index in range(count)
        )
        self.residual = bool(residual)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Generate node embeddings for seen or previously unseen graphs."""
        for index, layer in enumerate(self.layers):
            identity = x
            x = layer(x, edge_index)
            if index < len(self.activations):
                x = self.dropouts[index](self.activations[index](self.normalizations[index](x)))
                if self.residual and x.shape == identity.shape:
                    x = x + identity
        return x
