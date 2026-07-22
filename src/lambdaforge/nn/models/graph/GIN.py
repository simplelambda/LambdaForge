"""Configurable Graph Isomorphism Network."""

from __future__ import annotations

import torch
from torch import nn

from lambdaforge.nn.activations.Activation import Activation
from lambdaforge.nn.activations.ReLU import ReLU
from lambdaforge.nn.ComponentRegistry import ComponentRegistry
from lambdaforge.nn.models.graph.GINLayer import GINLayer
from lambdaforge.nn.models.graph.GraphNormalization import GraphNormalization
from lambdaforge.nn.models.Model import Model
from lambdaforge.nn.normalizations.IdentityNorm import IdentityNorm
from lambdaforge.nn.normalizations.Normalization import Normalization


class GIN(Model):
    """Stack expressive sum-aggregation GIN layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list[int] | tuple[int, ...] = (),
        mlp_hidden_channels: int | list[int] | None = None,
        epsilon: float | list[float] = 0.0,
        trainable_epsilon: bool | list[bool] = False,
        activation: type[Activation] | str | list[type[Activation] | str] = ReLU,
        normalization: (type[Normalization] | str | list[type[Normalization] | str]) = IdentityNorm,
        dropout: float | list[float] = 0.0,
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
        mlp_widths = (
            [mlp_hidden_channels] * layer_count
            if isinstance(mlp_hidden_channels, int) or mlp_hidden_channels is None
            else list(mlp_hidden_channels)
        )
        epsilons = (
            [float(epsilon)] * layer_count if isinstance(epsilon, int | float) else list(epsilon)
        )
        trainable = (
            [trainable_epsilon] * layer_count
            if isinstance(trainable_epsilon, bool)
            else list(trainable_epsilon)
        )
        activations = (
            [activation] * hidden_count if isinstance(activation, type | str) else list(activation)
        )
        normalizations = (
            [normalization] * hidden_count
            if isinstance(normalization, type | str)
            else list(normalization)
        )
        dropouts = (
            [float(dropout)] * hidden_count if isinstance(dropout, int | float) else list(dropout)
        )
        for name, values, expected in (
            ("mlp_hidden_channels", mlp_widths, layer_count),
            ("epsilon", epsilons, layer_count),
            ("trainable_epsilon", trainable, layer_count),
            ("activation", activations, hidden_count),
            ("normalization", normalizations, hidden_count),
            ("dropout", dropouts, hidden_count),
        ):
            if len(values) != expected:
                raise ValueError(f"{name} must contain exactly {expected} values.")
        if any(not 0.0 <= value < 1.0 for value in dropouts):
            raise ValueError("dropout probabilities must be in [0, 1).")
        self.layers = nn.ModuleList(
            GINLayer(
                widths[index],
                widths[index + 1],
                mlp_hidden_channels=mlp_widths[index],
                epsilon=epsilons[index],
                trainable_epsilon=trainable[index],
                activation=activations[index] if index < hidden_count else "relu",
                normalization=normalizations[index] if index < hidden_count else "identity",
                dropout=dropouts[index] if index < hidden_count else 0.0,
                bias=bias,
            )
            for index in range(layer_count)
        )
        self.activations = nn.ModuleList(
            ComponentRegistry.resolve_activation(value)() for value in activations
        )
        self.normalizations = nn.ModuleList(
            GraphNormalization(normalizations[index], hidden[index])
            for index in range(hidden_count)
        )
        self.dropouts = nn.ModuleList(
            nn.Dropout(value) if value else nn.Identity() for value in dropouts
        )
        self.residual = bool(residual)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return node embeddings after repeated injective aggregation."""
        for index, layer in enumerate(self.layers):
            identity = x
            x = layer(x, edge_index)
            if index < len(self.activations):
                x = self.dropouts[index](self.activations[index](self.normalizations[index](x)))
                if self.residual and x.shape == identity.shape:
                    x = x + identity
        return x
