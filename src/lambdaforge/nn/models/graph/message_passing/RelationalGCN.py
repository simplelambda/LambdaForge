"""Configurable stack of relational graph-convolution layers."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from lambdaforge.nn.activations.Activation import Activation
from lambdaforge.nn.activations.ReLU import ReLU
from lambdaforge.nn.ComponentRegistry import ComponentRegistry
from lambdaforge.nn.models.Aggregation import Aggregation
from lambdaforge.nn.models.graph.GraphNormalization import GraphNormalization
from lambdaforge.nn.models.graph.message_passing.RelationalGCNLayer import (
    RelationalGCNLayer,
)
from lambdaforge.nn.models.Model import Model
from lambdaforge.nn.normalizations.IdentityNorm import IdentityNorm
from lambdaforge.nn.normalizations.Normalization import Normalization


class RelationalGCN(Model):
    """Stack relation-aware convolutions behind one sparse graph API.

    Intrinsic layer options accept either one shared value or one value per
    relational layer. Feature dropout is applied before its corresponding
    convolution, and residuals are added only when input and output shapes
    match. As in the other LambdaForge graph stacks, normalization and
    activation apply only to hidden layers; the final node tensor remains
    linear.

    `message_chunk_size` is broadcast or configured per layer. Its finite
    default bounds temporary projected messages; `None` processes each
    relation group as one block without changing the sparse graph contract.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        hidden_channels: list[int] | tuple[int, ...] = (),
        num_bases: int | None | list[int | None] = None,
        aggregation: Aggregation | str | list[Aggregation | str] = Aggregation.MEAN,
        activation: type[Activation] | str | list[type[Activation] | str] = ReLU,
        normalization: (type[Normalization] | str | list[type[Normalization] | str]) = IdentityNorm,
        dropout: float | list[float] = 0.0,
        activation_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        normalization_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        residual: bool | list[bool] = False,
        root_weight: bool | list[bool] = True,
        bias: bool | list[bool] = True,
        message_chunk_size: int | None | list[int | None] = 65_536,
    ) -> None:
        """Build all relation matrices and hidden components eagerly."""
        super().__init__()
        hidden = list(hidden_channels)
        widths = [in_channels, *hidden, out_channels]
        if any(isinstance(width, bool) or not isinstance(width, int) for width in widths):
            raise TypeError("All channel sizes must be integers.")
        if any(width < 1 for width in widths):
            raise ValueError("All channel sizes must be positive.")
        if isinstance(num_relations, bool) or not isinstance(num_relations, int):
            raise TypeError("num_relations must be an integer.")
        if num_relations < 1:
            raise ValueError("num_relations must be positive.")

        layer_count = len(widths) - 1
        hidden_count = len(hidden)
        basis_values = self._expand(num_bases, layer_count, "num_bases")
        chunk_values = self._expand(
            message_chunk_size,
            layer_count,
            "message_chunk_size",
        )
        aggregation_values = self._expand(
            aggregation,
            layer_count,
            "aggregation",
        )
        dropout_values = self._expand(dropout, layer_count, "dropout")
        residual_values = self._expand(residual, layer_count, "residual")
        root_values = self._expand(root_weight, layer_count, "root_weight")
        bias_values = self._expand(bias, layer_count, "bias")

        for value in basis_values:
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("num_bases values must be integers or None.")
            if value < 1:
                raise ValueError("num_bases values must be positive when provided.")
            if value > num_relations:
                raise ValueError("num_bases values cannot exceed num_relations.")
        for value in chunk_values:
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("message_chunk_size values must be integers or None.")
            if value < 1:
                raise ValueError("message_chunk_size values must be positive when provided.")
        aggregation_modes = [Aggregation(value) for value in aggregation_values]
        if any(value not in {Aggregation.SUM, Aggregation.MEAN} for value in aggregation_modes):
            raise ValueError("aggregation values must be 'sum' or 'mean'.")
        for value in dropout_values:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError("dropout values must be real numbers.")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) < 1.0:
                raise ValueError("dropout values must be finite and in [0, 1).")
        for name, values in (
            ("residual", residual_values),
            ("root_weight", root_values),
            ("bias", bias_values),
        ):
            if any(not isinstance(value, bool) for value in values):
                raise TypeError(f"{name} values must be Boolean.")

        activations = self._expand_component(
            activation,
            hidden_count,
            "activation",
        )
        normalizations = self._expand_component(
            normalization,
            hidden_count,
            "normalization",
        )
        activation_options = self._expand_options(
            activation_kwargs,
            hidden_count,
            "activation_kwargs",
        )
        normalization_options = self._expand_options(
            normalization_kwargs,
            hidden_count,
            "normalization_kwargs",
        )

        self.num_relations = num_relations
        self.layers = nn.ModuleList(
            RelationalGCNLayer(
                widths[index],
                widths[index + 1],
                num_relations,
                num_bases=basis_values[index],
                message_chunk_size=chunk_values[index],
                aggregation=aggregation_modes[index],
                root_weight=root_values[index],
                bias=bias_values[index],
            )
            for index in range(layer_count)
        )
        self.dropouts = nn.ModuleList(
            nn.Dropout(float(value)) if float(value) else nn.Identity() for value in dropout_values
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

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
    ) -> torch.Tensor:
        """Return the final node-feature tensor for one relational graph."""
        for index, layer in enumerate(self.layers):
            identity = x
            x = layer(self.dropouts[index](x), edge_index, edge_types)
            if index < len(self.activations):
                x = self.activations[index](self.normalizations[index](x))
            if self.residual[index] and x.shape == identity.shape:
                x = x + identity
        return x

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
