"""Configurable stack of dependency-light E(n)-equivariant graph layers."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import torch
from torch import nn

from lambdaforge.nn.activations.base import Activation
from lambdaforge.nn.activations.smooth import SiLU
from lambdaforge.nn.ComponentRegistry import ComponentRegistry
from lambdaforge.nn.models.graph.equivariant.EGNNLayer import EGNNLayer
from lambdaforge.nn.models.graph.equivariant.EquivariantOutputMode import (
    EquivariantOutputMode,
)
from lambdaforge.nn.models.graph.GraphNormalization import GraphNormalization
from lambdaforge.nn.models.Model import Model
from lambdaforge.nn.normalizations.IdentityNorm import IdentityNorm
from lambdaforge.nn.normalizations.Normalization import Normalization


class EGNN(Model):
    """Stack EGNN layers behind one feature-compatible public forward method.

    The default output is a node-feature tensor, so the model composes directly
    with `GraphReadout`. Use `forward_with_coordinates` when downstream code
    needs the equivariantly updated geometry, or select mapping output for a
    multi-target training task.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list[int] | tuple[int, ...] = (),
        edge_channels: int = 0,
        message_channels: int | list[int] = 64,
        activation: type[Activation] | str | list[type[Activation] | str] = SiLU,
        normalization: (type[Normalization] | str | list[type[Normalization] | str]) = IdentityNorm,
        feature_dropout: float | list[float] = 0.0,
        activation_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        normalization_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        residual: bool | list[bool] = True,
        bias: bool | list[bool] = True,
        layer_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        output_mode: EquivariantOutputMode | str = EquivariantOutputMode.FEATURES,
        feature_output_key: str = "node_features",
        coordinate_output_key: str = "coordinates",
    ) -> None:
        """Build a fully eager, per-layer configurable equivariant stack."""
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
        hidden_count = len(hidden)

        message_values = self._expand(message_channels, layer_count, "message_channels")
        dropout_values = self._expand(feature_dropout, layer_count, "feature_dropout")
        residual_values = self._expand(residual, layer_count, "residual")
        bias_values = self._expand(bias, layer_count, "bias")
        layer_options = self._expand_options(layer_kwargs, layer_count, "layer_kwargs")
        for value in message_values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("message_channels values must be integers.")
            if value < 1:
                raise ValueError("message_channels values must be positive.")
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in dropout_values):
            raise TypeError("feature_dropout values must be real numbers.")
        if any(
            not math.isfinite(float(value)) or not 0.0 <= float(value) < 1.0
            for value in dropout_values
        ):
            raise ValueError("feature_dropout values must be finite and in [0, 1).")
        if any(not isinstance(value, bool) for value in residual_values):
            raise TypeError("residual values must be Boolean.")
        if any(not isinstance(value, bool) for value in bias_values):
            raise TypeError("bias values must be Boolean.")

        reserved = {
            "in_channels",
            "out_channels",
            "edge_channels",
            "message_channels",
            "feature_dropout",
            "residual",
            "bias",
        }
        for options in layer_options:
            overlap = reserved.intersection(options)
            if overlap:
                names = ", ".join(sorted(overlap))
                raise ValueError(f"layer_kwargs cannot override stack-owned fields: {names}.")
        self.layers = nn.ModuleList(
            EGNNLayer(
                widths[index],
                widths[index + 1],
                edge_channels=edge_channels,
                message_channels=int(message_values[index]),
                feature_dropout=float(dropout_values[index]),
                residual=bool(residual_values[index]),
                bias=bool(bias_values[index]),
                **layer_options[index],
            )
            for index in range(layer_count)
        )

        activations = self._expand_component(activation, hidden_count, "activation")
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

        self.output_mode = EquivariantOutputMode(output_mode)
        if not isinstance(feature_output_key, str) or not feature_output_key:
            raise ValueError("feature_output_key must be a non-empty string.")
        if not isinstance(coordinate_output_key, str) or not coordinate_output_key:
            raise ValueError("coordinate_output_key must be a non-empty string.")
        if feature_output_key == coordinate_output_key:
            raise ValueError("Feature and coordinate output keys must be different.")
        self.feature_output_key = feature_output_key
        self.coordinate_output_key = coordinate_output_key
        self.output_schema = (
            {"output": "Tensor[N, F]"}
            if self.output_mode is EquivariantOutputMode.FEATURES
            else {
                self.feature_output_key: "Tensor[N, F]",
                self.coordinate_output_key: "Tensor[N, D]",
            }
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        coordinates: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Return node features by default or a configured tensor mapping."""
        features, updated_coordinates = self.forward_with_coordinates(
            x,
            edge_index,
            coordinates,
            edge_features,
        )
        if self.output_mode is EquivariantOutputMode.FEATURES:
            return features
        return {
            self.feature_output_key: features,
            self.coordinate_output_key: updated_coordinates,
        }

    def forward_with_coordinates(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        coordinates: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return final invariant features and equivariant coordinates."""
        features = x
        updated_coordinates = coordinates
        for index, layer in enumerate(self.layers):
            features, updated_coordinates = layer(
                features,
                edge_index,
                updated_coordinates,
                edge_features,
            )
            if index < len(self.activations):
                features = self.activations[index](self.normalizations[index](features))
        return features, updated_coordinates

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
