"""Dependency-light E(n)-equivariant graph message-passing layer."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import torch
from torch import nn

from lambdaforge.nn.activations.base import Activation
from lambdaforge.nn.activations.smooth import SiLU
from lambdaforge.nn.models.Aggregation import Aggregation
from lambdaforge.nn.models.graph.GraphEdgeData import GraphEdgeData
from lambdaforge.nn.models.graph.GraphEdgeIndex import GraphEdgeIndex
from lambdaforge.nn.models.MLP import MLP
from lambdaforge.nn.models.Scatter import Scatter
from lambdaforge.nn.normalizations.IdentityNorm import IdentityNorm
from lambdaforge.nn.normalizations.Normalization import Normalization


class EGNNLayer(nn.Module):
    r"""Update invariant node features and E(n)-equivariant coordinates.

    Node and edge features are interpreted as geometric scalars. Messages see
    only scalar features and squared pairwise distances. Coordinate updates are
    scalar multiples of relative displacements, so rotations, reflections and
    translations transform the output coordinates equivariantly in any spatial
    dimension. This contract does not claim scale equivariance or support
    higher-order tensor features. Feature dropout is confined to the learned
    message/update branch; an enabled residual always receives the clean input.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_channels: int = 0,
        message_channels: int = 64,
        message_hidden_channels: int | list[int] | tuple[int, ...] | None = None,
        node_hidden_channels: int | list[int] | tuple[int, ...] | None = None,
        coordinate_hidden_channels: int | list[int] | tuple[int, ...] | None = None,
        activation: type[Activation] | str = SiLU,
        activation_kwargs: dict[str, Any] | None = None,
        mlp_normalization: type[Normalization] | str = IdentityNorm,
        mlp_normalization_kwargs: dict[str, Any] | None = None,
        message_aggregation: Aggregation | str = Aggregation.SUM,
        coordinate_aggregation: Aggregation | str = Aggregation.MEAN,
        feature_dropout: float = 0.0,
        message_dropout: float = 0.0,
        update_dropout: float = 0.0,
        residual: bool = True,
        update_coordinates: bool = True,
        normalize_displacements: bool = False,
        distance_epsilon: float = 1e-8,
        distance_scale: float = 1.0,
        coordinate_tanh: bool = False,
        coordinate_scale: float = 1.0,
        attention: bool = False,
        bias: bool = True,
    ) -> None:
        """Build all message, feature and coordinate transformations eagerly."""
        super().__init__()
        for name, integer_value in (
            ("in_channels", in_channels),
            ("out_channels", out_channels),
            ("message_channels", message_channels),
        ):
            if isinstance(integer_value, bool) or not isinstance(integer_value, int):
                raise TypeError(f"{name} must be an integer.")
            if integer_value < 1:
                raise ValueError(f"{name} must be positive.")
        if isinstance(edge_channels, bool) or not isinstance(edge_channels, int):
            raise TypeError("edge_channels must be an integer.")
        if edge_channels < 0:
            raise ValueError("edge_channels must be non-negative.")
        for name, flag in (
            ("residual", residual),
            ("update_coordinates", update_coordinates),
            ("normalize_displacements", normalize_displacements),
            ("coordinate_tanh", coordinate_tanh),
            ("attention", attention),
            ("bias", bias),
        ):
            if not isinstance(flag, bool):
                raise TypeError(f"{name} must be a boolean.")
        for name, probability in (
            ("feature_dropout", feature_dropout),
            ("message_dropout", message_dropout),
            ("update_dropout", update_dropout),
        ):
            if isinstance(probability, bool) or not isinstance(probability, Real):
                raise TypeError(f"{name} must be a real number.")
            if not math.isfinite(float(probability)) or not 0.0 <= float(probability) < 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1).")
        for name, scale in (
            ("distance_epsilon", distance_epsilon),
            ("distance_scale", distance_scale),
            ("coordinate_scale", coordinate_scale),
        ):
            if isinstance(scale, bool) or not isinstance(scale, Real):
                raise TypeError(f"{name} must be a real number.")
        if not math.isfinite(float(distance_epsilon)) or distance_epsilon <= 0.0:
            raise ValueError("distance_epsilon must be positive and finite.")
        if not math.isfinite(float(distance_scale)) or distance_scale <= 0.0:
            raise ValueError("distance_scale must be positive and finite.")
        if not math.isfinite(float(coordinate_scale)):
            raise ValueError("coordinate_scale must be finite.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.edge_channels = int(edge_channels)
        self.message_channels = int(message_channels)
        self.message_aggregation = Aggregation(message_aggregation)
        self.coordinate_aggregation = Aggregation(coordinate_aggregation)
        if self.coordinate_aggregation not in {Aggregation.SUM, Aggregation.MEAN}:
            raise ValueError("coordinate_aggregation must be 'sum' or 'mean'.")
        self.update_coordinates = update_coordinates
        self.normalize_displacements = normalize_displacements
        self.distance_epsilon = float(distance_epsilon)
        self.distance_scale_squared = float(distance_scale) ** 2
        self.coordinate_tanh = coordinate_tanh
        self.coordinate_scale = float(coordinate_scale)
        feature_dropout_value = float(feature_dropout)
        message_dropout_value = float(message_dropout)
        update_dropout_value = float(update_dropout)

        message_hidden = self._hidden(message_hidden_channels, message_channels)
        node_hidden = self._hidden(node_hidden_channels, message_channels)
        coordinate_hidden = self._hidden(coordinate_hidden_channels, message_channels)
        shared_mlp_options: dict[str, Any] = {
            "activation": activation,
            "normalization": mlp_normalization,
            "activation_kwargs": activation_kwargs,
            "normalization_kwargs": mlp_normalization_kwargs,
            "bias": bias,
        }
        self.message_mlp = MLP(
            2 * in_channels + 1 + edge_channels,
            message_channels,
            hidden=message_hidden,
            dropout=message_dropout_value,
            **shared_mlp_options,
        )
        self.node_mlp = MLP(
            in_channels + message_channels,
            out_channels,
            hidden=node_hidden,
            dropout=update_dropout_value,
            **shared_mlp_options,
        )
        self.coordinate_mlp = (
            MLP(
                message_channels,
                1,
                hidden=coordinate_hidden,
                dropout=message_dropout_value,
                **shared_mlp_options,
            )
            if update_coordinates
            else None
        )
        self.attention_mlp = MLP(message_channels, 1, hidden=None, bias=bias) if attention else None
        self.feature_dropout = nn.Dropout(feature_dropout_value)
        self.update_dropout = nn.Dropout(update_dropout_value)
        self.residual_projection = (
            None
            if not residual
            else nn.Identity()
            if in_channels == out_channels
            else nn.Linear(in_channels, out_channels, bias=False)
        )
        self.residual = residual

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        coordinates: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return updated invariant features and equivariant coordinates."""
        self._validate_inputs(x, coordinates)
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
        source, destination = routed
        displacement = coordinates[destination] - coordinates[source]
        squared_distance = displacement.square().sum(dim=-1, keepdim=True)
        distance_input = squared_distance / self.distance_scale_squared
        branch_features = self.feature_dropout(x)
        message_parts = [
            branch_features[destination],
            branch_features[source],
            distance_input,
        ]
        if features is not None:
            message_parts.append(features)
        messages = self.message_mlp(torch.cat(message_parts, dim=-1))
        if self.attention_mlp is not None:
            messages = messages * torch.sigmoid(self.attention_mlp(messages))

        aggregated = Scatter.reduce(
            messages,
            destination,
            num_nodes,
            self.message_aggregation,
        )
        updated_features = self.node_mlp(torch.cat((branch_features, aggregated), dim=-1))
        updated_features = self.update_dropout(updated_features)
        if self.residual:
            if self.residual_projection is None:
                raise RuntimeError("Enabled residual projection is unexpectedly unavailable.")
            updated_features = updated_features + self.residual_projection(x)

        if not self.update_coordinates:
            return updated_features, coordinates
        direction = displacement
        if self.normalize_displacements:
            direction = direction / (squared_distance + self.distance_epsilon).sqrt()
        if self.coordinate_mlp is None:
            raise RuntimeError("Enabled coordinate update is unexpectedly unavailable.")
        coordinate_weight = self.coordinate_mlp(messages)
        if self.coordinate_tanh:
            coordinate_weight = coordinate_weight.tanh()
        coordinate_messages = direction * coordinate_weight * self.coordinate_scale
        coordinate_update = Scatter.reduce(
            coordinate_messages,
            destination,
            num_nodes,
            self.coordinate_aggregation,
        )
        return updated_features, coordinates + coordinate_update

    @staticmethod
    def _hidden(
        specification: int | list[int] | tuple[int, ...] | None,
        default_width: int,
    ) -> list[int]:
        if specification is None:
            return [default_width]
        if isinstance(specification, bool):
            raise TypeError("Hidden channel specifications must contain integers.")
        values = [specification] if isinstance(specification, int) else list(specification)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("Hidden channel specifications must contain integers.")
        if any(value < 1 for value in values):
            raise ValueError("Hidden channel specifications must be positive.")
        return values

    def _validate_inputs(self, x: torch.Tensor, coordinates: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError("x must be a tensor with shape (N, in_channels).")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"x must have shape (N, {self.in_channels}).")
        if not x.is_floating_point():
            raise TypeError("x must use a floating-point dtype.")
        if not isinstance(coordinates, torch.Tensor) or coordinates.ndim != 2:
            raise ValueError("coordinates must be a tensor with shape (N, D).")
        if coordinates.shape[0] != x.shape[0] or coordinates.shape[1] < 1:
            raise ValueError("coordinates must have shape (N, D) with D >= 1.")
        if not coordinates.is_floating_point():
            raise TypeError("coordinates must use a floating-point dtype.")
        if coordinates.device != x.device or coordinates.dtype != x.dtype:
            raise ValueError("coordinates must match the device and dtype of x.")
