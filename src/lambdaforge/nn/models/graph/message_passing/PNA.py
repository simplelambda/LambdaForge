"""Configurable stack of Principal Neighbourhood Aggregation layers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from lambdaforge.nn.activations.Activation import Activation
from lambdaforge.nn.activations.ReLU import ReLU
from lambdaforge.nn.ComponentRegistry import ComponentRegistry
from lambdaforge.nn.models.graph.GraphNormalization import GraphNormalization
from lambdaforge.nn.models.graph.message_passing.DegreeScaler import DegreeScaler
from lambdaforge.nn.models.graph.message_passing.PNAAggregator import PNAAggregator
from lambdaforge.nn.models.graph.message_passing.PNALayer import PNALayer
from lambdaforge.nn.models.Model import Model
from lambdaforge.nn.normalizations.IdentityNorm import IdentityNorm
from lambdaforge.nn.normalizations.Normalization import Normalization


class PNA(Model):
    """Stack PNA layers with per-layer statistics and hidden transforms.

    Degree statistics, message widths, dropout, epsilon, activation kwargs and
    bias may be shared as scalars or supplied as one value per PNA layer.
    Normalization, output activation and shape-safe residual connections are
    applied only between layers, leaving the final node tensor unmodified.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: list[int] | tuple[int, ...] = (),
        aggregators: (PNAAggregator | str | Sequence[PNAAggregator | str]) = (
            PNAAggregator.MEAN,
            PNAAggregator.MIN,
            PNAAggregator.MAX,
            PNAAggregator.STD,
        ),
        scalers: DegreeScaler | str | Sequence[DegreeScaler | str] = (DegreeScaler.IDENTITY,),
        edge_channels: int = 0,
        message_channels: int | None | list[int | None] = None,
        pre_mlp_hidden_channels: int | list[int] | tuple[int, ...] | None = None,
        post_mlp_hidden_channels: int | list[int] | tuple[int, ...] | None = None,
        average_degree: float | list[float] = 1.0,
        average_log_degree: float | list[float] = 1.0,
        epsilon: float | list[float] = 1e-12,
        dropout: float | list[float] = 0.0,
        activation: type[Activation] | str | list[type[Activation] | str] = ReLU,
        normalization: (type[Normalization] | str | list[type[Normalization] | str]) = IdentityNorm,
        activation_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        normalization_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        residual: bool | list[bool] = False,
        bias: bool | list[bool] = True,
        layer_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        hidden = list(hidden_channels)
        widths = [in_channels, *hidden, out_channels]
        if any(isinstance(width, bool) or not isinstance(width, int) for width in widths):
            raise TypeError("All channel sizes must be integers.")
        if any(width < 1 for width in widths):
            raise ValueError("All channel sizes must be positive.")
        layer_count = len(widths) - 1
        hidden_count = len(hidden)

        message_widths = self._expand_optional_integers(
            message_channels,
            layer_count,
            "message_channels",
        )
        average_degrees = self._expand_numeric(
            average_degree,
            layer_count,
            "average_degree",
        )
        average_log_degrees = self._expand_numeric(
            average_log_degree,
            layer_count,
            "average_log_degree",
        )
        epsilons = self._expand_numeric(epsilon, layer_count, "epsilon")
        dropouts = self._expand_numeric(dropout, layer_count, "dropout")
        activations: list[type[Activation] | str] = self._expand_component(
            activation,
            layer_count,
            "activation",
        )
        activation_options = self._expand_kwargs(
            activation_kwargs,
            layer_count,
            "activation_kwargs",
        )
        biases = self._expand_booleans(bias, layer_count, "bias")
        normalizations: list[type[Normalization] | str] = self._expand_component(
            normalization,
            hidden_count,
            "normalization",
        )
        normalization_options = self._expand_kwargs(
            normalization_kwargs,
            hidden_count,
            "normalization_kwargs",
        )
        residuals = self._expand_booleans(residual, hidden_count, "residual")
        layer_options = self._expand_kwargs(layer_kwargs, layer_count, "layer_kwargs")
        reserved = {"in_channels", "out_channels", "edge_channels"}
        for layer_option in layer_options:
            overlap = reserved.intersection(layer_option)
            if overlap:
                names = ", ".join(sorted(overlap))
                raise ValueError(f"layer_kwargs cannot override stack-owned fields: {names}.")

        self.layers = nn.ModuleList()
        for index in range(layer_count):
            options: dict[str, Any] = {
                "aggregators": aggregators,
                "scalers": scalers,
                "message_channels": message_widths[index],
                "pre_mlp_hidden_channels": pre_mlp_hidden_channels,
                "post_mlp_hidden_channels": post_mlp_hidden_channels,
                "average_degree": average_degrees[index],
                "average_log_degree": average_log_degrees[index],
                "epsilon": epsilons[index],
                "dropout": dropouts[index],
                "activation": activations[index],
                "activation_kwargs": activation_options[index],
                "bias": biases[index],
            }
            options.update(layer_options[index])
            self.layers.append(
                PNALayer(
                    widths[index],
                    widths[index + 1],
                    edge_channels=edge_channels,
                    **options,
                )
            )
        self.normalizations = nn.ModuleList(
            GraphNormalization(
                normalizations[index],
                hidden[index],
                normalization_options[index],
            )
            for index in range(hidden_count)
        )
        self.activations = nn.ModuleList(
            ComponentRegistry.resolve_activation(activations[index])(**activation_options[index])
            for index in range(hidden_count)
        )
        self.residuals = tuple(residuals)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the final node tensor after all configured PNA layers."""
        for index, layer in enumerate(self.layers):
            identity = x
            x = layer(x, edge_index, edge_features)
            if index < len(self.normalizations):
                x = self.activations[index](self.normalizations[index](x))
                if self.residuals[index] and x.shape == identity.shape:
                    x = x + identity
        return x

    @staticmethod
    def _expand_numeric(
        value: float | list[float],
        count: int,
        name: str,
    ) -> list[float]:
        values = [value] * count if isinstance(value, int | float) else list(value)
        if len(values) != count:
            raise ValueError(f"{name} must contain exactly {count} values.")
        if any(isinstance(item, bool) or not isinstance(item, int | float) for item in values):
            raise TypeError(f"{name} values must be real numbers.")
        return [float(item) for item in values]

    @staticmethod
    def _expand_optional_integers(
        value: int | None | list[int | None],
        count: int,
        name: str,
    ) -> list[int | None]:
        values = [value] * count if isinstance(value, int) or value is None else list(value)
        if len(values) != count:
            raise ValueError(f"{name} must contain exactly {count} values.")
        if any(
            item is not None and (isinstance(item, bool) or not isinstance(item, int))
            for item in values
        ):
            raise TypeError(f"{name} values must be integers or None.")
        return values

    @staticmethod
    def _expand_component(
        value: Any,
        count: int,
        name: str,
    ) -> list[Any]:
        values = list(value) if isinstance(value, list) else [value] * count
        if len(values) != count:
            raise ValueError(f"{name} must contain exactly {count} values.")
        return values

    @staticmethod
    def _expand_kwargs(
        value: dict[str, Any] | list[dict[str, Any]] | None,
        count: int,
        name: str,
    ) -> list[dict[str, Any]]:
        values: list[dict[str, Any]]
        if value is None:
            values = [{} for _ in range(count)]
        elif isinstance(value, dict):
            values = [dict(value) for _ in range(count)]
        else:
            values = list(value)
        if len(values) != count:
            raise ValueError(f"{name} must contain exactly {count} values.")
        if any(not isinstance(item, dict) for item in values):
            raise TypeError(f"{name} values must be mappings.")
        return [dict(item) for item in values]

    @staticmethod
    def _expand_booleans(
        value: bool | list[bool],
        count: int,
        name: str,
    ) -> list[bool]:
        values = [value] * count if isinstance(value, bool) else list(value)
        if len(values) != count:
            raise ValueError(f"{name} must contain exactly {count} values.")
        if any(not isinstance(item, bool) for item in values):
            raise TypeError(f"{name} values must be booleans.")
        return values
