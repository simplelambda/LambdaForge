"""Highly configurable residual multilayer perceptron."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from lambdaforge.nn.activations.base import Activation
from lambdaforge.nn.activations.smooth import GELU
from lambdaforge.nn.models.Model import Model
from lambdaforge.nn.models.tabular.ResidualDenseBlock import ResidualDenseBlock
from lambdaforge.nn.normalizations.LayerNorm import LayerNorm
from lambdaforge.nn.normalizations.Normalization import Normalization


class ResidualMLP(Model):
    """Project vectors into a fixed-width stack of configurable residual blocks."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 256,
        num_blocks: int = 4,
        expansion_factor: float | list[float] = 2.0,
        activation: type[Activation] | str | list[type[Activation] | str] = GELU,
        activation_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        normalization: type[Normalization] | str | list[type[Normalization] | str] = LayerNorm,
        normalization_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        dropout: float | list[float] = 0.0,
        input_dropout: float = 0.0,
        bias: bool = True,
        pre_normalization: bool = True,
        final_normalization: bool = True,
        layer_scale_init: float | None = None,
    ) -> None:
        super().__init__()
        if min(in_features, out_features, hidden_features) < 1 or num_blocks < 0:
            raise ValueError("Feature sizes must be positive and num_blocks non-negative.")
        if not 0.0 <= input_dropout < 1.0:
            raise ValueError("input_dropout must be in [0, 1).")
        expansions = self._expand(expansion_factor, num_blocks, "expansion_factor")
        activations = self._expand(activation, num_blocks, "activation")
        normalizations = self._expand(normalization, num_blocks, "normalization")
        dropouts = self._expand(dropout, num_blocks, "dropout")
        activation_parameters = self._expand_kwargs(
            activation_kwargs, num_blocks, "activation_kwargs"
        )
        normalization_parameters = self._expand_kwargs(
            normalization_kwargs, num_blocks, "normalization_kwargs"
        )
        self.input = nn.Sequential(
            nn.Linear(in_features, hidden_features, bias=bias), nn.Dropout(input_dropout)
        )
        self.blocks = nn.ModuleList(
            ResidualDenseBlock(
                hidden_features,
                float(expansions[index]),
                activations[index],
                activation_parameters[index],
                normalizations[index],
                normalization_parameters[index],
                float(dropouts[index]),
                bias,
                pre_normalization,
                layer_scale_init,
            )
            for index in range(num_blocks)
        )
        self.final_normalization = (
            LayerNorm(hidden_features) if final_normalization else nn.Identity()
        )
        self.output = nn.Linear(hidden_features, out_features, bias=bias)

    @staticmethod
    def _expand(value: Any, count: int, name: str) -> list[Any]:
        resolved = list(value) if isinstance(value, (list, tuple)) else [value] * count
        if len(resolved) != count:
            raise ValueError(f"{name} must contain exactly {count} values.")
        return resolved

    @staticmethod
    def _expand_kwargs(
        value: dict[str, Any] | list[dict[str, Any]] | None, count: int, name: str
    ) -> list[dict[str, Any]]:
        if value is None:
            return [{} for _ in range(count)]
        if isinstance(value, dict):
            return [dict(value) for _ in range(count)]
        resolved = [dict(item) for item in value]
        if len(resolved) != count:
            raise ValueError(f"{name} must contain exactly {count} values.")
        return resolved

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform vectors whose last dimension is ``in_features``."""
        output = self.input(x)
        for block in self.blocks:
            output = block(output)
        return self.output(self.final_normalization(output))
