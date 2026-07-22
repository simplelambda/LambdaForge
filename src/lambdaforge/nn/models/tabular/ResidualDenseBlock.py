"""Residual dense block for tabular and vector models."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from lambdaforge.nn.activations.Activation import Activation
from lambdaforge.nn.activations.GELU import GELU
from lambdaforge.nn.ComponentRegistry import ComponentRegistry
from lambdaforge.nn.normalizations.LayerNorm import LayerNorm
from lambdaforge.nn.normalizations.Normalization import Normalization


class ResidualDenseBlock(nn.Module):
    """Pre- or post-normalized two-layer residual feed-forward block."""

    def __init__(
        self,
        features: int,
        expansion_factor: float = 2.0,
        activation: type[Activation] | str = GELU,
        activation_kwargs: dict[str, Any] | None = None,
        normalization: type[Normalization] | str = LayerNorm,
        normalization_kwargs: dict[str, Any] | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        pre_normalization: bool = True,
        layer_scale_init: float | None = None,
    ) -> None:
        super().__init__()
        if features < 1 or expansion_factor <= 0:
            raise ValueError("features and expansion_factor must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        expanded = max(1, round(features * expansion_factor))
        normalization_cls = ComponentRegistry.resolve_normalization(normalization)
        activation_cls = ComponentRegistry.resolve_activation(activation)
        self.normalization = normalization_cls(features, **(normalization_kwargs or {}))
        self.input = nn.Linear(features, expanded, bias=bias)
        self.activation = activation_cls(**(activation_kwargs or {}))
        self.hidden_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(expanded, features, bias=bias)
        self.output_dropout = nn.Dropout(dropout)
        self.pre_normalization = bool(pre_normalization)
        self.layer_scale = (
            nn.Parameter(torch.full((features,), float(layer_scale_init)))
            if layer_scale_init is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform a tensor whose last dimension equals ``features``."""
        residual = x
        output = self.normalization(x) if self.pre_normalization else x
        output = self.input(output)
        output = self.hidden_dropout(self.activation(output))
        output = self.output_dropout(self.output(output))
        if self.layer_scale is not None:
            output = output * self.layer_scale
        output = residual + output
        return output if self.pre_normalization else self.normalization(output)
