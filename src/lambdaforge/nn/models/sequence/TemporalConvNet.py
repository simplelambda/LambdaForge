"""Configurable temporal convolutional network."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from lambdaforge.nn.activations.Activation import Activation
from lambdaforge.nn.activations.ReLU import ReLU
from lambdaforge.nn.models.Model import Model
from lambdaforge.nn.models.sequence.SequenceOutput import SequenceOutput
from lambdaforge.nn.models.sequence.SequenceOutputMode import SequenceOutputMode
from lambdaforge.nn.models.sequence.TemporalBlock1D import TemporalBlock1D
from lambdaforge.nn.normalizations.IdentityNorm import IdentityNorm
from lambdaforge.nn.normalizations.Normalization import Normalization


class TemporalConvNet(Model):
    """Dilated residual TCN consuming and returning batch-first sequences."""

    def __init__(
        self,
        in_features: int,
        channels: list[int] | tuple[int, ...] = (128, 128, 128),
        out_features: int | None = None,
        kernel_size: int | list[int] = 3,
        dilations: list[int] | None = None,
        dilation_base: int = 2,
        dropout: float | list[float] = 0.0,
        activation: type[Activation] | str | list[type[Activation] | str] = ReLU,
        activation_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        normalization: type[Normalization] | str | list[type[Normalization] | str] = IdentityNorm,
        normalization_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        causal: bool = True,
        residual: bool = True,
        bias: bool = True,
        weight_normalization: bool = False,
        output_mode: SequenceOutputMode | str = SequenceOutputMode.SEQUENCE,
    ) -> None:
        super().__init__()
        if in_features < 1 or not channels or any(value < 1 for value in channels):
            raise ValueError("in_features and every channel count must be positive.")
        if out_features is not None and out_features < 1:
            raise ValueError("out_features must be positive when provided.")
        if dilation_base < 1:
            raise ValueError("dilation_base must be positive.")
        count = len(channels)
        kernels = self._expand(kernel_size, count, "kernel_size")
        resolved_dilations = dilations or [dilation_base**index for index in range(count)]
        if len(resolved_dilations) != count or any(value < 1 for value in resolved_dilations):
            raise ValueError("dilations must contain one positive value per block.")
        dropouts = self._expand(dropout, count, "dropout")
        if any(not 0.0 <= float(value) < 1.0 for value in dropouts):
            raise ValueError("dropout values must be in [0, 1).")
        activations = self._expand(activation, count, "activation")
        normalizations = self._expand(normalization, count, "normalization")
        activation_parameters = self._expand_kwargs(activation_kwargs, count, "activation_kwargs")
        normalization_parameters = self._expand_kwargs(
            normalization_kwargs, count, "normalization_kwargs"
        )
        sizes = [in_features, *channels]
        self.blocks = nn.ModuleList(
            TemporalBlock1D(
                sizes[index],
                sizes[index + 1],
                int(kernels[index]),
                resolved_dilations[index],
                float(dropouts[index]),
                activations[index],
                activation_parameters[index],
                normalizations[index],
                normalization_parameters[index],
                causal,
                residual,
                bias,
                weight_normalization,
            )
            for index in range(count)
        )
        self.output = (
            nn.Linear(channels[-1], out_features, bias=bias)
            if out_features is not None
            else nn.Identity()
        )
        self.output_mode = SequenceOutputMode(output_mode)

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

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Transform ``x`` shaped ``(batch, length, in_features)``."""
        SequenceOutput.lengths(x, padding_mask, None)
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        encoded = x.transpose(1, 2)
        for block in self.blocks:
            encoded = block(encoded)
            if padding_mask is not None:
                encoded = encoded.masked_fill(padding_mask.unsqueeze(1), 0.0)
        encoded = self.output(encoded.transpose(1, 2))
        return SequenceOutput.select(encoded, self.output_mode, padding_mask)
