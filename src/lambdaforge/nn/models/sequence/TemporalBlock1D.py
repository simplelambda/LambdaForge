"""Residual temporal-convolution block."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from lambdaforge.nn.activations.base import Activation
from lambdaforge.nn.activations.rectifiers import ReLU
from lambdaforge.nn.ComponentRegistry import ComponentRegistry
from lambdaforge.nn.normalizations.BatchNorm import BatchNorm
from lambdaforge.nn.normalizations.IdentityNorm import IdentityNorm
from lambdaforge.nn.normalizations.LayerNorm import LayerNorm
from lambdaforge.nn.normalizations.Normalization import Normalization
from lambdaforge.nn.normalizations.RMSNorm import RMSNorm


class TemporalBlock1D(nn.Module):
    """Two-convolution temporal block with normalization and residual projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
        activation: type[Activation] | str = ReLU,
        activation_kwargs: dict[str, Any] | None = None,
        normalization: type[Normalization] | str = IdentityNorm,
        normalization_kwargs: dict[str, Any] | None = None,
        causal: bool = True,
        residual: bool = True,
        bias: bool = True,
        weight_normalization: bool = False,
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1 or kernel_size < 1 or dilation < 1:
            raise ValueError("Channel counts, kernel_size and dilation must be positive.")
        if not causal and kernel_size % 2 == 0:
            raise ValueError("Non-causal blocks require an odd kernel_size to preserve length.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        padding = (kernel_size - 1) * dilation if causal else ((kernel_size - 1) * dilation) // 2
        first = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias
        )
        second = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias
        )
        if weight_normalization:
            first = nn.utils.parametrizations.weight_norm(first)
            second = nn.utils.parametrizations.weight_norm(second)
        self.convolutions = nn.ModuleList([first, second])
        activation_cls = ComponentRegistry.resolve_activation(activation)
        self.activations = nn.ModuleList(
            [activation_cls(**(activation_kwargs or {})) for _ in range(2)]
        )
        normalization_cls = ComponentRegistry.resolve_normalization(normalization)
        norm_kwargs = dict(normalization_kwargs or {})
        if normalization_cls is BatchNorm:
            norm_kwargs.setdefault("dim", 1)
        self.normalizations = nn.ModuleList(
            [normalization_cls(out_channels, **norm_kwargs) for _ in range(2)]
        )
        self.channel_last_normalization = normalization_cls in {LayerNorm, RMSNorm}
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.use_residual = bool(residual)
        self.causal = bool(causal)
        self.trim = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform a channel-first sequence shaped ``(batch, channels, length)``."""
        identity = self.residual(x)
        output = x
        for convolution, normalization, activation, dropout in zip(
            self.convolutions,
            self.normalizations,
            self.activations,
            self.dropouts,
            strict=True,
        ):
            output = convolution(output)
            if self.causal and self.trim > 0:
                output = output[..., : -self.trim]
            if self.channel_last_normalization:
                output = normalization(output.transpose(1, 2)).transpose(1, 2)
            else:
                output = normalization(output)
            output = dropout(activation(output))
        return output + identity if self.use_residual else output
