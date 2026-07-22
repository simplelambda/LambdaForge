"""ConvNeXt residual block for two-dimensional feature maps."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from lambdaforge.nn.activations.Activation import Activation
from lambdaforge.nn.activations.GELU import GELU
from lambdaforge.nn.ComponentRegistry import ComponentRegistry
from lambdaforge.nn.normalizations.ChannelLayerNorm import ChannelLayerNorm
from lambdaforge.nn.regularization.DropPath import DropPath


class ConvNeXtBlock2D(nn.Module):
    """Apply the neural ConvNeXt block to an NCHW feature map.

    The block follows the ConvNeXt ordering: depthwise convolution, channel-last
    LayerNorm, expanding pointwise linear layer, activation, projecting pointwise
    linear layer, optional layer scale, stochastic depth and residual addition.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        expansion_ratio: float = 4.0,
        activation: type[Activation] | str = GELU,
        activation_kwargs: dict[str, Any] | None = None,
        dropout: float = 0.0,
        drop_path_probability: float = 0.0,
        layer_scale_init: float | None = 1e-6,
        layer_norm_eps: float = 1e-6,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if channels < 1 or kernel_size < 1 or expansion_ratio <= 0:
            raise ValueError("channels, kernel_size and expansion_ratio must be positive.")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve spatial dimensions.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if not 0.0 <= drop_path_probability < 1.0:
            raise ValueError("drop_path_probability must be in [0, 1).")
        if layer_norm_eps <= 0:
            raise ValueError("layer_norm_eps must be positive.")
        hidden_channels = max(1, round(channels * expansion_ratio))
        self.depthwise_convolution = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=bias,
        )
        self.normalization = ChannelLayerNorm(
            channels,
            eps=layer_norm_eps,
            bias=bias,
            channel_dim=-1,
        )
        self.input_projection = nn.Linear(channels, hidden_channels, bias=bias)
        activation_cls = ComponentRegistry.resolve_activation(activation)
        self.activation = activation_cls(**(activation_kwargs or {}))
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_channels, channels, bias=bias)
        self.layer_scale = (
            nn.Parameter(torch.full((channels,), float(layer_scale_init)))
            if layer_scale_init is not None
            else None
        )
        self.drop_path = DropPath(drop_path_probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform an NCHW tensor while preserving its shape."""
        if x.ndim != 4:
            raise ValueError("x must have shape (batch, channels, height, width).")
        output = self.depthwise_convolution(x)
        output = output.permute(0, 2, 3, 1)
        output = self.normalization(output)
        output = self.input_projection(output)
        output = self.dropout(self.activation(output))
        output = self.output_projection(output)
        if self.layer_scale is not None:
            output = output * self.layer_scale
        output = output.permute(0, 3, 1, 2)
        return x + self.drop_path(output)
