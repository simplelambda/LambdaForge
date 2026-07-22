"""Mobile inverted residual block for two-dimensional feature maps."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from lambdaforge.nn.activations.Activation import Activation
from lambdaforge.nn.activations.ReLU6 import ReLU6
from lambdaforge.nn.ComponentRegistry import ComponentRegistry
from lambdaforge.nn.models.vision.ResidualBlock2D import ResidualBlock2D
from lambdaforge.nn.normalizations.BatchNorm import BatchNorm
from lambdaforge.nn.normalizations.Normalization import Normalization
from lambdaforge.nn.regularization.DropPath import DropPath


class InvertedResidualBlock2D(nn.Module):
    """Expand, depthwise-filter and project an NCHW tensor."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion_ratio: float = 6.0,
        kernel_size: int = 3,
        activation: type[Activation] | str = ReLU6,
        activation_kwargs: dict[str, Any] | None = None,
        normalization: type[Normalization] | str = BatchNorm,
        normalization_kwargs: dict[str, Any] | None = None,
        dropout: float = 0.0,
        drop_path_probability: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, stride, kernel_size) < 1 or expansion_ratio <= 0:
            raise ValueError("Channels, stride, kernel_size and expansion_ratio must be positive.")
        if stride not in {1, 2}:
            raise ValueError("stride must be 1 or 2.")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        if not 0.0 <= dropout < 1.0 or not 0.0 <= drop_path_probability < 1.0:
            raise ValueError("Dropout and drop-path probabilities must be in [0, 1).")
        hidden_channels = max(1, round(in_channels * expansion_ratio))
        activation_cls = ComponentRegistry.resolve_activation(activation)
        normalization_cls = ComponentRegistry.resolve_normalization(normalization)
        norm_options = ResidualBlock2D._normalization_options(
            normalization_cls, normalization_kwargs
        )
        layers: list[nn.Module] = []
        if hidden_channels != in_channels:
            layers.extend(
                (
                    nn.Conv2d(in_channels, hidden_channels, 1, bias=bias),
                    normalization_cls(hidden_channels, **norm_options),
                    activation_cls(**(activation_kwargs or {})),
                )
            )
        self.depthwise_convolution = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size,
            stride,
            kernel_size // 2,
            groups=hidden_channels,
            bias=bias,
        )
        layers.extend(
            (
                self.depthwise_convolution,
                normalization_cls(hidden_channels, **norm_options),
                activation_cls(**(activation_kwargs or {})),
                nn.Dropout2d(dropout),
                nn.Conv2d(hidden_channels, out_channels, 1, bias=bias),
                normalization_cls(out_channels, **norm_options),
            )
        )
        self.layers = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels
        self.drop_path = DropPath(drop_path_probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the mobile block and its shape-safe residual path."""
        if x.ndim != 4:
            raise ValueError("x must have shape (batch, channels, height, width).")
        output = self.layers(x)
        return x + self.drop_path(output) if self.use_residual else output
