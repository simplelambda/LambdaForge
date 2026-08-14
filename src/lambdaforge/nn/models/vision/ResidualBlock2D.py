"""Basic residual block for two-dimensional convolutional models."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from lambdaforge.nn.activations.base import Activation
from lambdaforge.nn.activations.rectifiers import ReLU
from lambdaforge.nn.ComponentRegistry import ComponentRegistry
from lambdaforge.nn.normalizations.BatchNorm import BatchNorm
from lambdaforge.nn.normalizations.ChannelLayerNorm import ChannelLayerNorm
from lambdaforge.nn.normalizations.InstanceNorm import InstanceNorm
from lambdaforge.nn.normalizations.L2Norm import L2Norm
from lambdaforge.nn.normalizations.LayerNorm import LayerNorm
from lambdaforge.nn.normalizations.Normalization import Normalization
from lambdaforge.nn.normalizations.RMSNorm import RMSNorm
from lambdaforge.nn.normalizations.ScaleNorm import ScaleNorm


class ResidualBlock2D(nn.Module):
    """Two-convolution residual block with an automatic projection shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        groups: int = 1,
        activation: type[Activation] | str = ReLU,
        activation_kwargs: dict[str, Any] | None = None,
        normalization: type[Normalization] | str = BatchNorm,
        normalization_kwargs: dict[str, Any] | None = None,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, stride, kernel_size, dilation, groups) < 1:
            raise ValueError("Channels and convolution parameters must be positive.")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve spatial alignment.")
        if in_channels % groups or out_channels % groups:
            raise ValueError("groups must divide both input and output channels.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        padding = dilation * (kernel_size // 2)
        self.first_convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        self.second_convolution = nn.Conv2d(
            out_channels, out_channels, kernel_size, 1, padding, dilation, groups, bias
        )
        normalization_cls = ComponentRegistry.resolve_normalization(normalization)
        norm_kwargs = self._normalization_options(
            normalization_cls,
            normalization_kwargs,
        )
        self.first_normalization = normalization_cls(out_channels, **norm_kwargs)
        self.second_normalization = normalization_cls(out_channels, **norm_kwargs)
        activation_cls = ComponentRegistry.resolve_activation(activation)
        self.first_activation = activation_cls(**(activation_kwargs or {}))
        self.output_activation = activation_cls(**(activation_kwargs or {}))
        self.dropout = nn.Dropout2d(dropout)
        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            projection_norm_kwargs = self._normalization_options(
                normalization_cls,
                normalization_kwargs,
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=bias),
                normalization_cls(out_channels, **projection_norm_kwargs),
            )
        else:
            self.shortcut = nn.Identity()

    @staticmethod
    def _normalization_options(
        normalization_cls: type[Normalization],
        options: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Adapt generic normalization options to NCHW channel layout."""
        resolved = dict(options or {})
        if normalization_cls in {BatchNorm, InstanceNorm}:
            resolved.setdefault("dim", 2)
        elif normalization_cls is ChannelLayerNorm:
            resolved.setdefault("channel_dim", 1)
        elif normalization_cls in {L2Norm, ScaleNorm}:
            resolved.setdefault("dim", 1)
        elif normalization_cls in {LayerNorm, RMSNorm}:
            raise ValueError(
                "LayerNorm and RMSNorm target trailing dimensions and are unsafe for variable-size "
                "NCHW tensors; use ChannelLayerNorm for channel-wise image normalization."
            )
        return resolved

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block to an NCHW tensor."""
        identity = self.shortcut(x)
        output = self.first_convolution(x)
        output = self.dropout(self.first_activation(self.first_normalization(output)))
        output = self.second_normalization(self.second_convolution(output))
        return self.output_activation(output + identity)
