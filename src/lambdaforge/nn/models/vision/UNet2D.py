"""Configurable U-Net for two-dimensional dense prediction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from lambdaforge.nn.activations.base import Activation
from lambdaforge.nn.activations.rectifiers import ReLU
from lambdaforge.nn.ComponentRegistry import ComponentRegistry
from lambdaforge.nn.models.Model import Model
from lambdaforge.nn.models.vision.ResidualBlock2D import ResidualBlock2D
from lambdaforge.nn.normalizations.BatchNorm import BatchNorm
from lambdaforge.nn.normalizations.Normalization import Normalization


class UNet2D(Model):
    """Symmetric encoder-decoder with skip connections and exact-size output."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stage_channels: Sequence[int] = (64, 128, 256, 512),
        blocks_per_stage: int | Sequence[int] = 2,
        bottleneck_channels: int | None = None,
        kernel_size: int = 3,
        downsample_factor: int = 2,
        activation: type[Activation] | str = ReLU,
        activation_kwargs: dict[str, Any] | None = None,
        normalization: type[Normalization] | str = BatchNorm,
        normalization_kwargs: dict[str, Any] | None = None,
        dropout: float = 0.0,
        bias: bool = False,
        head_bias: bool = True,
    ) -> None:
        super().__init__()
        channels = tuple(int(value) for value in stage_channels)
        if min(in_channels, out_channels, kernel_size, downsample_factor) < 1:
            raise ValueError("Channels, kernel_size and downsample_factor must be positive.")
        if not channels or any(value < 1 for value in channels):
            raise ValueError("stage_channels must contain positive values.")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve spatial dimensions.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        block_counts = self._expand(blocks_per_stage, len(channels), "blocks_per_stage")
        if any(value < 1 for value in block_counts):
            raise ValueError("Every blocks_per_stage value must be positive.")
        resolved_bottleneck = bottleneck_channels or channels[-1] * 2
        if resolved_bottleneck < 1:
            raise ValueError("bottleneck_channels must be positive.")
        self.downsample_factor = int(downsample_factor)
        self.encoder_stages = nn.ModuleList()
        current_channels = in_channels
        for output_channels, block_count in zip(channels, block_counts, strict=True):
            self.encoder_stages.append(
                self._convolution_block(
                    current_channels,
                    output_channels,
                    block_count,
                    kernel_size,
                    activation,
                    activation_kwargs,
                    normalization,
                    normalization_kwargs,
                    dropout,
                    bias,
                )
            )
            current_channels = output_channels
        self.pool = nn.MaxPool2d(downsample_factor, downsample_factor)
        self.bottleneck = self._convolution_block(
            channels[-1],
            resolved_bottleneck,
            block_counts[-1],
            kernel_size,
            activation,
            activation_kwargs,
            normalization,
            normalization_kwargs,
            dropout,
            bias,
        )
        self.upsampling_layers = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        current_channels = resolved_bottleneck
        for stage_index in range(len(channels) - 1, -1, -1):
            skip_channels = channels[stage_index]
            self.upsampling_layers.append(
                nn.ConvTranspose2d(
                    current_channels,
                    skip_channels,
                    kernel_size=downsample_factor,
                    stride=downsample_factor,
                    bias=bias,
                )
            )
            self.decoder_stages.append(
                self._convolution_block(
                    2 * skip_channels,
                    skip_channels,
                    block_counts[stage_index],
                    kernel_size,
                    activation,
                    activation_kwargs,
                    normalization,
                    normalization_kwargs,
                    dropout,
                    bias,
                )
            )
            current_channels = skip_channels
        self.head = nn.Conv2d(channels[0], out_channels, 1, bias=head_bias)
        self.feature_channels = channels

    @staticmethod
    def _expand(value: int | Sequence[int], count: int, name: str) -> list[int]:
        resolved = (
            [int(item) for item in value] if isinstance(value, Sequence) else [int(value)] * count
        )
        if len(resolved) != count:
            raise ValueError(f"{name} must contain exactly {count} values.")
        return resolved

    @staticmethod
    def _convolution_block(
        in_channels: int,
        out_channels: int,
        block_count: int,
        kernel_size: int,
        activation: type[Activation] | str,
        activation_kwargs: dict[str, Any] | None,
        normalization: type[Normalization] | str,
        normalization_kwargs: dict[str, Any] | None,
        dropout: float,
        bias: bool,
    ) -> nn.Sequential:
        activation_cls = ComponentRegistry.resolve_activation(activation)
        normalization_cls = ComponentRegistry.resolve_normalization(normalization)
        norm_options = ResidualBlock2D._normalization_options(
            normalization_cls, normalization_kwargs
        )
        layers: list[nn.Module] = []
        current_channels = in_channels
        for _ in range(block_count):
            layers.extend(
                (
                    nn.Conv2d(
                        current_channels,
                        out_channels,
                        kernel_size,
                        padding=kernel_size // 2,
                        bias=bias,
                    ),
                    normalization_cls(out_channels, **norm_options),
                    activation_cls(**(activation_kwargs or {})),
                    nn.Dropout2d(dropout),
                )
            )
            current_channels = out_channels
        return nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final decoder map before the dense prediction head."""
        if x.ndim != 4:
            raise ValueError("x must have shape (batch, channels, height, width).")
        minimum_size = self.downsample_factor ** len(self.encoder_stages)
        if x.shape[-2] < minimum_size or x.shape[-1] < minimum_size:
            raise ValueError(
                f"Image dimensions must be at least {minimum_size} for the configured depth."
            )
        skips: list[torch.Tensor] = []
        output = x
        for stage in self.encoder_stages:
            output = stage(output)
            skips.append(output)
            output = self.pool(output)
        output = self.bottleneck(output)
        for upsample, decoder, skip in zip(
            self.upsampling_layers,
            self.decoder_stages,
            reversed(skips),
            strict=True,
        ):
            output = upsample(output)
            if output.shape[-2:] != skip.shape[-2:]:
                output = F.interpolate(
                    output, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            output = decoder(torch.cat((skip, output), dim=1))
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return dense logits aligned with the input spatial dimensions."""
        return self.head(self.forward_features(x))
