"""Configurable MobileNetV2-style hierarchical image encoder."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from lambdaforge.nn.activations.base import Activation
from lambdaforge.nn.activations.rectifiers import ReLU6
from lambdaforge.nn.ComponentRegistry import ComponentRegistry
from lambdaforge.nn.models.vision.HierarchicalBackbone2D import HierarchicalBackbone2D
from lambdaforge.nn.models.vision.InvertedResidualBlock2D import InvertedResidualBlock2D
from lambdaforge.nn.models.vision.ResidualBlock2D import ResidualBlock2D
from lambdaforge.nn.normalizations.BatchNorm import BatchNorm
from lambdaforge.nn.normalizations.Normalization import Normalization


class MobileNetV2(HierarchicalBackbone2D):
    """Efficient image backbone built from configurable inverted residual stages."""

    def __init__(
        self,
        in_channels: int,
        out_features: int | None = None,
        stage_channels: Sequence[int] = (16, 24, 32, 64, 96, 160, 320),
        blocks_per_stage: Sequence[int] = (1, 2, 3, 4, 3, 3, 1),
        stage_strides: Sequence[int] = (1, 2, 2, 2, 1, 2, 1),
        expansion_ratios: float | Sequence[float] = (1.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0),
        width_multiplier: float = 1.0,
        round_channels_to: int = 8,
        stem_channels: int = 32,
        stem_stride: int = 2,
        final_channels: int = 1280,
        kernel_size: int = 3,
        activation: type[Activation] | str = ReLU6,
        activation_kwargs: dict[str, Any] | None = None,
        normalization: type[Normalization] | str = BatchNorm,
        normalization_kwargs: dict[str, Any] | None = None,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        head_dropout: float = 0.2,
        bias: bool = False,
    ) -> None:
        super().__init__()
        raw_channels = [int(value) for value in stage_channels]
        depths = [int(value) for value in blocks_per_stage]
        strides = [int(value) for value in stage_strides]
        if in_channels < 1 or not raw_channels or any(value < 1 for value in raw_channels):
            raise ValueError("in_channels and stage_channels must be positive.")
        if len(depths) != len(raw_channels) or any(value < 1 for value in depths):
            raise ValueError("blocks_per_stage must contain one positive count per stage.")
        if len(strides) != len(raw_channels) or any(value not in {1, 2} for value in strides):
            raise ValueError("stage_strides must contain one value (1 or 2) per stage.")
        if width_multiplier <= 0 or round_channels_to < 1:
            raise ValueError("width_multiplier and round_channels_to must be positive.")
        if min(stem_channels, stem_stride, final_channels, kernel_size) < 1:
            raise ValueError("Stem, final and kernel parameters must be positive.")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        if out_features is not None and out_features < 1:
            raise ValueError("out_features must be positive when provided.")
        if not 0.0 <= dropout < 1.0 or not 0.0 <= drop_path_rate < 1.0:
            raise ValueError("dropout and drop_path_rate must be in [0, 1).")
        if not 0.0 <= head_dropout < 1.0:
            raise ValueError("head_dropout must be in [0, 1).")
        expansions = self._expand(expansion_ratios, len(raw_channels), "expansion_ratios")
        if any(value <= 0 for value in expansions):
            raise ValueError("Every expansion ratio must be positive.")

        channels = tuple(
            self._make_divisible(value * width_multiplier, round_channels_to)
            for value in raw_channels
        )
        resolved_stem_channels = self._make_divisible(
            stem_channels * width_multiplier, round_channels_to
        )
        resolved_final_channels = self._make_divisible(
            final_channels * max(1.0, width_multiplier), round_channels_to
        )
        activation_cls = ComponentRegistry.resolve_activation(activation)
        normalization_cls = ComponentRegistry.resolve_normalization(normalization)
        norm_options = ResidualBlock2D._normalization_options(
            normalization_cls, normalization_kwargs
        )
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                resolved_stem_channels,
                kernel_size,
                stem_stride,
                kernel_size // 2,
                bias=bias,
            ),
            normalization_cls(resolved_stem_channels, **norm_options),
            activation_cls(**(activation_kwargs or {})),
        )
        total_blocks = sum(depths)
        drop_paths = torch.linspace(0.0, drop_path_rate, total_blocks).tolist()
        stages: list[nn.Module] = []
        current_channels = resolved_stem_channels
        block_index = 0
        for stage_index, output_channels in enumerate(channels):
            blocks: list[nn.Module] = []
            for depth_index in range(depths[stage_index]):
                blocks.append(
                    InvertedResidualBlock2D(
                        current_channels,
                        output_channels,
                        stride=strides[stage_index] if depth_index == 0 else 1,
                        expansion_ratio=expansions[stage_index],
                        kernel_size=kernel_size,
                        activation=activation,
                        activation_kwargs=activation_kwargs,
                        normalization=normalization,
                        normalization_kwargs=normalization_kwargs,
                        dropout=dropout,
                        drop_path_probability=drop_paths[block_index],
                        bias=bias,
                    )
                )
                current_channels = output_channels
                block_index += 1
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.feature_channels = channels
        self.final_projection = nn.Sequential(
            nn.Conv2d(current_channels, resolved_final_channels, 1, bias=bias),
            normalization_cls(resolved_final_channels, **norm_options),
            activation_cls(**(activation_kwargs or {})),
        )
        self.final_channels = resolved_final_channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_dropout = nn.Dropout(head_dropout)
        self.head = (
            nn.Linear(resolved_final_channels, out_features)
            if out_features is not None
            else nn.Identity()
        )

    @staticmethod
    def _make_divisible(value: float, divisor: int) -> int:
        rounded = max(divisor, int(value + divisor / 2) // divisor * divisor)
        return rounded + divisor if rounded < 0.9 * value else rounded

    @staticmethod
    def _expand(value: float | Sequence[float], count: int, name: str) -> list[float]:
        resolved = (
            [float(item) for item in value]
            if isinstance(value, Sequence)
            else [float(value)] * count
        )
        if len(resolved) != count:
            raise ValueError(f"{name} must contain exactly {count} values.")
        return resolved

    def forward_feature_maps(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return one NCHW feature map per inverted-residual stage."""
        if x.ndim != 4:
            raise ValueError("x must have shape (batch, channels, height, width).")
        output = self.stem(x)
        feature_maps: list[torch.Tensor] = []
        for stage in self.stages:
            output = stage(output)
            feature_maps.append(output)
        return tuple(feature_maps)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled representation after the final pointwise projection."""
        output = self.final_projection(self.forward_feature_maps(x)[-1])
        return self.pool(output).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an image and apply the optional prediction head."""
        return self.head(self.head_dropout(self.forward_features(x)))
