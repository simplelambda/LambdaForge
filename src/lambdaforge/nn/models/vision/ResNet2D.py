"""Configurable residual network for two-dimensional inputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from lambdaforge.nn.activations.base import Activation
from lambdaforge.nn.activations.rectifiers import ReLU
from lambdaforge.nn.ComponentRegistry import ComponentRegistry
from lambdaforge.nn.models.vision.HierarchicalBackbone2D import HierarchicalBackbone2D
from lambdaforge.nn.models.vision.ResidualBlock2D import ResidualBlock2D
from lambdaforge.nn.normalizations.BatchNorm import BatchNorm
from lambdaforge.nn.normalizations.Normalization import Normalization


class ResNet2D(HierarchicalBackbone2D):
    """Residual CNN with configurable stages, stem and optional classification head."""

    def __init__(
        self,
        in_channels: int,
        out_features: int | None = None,
        stage_channels: Sequence[int] = (64, 128, 256, 512),
        blocks_per_stage: int | Sequence[int] = (2, 2, 2, 2),
        stage_strides: int | Sequence[int] = (1, 2, 2, 2),
        block_kernel_size: int = 3,
        stage_dilations: int | Sequence[int] = 1,
        stage_groups: int | Sequence[int] = 1,
        dropout: float | Sequence[float] = 0.0,
        activation: type[Activation] | str = ReLU,
        activation_kwargs: dict[str, Any] | None = None,
        normalization: type[Normalization] | str = BatchNorm,
        normalization_kwargs: dict[str, Any] | None = None,
        stem_channels: int | None = None,
        stem_kernel_size: int = 7,
        stem_stride: int = 2,
        stem_padding: int | None = None,
        use_stem_pooling: bool = True,
        stem_pool_kernel_size: int = 3,
        stem_pool_stride: int = 2,
        head_dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        channels = list(stage_channels)
        if in_channels < 1 or not channels or any(value < 1 for value in channels):
            raise ValueError("Input and stage channel counts must be positive.")
        if out_features is not None and out_features < 1:
            raise ValueError("out_features must be positive when provided.")
        if min(block_kernel_size, stem_kernel_size, stem_stride) < 1:
            raise ValueError("Kernel sizes and strides must be positive.")
        if stem_kernel_size % 2 == 0:
            raise ValueError("stem_kernel_size must be odd when automatic padding is used.")
        if not 0.0 <= head_dropout < 1.0:
            raise ValueError("head_dropout must be in [0, 1).")
        count = len(channels)
        block_counts = self._expand(blocks_per_stage, count, "blocks_per_stage")
        strides = self._expand(stage_strides, count, "stage_strides")
        dilations = self._expand(stage_dilations, count, "stage_dilations")
        groups = self._expand(stage_groups, count, "stage_groups")
        dropouts = self._expand(dropout, count, "dropout")
        if any(int(value) < 1 for value in [*block_counts, *strides, *dilations, *groups]):
            raise ValueError("Block counts, strides, dilations and groups must be positive.")
        if any(not 0.0 <= float(value) < 1.0 for value in dropouts):
            raise ValueError("dropout values must be in [0, 1).")
        first_channels = stem_channels or channels[0]
        if first_channels < 1:
            raise ValueError("stem_channels must be positive.")
        normalization_cls = ComponentRegistry.resolve_normalization(normalization)
        norm_kwargs = ResidualBlock2D._normalization_options(
            normalization_cls,
            normalization_kwargs,
        )
        activation_cls = ComponentRegistry.resolve_activation(activation)
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                first_channels,
                stem_kernel_size,
                stem_stride,
                stem_kernel_size // 2 if stem_padding is None else stem_padding,
                bias=bias,
            ),
            normalization_cls(first_channels, **norm_kwargs),
            activation_cls(**(activation_kwargs or {})),
            nn.MaxPool2d(stem_pool_kernel_size, stem_pool_stride, stem_pool_kernel_size // 2)
            if use_stem_pooling
            else nn.Identity(),
        )
        stages: list[nn.Module] = []
        current_channels = first_channels
        for stage_index, out_channels in enumerate(channels):
            blocks: list[nn.Module] = []
            for block_index in range(int(block_counts[stage_index])):
                stride = int(strides[stage_index]) if block_index == 0 else 1
                blocks.append(
                    ResidualBlock2D(
                        current_channels,
                        out_channels,
                        stride,
                        block_kernel_size,
                        int(dilations[stage_index]),
                        int(groups[stage_index]),
                        activation,
                        activation_kwargs,
                        normalization,
                        normalization_kwargs,
                        float(dropouts[stage_index]),
                        bias,
                    )
                )
                current_channels = out_channels
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.feature_channels = tuple(channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_dropout = nn.Dropout(head_dropout)
        self.head = (
            nn.Linear(channels[-1], out_features, bias=True)
            if out_features is not None
            else nn.Identity()
        )

    @staticmethod
    def _expand(value: Any, count: int, name: str) -> list[Any]:
        resolved = list(value) if isinstance(value, (list, tuple)) else [value] * count
        if len(resolved) != count:
            raise ValueError(f"{name} must contain exactly {count} values.")
        return resolved

    def forward_feature_maps(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return one NCHW map per residual stage, from fine to coarse."""
        if x.ndim != 4:
            raise ValueError("x must have shape (batch, channels, height, width).")
        output = self.stem(x)
        feature_maps: list[torch.Tensor] = []
        for stage in self.stages:
            output = stage(output)
            feature_maps.append(output)
        return tuple(feature_maps)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the globally pooled final-stage representation."""
        output = self.forward_feature_maps(x)[-1]
        return self.pool(output).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an NCHW image and return logits or pooled embeddings."""
        return self.head(self.head_dropout(self.forward_features(x)))
