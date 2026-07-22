"""Configurable ConvNeXt model for two-dimensional inputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from lambdaforge.nn.activations.Activation import Activation
from lambdaforge.nn.activations.GELU import GELU
from lambdaforge.nn.models.vision.ConvNeXtBlock2D import ConvNeXtBlock2D
from lambdaforge.nn.models.vision.HierarchicalBackbone2D import HierarchicalBackbone2D
from lambdaforge.nn.normalizations.ChannelLayerNorm import ChannelLayerNorm


class ConvNeXt2D(HierarchicalBackbone2D):
    """Hierarchical ConvNeXt encoder with configurable stages and output head.

    Stage channels and blocks per stage define the full architecture.
    Stochastic-depth probabilities can be supplied block-by-block or generated
    linearly from zero to drop_path_rate. Without out_features the model returns
    the final pooled embedding.
    """

    def __init__(
        self,
        in_channels: int,
        out_features: int | None = None,
        stage_channels: Sequence[int] = (96, 192, 384, 768),
        blocks_per_stage: Sequence[int] = (3, 3, 9, 3),
        stem_kernel_size: int = 4,
        stem_stride: int = 4,
        downsample_kernel_size: int = 2,
        downsample_stride: int = 2,
        block_kernel_size: int | Sequence[int] = 7,
        expansion_ratio: float | Sequence[float] = 4.0,
        activation: type[Activation] | str | Sequence[type[Activation] | str] = GELU,
        activation_kwargs: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
        dropout: float | Sequence[float] = 0.0,
        drop_path_rate: float = 0.0,
        drop_path_probabilities: Sequence[float] | None = None,
        layer_scale_init: float | None | Sequence[float | None] = 1e-6,
        layer_norm_eps: float = 1e-6,
        head_dropout: float = 0.0,
        bias: bool = True,
        head_bias: bool = True,
        initialization_std: float = 0.02,
    ) -> None:
        super().__init__()
        channels = [int(value) for value in stage_channels]
        depths = [int(value) for value in blocks_per_stage]
        if in_channels < 1 or not channels or any(value < 1 for value in channels):
            raise ValueError("in_channels and every stage channel count must be positive.")
        if len(depths) != len(channels) or any(value < 1 for value in depths):
            raise ValueError("blocks_per_stage must contain one positive count per stage.")
        if out_features is not None and out_features < 1:
            raise ValueError("out_features must be positive when provided.")
        if min(stem_kernel_size, stem_stride, downsample_kernel_size, downsample_stride) < 1:
            raise ValueError("Stem and downsampling kernel/stride values must be positive.")
        if layer_norm_eps <= 0 or initialization_std <= 0:
            raise ValueError("layer_norm_eps and initialization_std must be positive.")
        if not 0.0 <= head_dropout < 1.0 or not 0.0 <= drop_path_rate < 1.0:
            raise ValueError("Dropout and drop-path probabilities must be in [0, 1).")

        stage_count = len(channels)
        kernels = self._expand(block_kernel_size, stage_count, "block_kernel_size")
        expansions = self._expand(expansion_ratio, stage_count, "expansion_ratio")
        activations = self._expand(activation, stage_count, "activation")
        dropouts = self._expand(dropout, stage_count, "dropout")
        layer_scales = self._expand(layer_scale_init, stage_count, "layer_scale_init")
        activation_parameters = self._expand_kwargs(
            activation_kwargs, stage_count, "activation_kwargs"
        )
        if any(int(value) < 1 or int(value) % 2 == 0 for value in kernels):
            raise ValueError("Every block kernel size must be a positive odd integer.")
        if any(float(value) <= 0 for value in expansions):
            raise ValueError("Every expansion ratio must be positive.")
        if any(not 0.0 <= float(value) < 1.0 for value in dropouts):
            raise ValueError("Every dropout value must be in [0, 1).")

        total_blocks = sum(depths)
        if drop_path_probabilities is None:
            block_drop_paths = torch.linspace(0.0, drop_path_rate, total_blocks).tolist()
        else:
            block_drop_paths = [float(value) for value in drop_path_probabilities]
            if len(block_drop_paths) != total_blocks:
                raise ValueError(
                    "drop_path_probabilities must contain one value per ConvNeXt block."
                )
            if any(not 0.0 <= value < 1.0 for value in block_drop_paths):
                raise ValueError("Every drop-path probability must be in [0, 1).")

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                channels[0],
                kernel_size=stem_kernel_size,
                stride=stem_stride,
                bias=bias,
            ),
            ChannelLayerNorm(
                channels[0],
                eps=layer_norm_eps,
                bias=bias,
                channel_dim=1,
            ),
        )
        self.downsampling_layers = nn.ModuleList(
            nn.Sequential(
                ChannelLayerNorm(
                    channels[index - 1],
                    eps=layer_norm_eps,
                    bias=bias,
                    channel_dim=1,
                ),
                nn.Conv2d(
                    channels[index - 1],
                    channels[index],
                    kernel_size=downsample_kernel_size,
                    stride=downsample_stride,
                    bias=bias,
                ),
            )
            for index in range(1, stage_count)
        )

        stages: list[nn.Module] = []
        drop_path_index = 0
        for stage_index, depth in enumerate(depths):
            blocks: list[nn.Module] = []
            for _ in range(depth):
                blocks.append(
                    ConvNeXtBlock2D(
                        channels[stage_index],
                        kernel_size=int(kernels[stage_index]),
                        expansion_ratio=float(expansions[stage_index]),
                        activation=activations[stage_index],
                        activation_kwargs=activation_parameters[stage_index],
                        dropout=float(dropouts[stage_index]),
                        drop_path_probability=block_drop_paths[drop_path_index],
                        layer_scale_init=layer_scales[stage_index],
                        layer_norm_eps=layer_norm_eps,
                        bias=bias,
                    )
                )
                drop_path_index += 1
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.feature_channels = tuple(channels)
        self.final_normalization = nn.LayerNorm(
            channels[-1],
            eps=layer_norm_eps,
            bias=bias,
        )
        self.head_dropout = nn.Dropout(head_dropout)
        self.head = (
            nn.Linear(channels[-1], out_features, bias=head_bias)
            if out_features is not None
            else nn.Identity()
        )
        self.initialization_std = float(initialization_std)
        self.apply(self._initialize)

    @staticmethod
    def _expand(value: Any, count: int, name: str) -> list[Any]:
        resolved = list(value) if isinstance(value, (list, tuple)) else [value] * count
        if len(resolved) != count:
            raise ValueError(f"{name} must contain exactly {count} values.")
        return resolved

    @staticmethod
    def _expand_kwargs(
        value: dict[str, Any] | Sequence[dict[str, Any]] | None,
        count: int,
        name: str,
    ) -> list[dict[str, Any]]:
        if value is None:
            return [{} for _ in range(count)]
        if isinstance(value, dict):
            return [dict(value) for _ in range(count)]
        resolved = [dict(item) for item in value]
        if len(resolved) != count:
            raise ValueError(f"{name} must contain exactly {count} values.")
        return resolved

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=self.initialization_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_feature_maps(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return one NCHW map per ConvNeXt stage, from fine to coarse."""
        if x.ndim != 4:
            raise ValueError("x must have shape (batch, channels, height, width).")
        output = self.stem(x)
        feature_maps: list[torch.Tensor] = []
        for stage_index, stage in enumerate(self.stages):
            if stage_index > 0:
                output = self.downsampling_layers[stage_index - 1](output)
            output = stage(output)
            feature_maps.append(output)
        return tuple(feature_maps)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final globally pooled ConvNeXt representation."""
        output = self.forward_feature_maps(x)[-1]
        return self.final_normalization(output.mean(dim=(-2, -1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an NCHW image and return logits or pooled embeddings."""
        return self.head(self.head_dropout(self.forward_features(x)))
