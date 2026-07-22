"""Top-down feature pyramid for hierarchical image backbones."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lambdaforge.nn.models.vision.HierarchicalBackbone2D import HierarchicalBackbone2D


class FeaturePyramidNetwork2D(HierarchicalBackbone2D):
    """Fuse backbone stages into equal-width fine-to-coarse feature maps."""

    def __init__(
        self,
        backbone: HierarchicalBackbone2D,
        out_channels: int = 256,
        extra_levels: int = 0,
        interpolation_mode: str = "nearest",
        bias: bool = True,
    ) -> None:
        super().__init__()
        channels = tuple(int(value) for value in backbone.feature_channels)
        if not channels or any(value < 1 for value in channels):
            raise ValueError("backbone.feature_channels must contain positive channel counts.")
        if out_channels < 1 or extra_levels < 0:
            raise ValueError("out_channels must be positive and extra_levels cannot be negative.")
        if interpolation_mode not in {"nearest", "bilinear"}:
            raise ValueError("interpolation_mode must be 'nearest' or 'bilinear'.")
        self.backbone = backbone
        self.lateral_convolutions = nn.ModuleList(
            nn.Conv2d(channel_count, out_channels, 1, bias=bias) for channel_count in channels
        )
        self.output_convolutions = nn.ModuleList(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=bias) for _ in channels
        )
        self.feature_channels = (out_channels,) * (len(channels) + extra_levels)
        self.extra_levels = int(extra_levels)
        self.interpolation_mode = interpolation_mode

    def forward_feature_maps(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return fused pyramid maps ordered from finest to coarsest."""
        backbone_maps = self.backbone.forward_feature_maps(x)
        if len(backbone_maps) != len(self.lateral_convolutions):
            raise RuntimeError("Backbone feature-map count changed after FPN construction.")
        lateral = [
            convolution(feature)
            for convolution, feature in zip(self.lateral_convolutions, backbone_maps, strict=True)
        ]
        fused: list[torch.Tensor] = [torch.empty(0)] * len(lateral)
        top_down = lateral[-1]
        fused[-1] = self.output_convolutions[-1](top_down)
        for index in range(len(lateral) - 2, -1, -1):
            interpolation_options = {
                "size": lateral[index].shape[-2:],
                "mode": self.interpolation_mode,
            }
            if self.interpolation_mode == "bilinear":
                interpolation_options["align_corners"] = False
            top_down = lateral[index] + F.interpolate(top_down, **interpolation_options)
            fused[index] = self.output_convolutions[index](top_down)
        for _ in range(self.extra_levels):
            fused.append(F.max_pool2d(fused[-1], kernel_size=1, stride=2))
        return tuple(fused)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return a pooled representation of the coarsest pyramid level."""
        return self.forward_feature_maps(x)[-1].mean(dim=(-2, -1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Build the full fine-to-coarse feature pyramid."""
        return self.forward_feature_maps(x)
