"""Patch-based Transformer encoder for two-dimensional images."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from lambdaforge.nn.activations.Activation import Activation
from lambdaforge.nn.ComponentRegistry import ComponentRegistry
from lambdaforge.nn.models.Model import Model
from lambdaforge.nn.models.vision.PatchRemainderPolicy import PatchRemainderPolicy
from lambdaforge.nn.models.vision.VisionTransformerOutputMode import (
    VisionTransformerOutputMode,
)


class VisionTransformer2D(Model):
    """Encode NCHW images as patch tokens with interpolated learned positions.

    ``image_size`` defines the learned reference grid, not a fixed input size.
    Positional embeddings are interpolated for other resolutions. Images whose
    dimensions are not divisible by ``patch_size`` are either rejected or
    padded on the bottom and right according to ``remainder_policy``.
    """

    def __init__(
        self,
        in_channels: int,
        patch_size: int | Sequence[int] = 16,
        image_size: int | Sequence[int] = 224,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        out_features: int | None = None,
        feedforward_dim: int | None = None,
        dropout: float = 0.0,
        activation: type[Activation] | str = "gelu",
        activation_kwargs: dict[str, Any] | None = None,
        output_mode: VisionTransformerOutputMode | str = VisionTransformerOutputMode.CLASS_TOKEN,
        remainder_policy: PatchRemainderPolicy | str = PatchRemainderPolicy.ERROR,
        use_class_token: bool = True,
        norm_first: bool = True,
        final_normalization: bool = True,
        layer_norm_eps: float = 1e-6,
        bias: bool = True,
        initialization_std: float = 0.02,
        enable_nested_tensor: bool = False,
    ) -> None:
        super().__init__()
        patch_height, patch_width = self._pair(patch_size, "patch_size")
        image_height, image_width = self._pair(image_size, "image_size")
        if min(in_channels, d_model, num_heads, num_layers) < 1:
            raise ValueError("Channels, d_model, num_heads and num_layers must be positive.")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads.")
        if out_features is not None and out_features < 1:
            raise ValueError("out_features must be positive when provided.")
        if feedforward_dim is not None and feedforward_dim < 1:
            raise ValueError("feedforward_dim must be positive when provided.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if layer_norm_eps <= 0 or initialization_std <= 0:
            raise ValueError("layer_norm_eps and initialization_std must be positive.")

        self.in_channels = int(in_channels)
        self.patch_size = (patch_height, patch_width)
        self.output_mode = VisionTransformerOutputMode(output_mode)
        self.remainder_policy = PatchRemainderPolicy(remainder_policy)
        self.use_class_token = bool(use_class_token)
        if self.output_mode is VisionTransformerOutputMode.CLASS_TOKEN and not self.use_class_token:
            raise ValueError("class_token output requires use_class_token=True.")

        self.patch_embedding = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
        )
        reference_height = math.ceil(image_height / patch_height)
        reference_width = math.ceil(image_width / patch_width)
        self.position_grid = nn.Parameter(
            torch.empty(1, d_model, reference_height, reference_width)
        )
        self.class_token = (
            nn.Parameter(torch.empty(1, 1, d_model)) if self.use_class_token else None
        )
        self.class_position = (
            nn.Parameter(torch.empty(1, 1, d_model)) if self.use_class_token else None
        )
        activation_cls = ComponentRegistry.resolve_activation(activation)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=feedforward_dim or 4 * d_model,
            dropout=dropout,
            activation=activation_cls(**(activation_kwargs or {})),
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=norm_first,
            bias=bias,
        )
        final_norm = (
            nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias) if final_normalization else None
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers,
            norm=final_norm,
            enable_nested_tensor=enable_nested_tensor,
        )
        self.embedding_dropout = nn.Dropout(dropout)
        self.head = (
            nn.Linear(d_model, out_features, bias=bias)
            if out_features is not None
            else nn.Identity()
        )
        self.d_model = int(d_model)
        self.output_features = out_features or d_model
        self.initialization_std = float(initialization_std)
        self.apply(self._initialize)
        nn.init.trunc_normal_(self.position_grid, std=self.initialization_std)
        if self.class_token is not None:
            nn.init.trunc_normal_(self.class_token, std=self.initialization_std)
        if self.class_position is not None:
            nn.init.trunc_normal_(self.class_position, std=self.initialization_std)

    @staticmethod
    def _pair(value: int | Sequence[int], name: str) -> tuple[int, int]:
        if isinstance(value, int):
            resolved = (value, value)
        else:
            resolved_values = tuple(int(item) for item in value)
            if len(resolved_values) != 2:
                raise ValueError(f"{name} must be an integer or contain exactly two values.")
            resolved = resolved_values
        if min(resolved) < 1:
            raise ValueError(f"{name} values must be positive.")
        return resolved

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=self.initialization_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _patches(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if x.ndim != 4:
            raise ValueError("x must have shape (batch, channels, height, width).")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}.")
        patch_height, patch_width = self.patch_size
        remainder_height = x.shape[-2] % patch_height
        remainder_width = x.shape[-1] % patch_width
        if remainder_height or remainder_width:
            if self.remainder_policy is PatchRemainderPolicy.ERROR:
                raise ValueError("Image dimensions must be divisible by patch_size.")
            x = F.pad(
                x,
                (0, (-x.shape[-1]) % patch_width, 0, (-x.shape[-2]) % patch_height),
            )
        embedded = self.patch_embedding(x)
        grid_height, grid_width = embedded.shape[-2:]
        tokens = embedded.flatten(2).transpose(1, 2)
        return tokens, grid_height, grid_width

    def forward_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Return encoded patch tokens plus their two-dimensional grid size."""
        tokens, grid_height, grid_width = self._patches(x)
        position = F.interpolate(
            self.position_grid,
            size=(grid_height, grid_width),
            mode="bicubic",
            align_corners=False,
        )
        tokens = tokens + position.flatten(2).transpose(1, 2).to(dtype=tokens.dtype)
        if self.class_token is not None and self.class_position is not None:
            class_token = self.class_token.expand(tokens.shape[0], -1, -1)
            class_token = class_token + self.class_position.to(dtype=tokens.dtype)
            tokens = torch.cat((class_token, tokens), dim=1)
        tokens = self.encoder(self.embedding_dropout(tokens))
        return tokens, grid_height, grid_width

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the selected unprojected token representation."""
        encoded, grid_height, grid_width = self.forward_tokens(x)
        patch_tokens = encoded[:, 1:] if self.class_token is not None else encoded
        if self.output_mode is VisionTransformerOutputMode.CLASS_TOKEN:
            return encoded[:, 0]
        if self.output_mode is VisionTransformerOutputMode.MEAN:
            return patch_tokens.mean(dim=1)
        if self.output_mode is VisionTransformerOutputMode.TOKENS:
            return patch_tokens
        return patch_tokens.transpose(1, 2).reshape(
            encoded.shape[0], self.d_model, grid_height, grid_width
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an image and apply the optional feature projection."""
        features = self.forward_features(x)
        if self.output_mode is VisionTransformerOutputMode.FEATURE_MAP:
            projected = self.head(features.movedim(1, -1))
            return projected.movedim(-1, 1)
        return self.head(features)
