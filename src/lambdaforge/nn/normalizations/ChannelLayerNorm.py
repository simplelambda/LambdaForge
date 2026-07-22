"""Layer normalization over a configurable channel dimension."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from lambdaforge.nn.normalizations.Normalization import Normalization


class ChannelLayerNorm(Normalization):
    """Apply LayerNorm independently at every non-channel position.

    Unlike standard ``LayerNorm`` on NCHW tensors, this object normalizes only
    the channel axis. ``channel_dim`` supports both channel-first and
    channel-last data without hard-coded permutations.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        channel_dim: int = 1,
        name: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(name=name)
        if num_features < 1:
            raise ValueError("num_features must be positive.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.weight = (
            nn.Parameter(torch.ones(num_features, device=device, dtype=dtype))
            if elementwise_affine
            else None
        )
        self.bias = (
            nn.Parameter(torch.zeros(num_features, device=device, dtype=dtype))
            if elementwise_affine and bias
            else None
        )
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.channel_dim = int(channel_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resolved_dim = self.channel_dim if self.channel_dim >= 0 else x.ndim + self.channel_dim
        if resolved_dim < 0 or resolved_dim >= x.ndim:
            raise ValueError(f"channel_dim={self.channel_dim} is invalid for {x.ndim} dimensions.")
        if x.shape[resolved_dim] != self.num_features:
            raise ValueError(f"Expected {self.num_features} channels, got {x.shape[resolved_dim]}.")
        moved = x.movedim(resolved_dim, -1)
        normalized = F.layer_norm(
            moved,
            (self.num_features,),
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        )
        return normalized.movedim(-1, resolved_dim)
