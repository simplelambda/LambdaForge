"""Group normalization wrapper."""

from __future__ import annotations

import torch
from torch import nn

from lambdaforge.nn.normalizations.Normalization import Normalization


class GroupNorm(Normalization):
    """Normalize channel groups independently of batch size."""

    def __init__(
        self,
        num_features: int,
        num_groups: int = 32,
        eps: float = 1e-5,
        affine: bool = True,
        name: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(name=name)
        if num_features < 1 or num_groups < 1:
            raise ValueError("num_features and num_groups must be positive.")
        if num_features % num_groups != 0:
            raise ValueError("num_features must be divisible by num_groups.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.norm = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=num_features,
            eps=eps,
            affine=affine,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)
