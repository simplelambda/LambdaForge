"""Dimension-selecting instance normalization."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from lambdaforge.nn.normalizations.Normalization import Normalization


class InstanceNorm(Normalization):
    """Instance normalization for 1D, 2D or 3D channel-first inputs.

    ``dim`` selects :class:`torch.nn.InstanceNorm1d`, ``2d`` or ``3d``. Every
    PyTorch constructor option remains explicit and YAML-configurable.
    """

    def __init__(
        self,
        num_features: int,
        dim: Literal[1, 2, 3] = 1,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        name: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(name=name)
        if num_features < 1:
            raise ValueError("num_features must be positive.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must be in [0, 1].")
        norm_classes: dict[int, type[nn.Module]] = {
            1: nn.InstanceNorm1d,
            2: nn.InstanceNorm2d,
            3: nn.InstanceNorm3d,
        }
        if dim not in norm_classes:
            raise ValueError("dim must be 1, 2 or 3.")
        self.norm = norm_classes[dim](
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)
