"""Learnable ScaleNorm layer."""

from __future__ import annotations

import math

import torch
from torch import nn

from lambdaforge.nn.normalizations.Normalization import Normalization


class ScaleNorm(Normalization):
    r"""Scale vectors by ``scale / max(||x||_2, eps)``.

    The default scale is ``sqrt(num_features)``. It can be fixed or learned as
    a scalar, retaining ScaleNorm's low parameter count.
    """

    def __init__(
        self,
        num_features: int,
        scale: float | None = None,
        learnable: bool = True,
        dim: int = -1,
        eps: float = 1e-5,
        name: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(name=name)
        if num_features < 1:
            raise ValueError("num_features must be positive.")
        resolved_scale = math.sqrt(num_features) if scale is None else float(scale)
        if resolved_scale <= 0 or eps <= 0:
            raise ValueError("scale and eps must be positive.")
        value = torch.tensor(resolved_scale, device=device, dtype=dtype)
        if learnable:
            self.scale = nn.Parameter(value)
        else:
            self.register_buffer("scale", value)
        self.num_features = int(num_features)
        self.dim = int(dim)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resolved_dim = self.dim if self.dim >= 0 else x.ndim + self.dim
        if resolved_dim < 0 or resolved_dim >= x.ndim:
            raise ValueError(f"dim={self.dim} is invalid for an input with {x.ndim} dimensions.")
        if x.shape[resolved_dim] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features on dimension {self.dim}, "
                f"got {x.shape[resolved_dim]}."
            )
        denominator = torch.linalg.vector_norm(x, ord=2, dim=self.dim, keepdim=True)
        return x * self.scale / denominator.clamp_min(self.eps)
