"""L2 vector normalization."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lambdaforge.nn.normalizations.Normalization import Normalization


class L2Norm(Normalization):
    """Normalize vectors to unit L2 norm along a configurable dimension.

    ``num_features`` is optional but makes the object compatible with generic
    model factories and enables a runtime feature-size check.
    """

    def __init__(
        self,
        num_features: int | None = None,
        dim: int = -1,
        eps: float = 1e-12,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if num_features is not None and num_features < 1:
            raise ValueError("num_features must be positive or None.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.num_features = num_features
        self.dim = int(dim)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resolved_dim = self.dim if self.dim >= 0 else x.ndim + self.dim
        if resolved_dim < 0 or resolved_dim >= x.ndim:
            raise ValueError(f"dim={self.dim} is invalid for an input with {x.ndim} dimensions.")
        if self.num_features is not None and x.shape[resolved_dim] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features on dimension {self.dim}, "
                f"got {x.shape[resolved_dim]}."
            )
        return F.normalize(x, p=2.0, dim=self.dim, eps=self.eps)
