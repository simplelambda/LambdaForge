"""Feature-wise structured dropout regularization."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from lambdaforge.nn.regularization.Regularization import Regularization


class FeatureDropout(Regularization):
    """Drop complete features while sharing masks across selected axes.

    By default one independent feature mask is sampled per batch element and
    shared over every axis except the batch axis and ``feature_dim``. For
    example, input ``(B, T, D)`` with ``feature_dim=-1`` receives a mask of
    shape ``(B, 1, D)``.
    """

    def __init__(
        self,
        probability: float = 0.5,
        feature_dim: int = -1,
        shared_dims: Sequence[int] | None = None,
        scale_by_keep: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("probability must be in [0, 1).")
        self.probability = float(probability)
        self.feature_dim = int(feature_dim)
        self.shared_dims = None if shared_dims is None else tuple(int(dim) for dim in shared_dims)
        self.scale_by_keep = bool(scale_by_keep)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a structured Bernoulli mask without changing tensor shape."""
        if not self.training or self.probability == 0.0:
            return x
        if x.ndim < 2:
            raise ValueError("FeatureDropout requires batch and feature dimensions.")
        feature_dim = self.feature_dim % x.ndim
        if feature_dim == 0:
            raise ValueError("feature_dim cannot be the batch dimension.")
        if self.shared_dims is None:
            shared_dims = tuple(dim for dim in range(1, x.ndim) if dim != feature_dim)
        else:
            shared_dims = tuple(dim % x.ndim for dim in self.shared_dims)
            if 0 in shared_dims or feature_dim in shared_dims:
                raise ValueError("shared_dims cannot contain batch or feature_dim.")
        shape = list(x.shape)
        for dim in shared_dims:
            shape[dim] = 1
        keep = 1.0 - self.probability
        mask = x.new_empty(shape).bernoulli_(keep)
        if self.scale_by_keep:
            mask = mask / keep
        return x * mask
