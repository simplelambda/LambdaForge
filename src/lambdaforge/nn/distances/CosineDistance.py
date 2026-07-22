"""Implementation of the CosineDistance object."""

from __future__ import annotations

import math

import torch

from lambdaforge.nn.distances.Distance import Distance


class CosineDistance(Distance):
    r"""Pairwise cosine distance ``1 - cosine_similarity``.

    Parameters
    ----------
    eps : float
        Positive lower bound used when normalizing vector norms. Zero vectors
        are therefore represented by zero normalized vectors. Default:
        ``1e-12``.
    """

    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        if isinstance(eps, bool) or not isinstance(eps, (int, float)):
            raise TypeError("eps must be a real number")
        if not math.isfinite(float(eps)) or float(eps) <= 0.0:
            raise ValueError("eps must be finite and greater than zero")
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or y.ndim != 3:
            raise ValueError("x and y must both have shape (B, T, F)")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same batch size")
        if x.shape[-1] != y.shape[-1]:
            raise ValueError("x and y must have the same feature dimension")
        if x.device != y.device or x.dtype != y.dtype:
            raise ValueError("x and y must have the same device and dtype")
        if not x.is_floating_point() or not y.is_floating_point():
            raise TypeError("x and y must be floating-point tensors")
        normalized_x = torch.nn.functional.normalize(x, p=2.0, dim=-1, eps=self.eps)
        normalized_y = torch.nn.functional.normalize(y, p=2.0, dim=-1, eps=self.eps)
        similarity = torch.bmm(normalized_x, normalized_y.transpose(1, 2)).clamp(-1.0, 1.0)
        return 1.0 - similarity

    def extra_repr(self) -> str:
        return f"eps={self.eps}"
