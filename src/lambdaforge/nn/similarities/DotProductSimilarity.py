"""Scaled dot-product similarity."""

from __future__ import annotations

import math

import torch

from lambdaforge.nn.similarities.Similarity import Similarity


class DotProductSimilarity(Similarity):
    """Compute ``scale * x @ y.T`` for two batched vector sets.

    Parameters
    ----------
    scale:
        Multiplicative scale. ``None`` applies the common ``1 / sqrt(F)``
        scaled-dot-product convention using the runtime feature dimension.
    name:
        Optional diagnostic name.
    """

    def __init__(self, scale: float | None = 1.0, name: str | None = None) -> None:
        super().__init__(name=name)
        if scale is not None:
            if isinstance(scale, bool) or not isinstance(scale, (int, float)):
                raise TypeError("scale must be a real number or None.")
            if not math.isfinite(float(scale)):
                raise ValueError("scale must be finite or None.")
        self.scale = None if scale is None else float(scale)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.validate_inputs(x, y)
        scale = x.shape[-1] ** -0.5 if self.scale is None else self.scale
        return torch.matmul(x, y.transpose(-1, -2)) * scale
