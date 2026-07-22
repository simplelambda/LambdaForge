"""Cosine similarity for batched vector sets."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from lambdaforge.nn.similarities.Similarity import Similarity


class CosineSimilarity(Similarity):
    """Return pairwise cosine similarities in ``[-1, 1]``.

    ``eps`` bounds the norm denominator and is fully configurable for unusual
    numeric precision regimes.
    """

    def __init__(self, eps: float = 1e-8, name: str | None = None) -> None:
        super().__init__(name=name)
        if isinstance(eps, bool) or not isinstance(eps, (int, float)):
            raise TypeError("eps must be a real number.")
        if not math.isfinite(float(eps)) or eps <= 0:
            raise ValueError("eps must be finite and positive.")
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.validate_inputs(x, y)
        x_normalized = F.normalize(x, p=2.0, dim=-1, eps=self.eps)
        y_normalized = F.normalize(y, p=2.0, dim=-1, eps=self.eps)
        return torch.matmul(x_normalized, y_normalized.transpose(-1, -2))
