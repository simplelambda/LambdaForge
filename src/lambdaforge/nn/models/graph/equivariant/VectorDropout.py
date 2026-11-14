"""Drop whole polar-vector channels without selecting Cartesian axes."""

import math

import torch
from torch import nn


class VectorDropout(nn.Module):
    """Use one Bernoulli mask for all three coordinates of a vector channel."""

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        if (
            isinstance(probability, bool)
            or not math.isfinite(probability)
            or not 0 <= probability < 1
        ):
            raise ValueError("probability must be finite in [0,1).")
        self.probability = probability

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        """Return shape-preserving channel dropout (identity in eval mode)."""
        if vectors.ndim < 2 or vectors.shape[-1] != 3 or not vectors.is_floating_point():
            raise ValueError("vectors must end in (channels, 3).")
        if not self.training or not self.probability:
            return vectors
        mask = vectors.new_empty((*vectors.shape[:-1], 1)).bernoulli_(1 - self.probability)
        return vectors * mask / (1 - self.probability)
