"""Smooth compact radial support without constructing or removing graph edges."""

import math

import torch
from torch import nn

from lambdaforge.nn.models.graph.validation import positive


class CosineCutoff(nn.Module):
    """Return (cos(pi*d/cutoff)+1)/2 inside the radius, zero at/outside it."""

    def __init__(self, cutoff: float) -> None:
        super().__init__()
        self.cutoff = positive(cutoff, "cutoff")

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Preserve shape; reject negative/nonfinite distances."""
        if (
            not distances.is_floating_point()
            or not torch.isfinite(distances).all()
            or (distances < 0).any()
        ):
            raise ValueError("distances must be finite non-negative floating values.")
        return torch.where(
            distances < self.cutoff,
            0.5 * (torch.cos(math.pi * distances / self.cutoff) + 1),
            torch.zeros_like(distances),
        )
