"""Gaussian expansion of explicit distances, with no implicit cutoff mask."""

import torch
from torch import nn

from lambdaforge.nn.models.graph.validation import channels, positive


class GaussianRadialBasis(nn.Module):
    """Expand (E,) lengths to (E, num_radial) Gaussians on [0, radius]."""

    def __init__(self, num_radial: int = 32, radius: float = 5.0) -> None:
        super().__init__()
        channels(num_radial=num_radial)
        self.radius = positive(radius, "radius")
        self.num_radial = num_radial
        self.width = radius / max(num_radial - 1, 1)
        self.register_buffer("centers", torch.linspace(0, radius, num_radial))

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Return Gaussian features; topology/support remain the caller's choice."""
        if (
            distances.ndim != 1
            or not distances.is_floating_point()
            or not torch.isfinite(distances).all()
            or (distances < 0).any()
        ):
            raise ValueError("distances must be finite non-negative floating (E,).")
        return torch.exp(-0.5 * ((distances[:, None] - self.centers) / self.width).square())
