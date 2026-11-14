"""Sinusoidal radial basis used by PaiNN, including the analytic zero limit."""

import math

import torch
from torch import nn

from lambdaforge.nn.models.graph.validation import channels, positive


class SineRadialBasis(nn.Module):
    """Return sin(n*pi*d/radius)/d; no Gaussian substitution or implicit support mask."""

    def __init__(self, num_radial: int = 20, radius: float = 5.0) -> None:
        super().__init__()
        channels(num_radial=num_radial)
        self.radius = positive(radius, "radius")
        self.num_radial = num_radial
        self.register_buffer("frequencies", torch.arange(1, num_radial + 1) * math.pi / radius)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Evaluate stably via sinc, including zero-length scalar distances."""
        if (
            distances.ndim != 1
            or not distances.is_floating_point()
            or not torch.isfinite(distances).all()
            or (distances < 0).any()
        ):
            raise ValueError("distances must be finite non-negative floating (E,).")
        return self.frequencies * torch.sinc(distances[:, None] * self.frequencies / math.pi)
