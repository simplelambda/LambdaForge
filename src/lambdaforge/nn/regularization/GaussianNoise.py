"""Configurable additive Gaussian-noise regularization."""

from __future__ import annotations

import torch

from lambdaforge.nn.regularization.Regularization import Regularization


class GaussianNoise(Regularization):
    """Add absolute or input-relative Gaussian noise.

    Parameters
    ----------
    standard_deviation:
        Standard deviation of the sampled Gaussian.
    mean:
        Mean of the sampled Gaussian.
    relative:
        Scale noise elementwise by the detached absolute input.
    only_training:
        Disable noise automatically while the object is in evaluation mode.
    minimum_scale:
        Lower bound used by relative noise, useful for zero-valued inputs.
    """

    def __init__(
        self,
        standard_deviation: float = 0.1,
        mean: float = 0.0,
        relative: bool = False,
        only_training: bool = True,
        minimum_scale: float = 0.0,
    ) -> None:
        super().__init__()
        if standard_deviation < 0.0:
            raise ValueError("standard_deviation must be non-negative.")
        if minimum_scale < 0.0:
            raise ValueError("minimum_scale must be non-negative.")
        self.standard_deviation = float(standard_deviation)
        self.mean = float(mean)
        self.relative = bool(relative)
        self.only_training = bool(only_training)
        self.minimum_scale = float(minimum_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` plus newly sampled noise."""
        if self.standard_deviation == 0.0 or (self.only_training and not self.training):
            return x
        noise = torch.randn_like(x) * self.standard_deviation + self.mean
        if self.relative:
            noise = noise * x.detach().abs().clamp_min(self.minimum_scale)
        return x + noise
