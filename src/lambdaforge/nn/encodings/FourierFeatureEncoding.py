"""Random or trainable Fourier feature encoding."""

from __future__ import annotations

import math

import torch
from torch import nn

from lambdaforge.nn.encodings.Encoding import Encoding


class FourierFeatureEncoding(Encoding):
    """Map continuous inputs to periodic Fourier features.

    A private CPU generator initializes the projection, so constructing this
    object never advances PyTorch's global random state.
    """

    def __init__(
        self,
        in_features: int,
        num_frequencies: int = 64,
        scale: float = 1.0,
        learnable: bool = False,
        include_input: bool = False,
        phase: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if in_features < 1 or num_frequencies < 1:
            raise ValueError("in_features and num_frequencies must be positive.")
        if scale <= 0.0:
            raise ValueError("scale must be positive.")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        projection = (
            torch.randn(
                in_features,
                num_frequencies,
                generator=generator,
                dtype=torch.float32,
            )
            * scale
        )
        phase_values = (
            torch.rand(num_frequencies, generator=generator, dtype=torch.float32) * (2.0 * math.pi)
            if phase
            else torch.zeros(num_frequencies, dtype=torch.float32)
        )
        if learnable:
            self.projection = nn.Parameter(projection)
            self.phase = nn.Parameter(phase_values)
        else:
            self.register_buffer("projection", projection, persistent=True)
            self.register_buffer("phase", phase_values, persistent=True)
        self.in_features = in_features
        self.num_frequencies = num_frequencies
        self.include_input = bool(include_input)

    @property
    def out_features(self) -> int:
        """Return the encoded feature dimension."""
        return 2 * self.num_frequencies + (self.in_features if self.include_input else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the final dimension of ``x`` with sine and cosine features."""
        if x.shape[-1] != self.in_features:
            raise ValueError(f"x must end in {self.in_features} features.")
        angles = x @ self.projection.to(dtype=x.dtype) + self.phase.to(dtype=x.dtype)
        encoded = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return torch.cat((x, encoded), dim=-1) if self.include_input else encoded
