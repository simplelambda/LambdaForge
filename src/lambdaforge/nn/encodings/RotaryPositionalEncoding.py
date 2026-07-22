"""Rotary positional encoding for sequence representations."""

from __future__ import annotations

import torch

from lambdaforge.nn.encodings.Encoding import Encoding


class RotaryPositionalEncoding(Encoding):
    """Rotate adjacent feature pairs according to their sequence position.

    The sequence axis is configurable and the final feature dimension must be
    even. Applying the same object to query and key tensors preserves their
    relative-position dot-product structure.
    """

    def __init__(
        self,
        features: int,
        base: float = 10_000.0,
        sequence_dim: int = -2,
    ) -> None:
        super().__init__()
        if features < 2 or features % 2:
            raise ValueError("features must be a positive even integer.")
        if base <= 1.0:
            raise ValueError("base must be greater than one.")
        inverse_frequencies = base ** (
            -torch.arange(0, features, 2, dtype=torch.float32) / features
        )
        self.inverse_frequencies: torch.Tensor
        self.register_buffer("inverse_frequencies", inverse_frequencies, persistent=True)
        self.features = features
        self.sequence_dim = int(sequence_dim)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Apply rotary positions beginning at ``offset``."""
        if x.ndim < 2 or x.shape[-1] != self.features:
            raise ValueError(f"x must have at least two dimensions and end in {self.features}.")
        if offset < 0:
            raise ValueError("offset must be non-negative.")
        sequence_dim = self.sequence_dim % x.ndim
        if sequence_dim == x.ndim - 1:
            raise ValueError("sequence_dim cannot be the feature dimension.")
        moved = x.movedim(sequence_dim, -2)
        positions = torch.arange(
            offset,
            offset + moved.shape[-2],
            device=x.device,
            dtype=self.inverse_frequencies.dtype,
        )
        angles = torch.outer(positions, self.inverse_frequencies).to(dtype=x.dtype)
        shape = (1,) * (moved.ndim - 2) + angles.shape
        cosine = angles.cos().reshape(shape)
        sine = angles.sin().reshape(shape)
        even = moved[..., 0::2]
        odd = moved[..., 1::2]
        rotated = torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine),
            dim=-1,
        ).flatten(-2)
        return rotated.movedim(-2, sequence_dim)
