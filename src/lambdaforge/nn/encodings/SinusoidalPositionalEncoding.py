"""Deterministic sinusoidal positional encoding."""

from __future__ import annotations

import math

import torch
from torch import nn

from lambdaforge.nn.encodings.Encoding import Encoding


class SinusoidalPositionalEncoding(Encoding):
    """Add deterministic sine/cosine positions to a sequence.

    Inputs use ``(B, T, D)`` when ``batch_first=True`` and ``(T, B, D)``
    otherwise. The feature dimension must equal ``features``.
    """

    def __init__(
        self,
        features: int,
        max_length: int = 4096,
        base: float = 10_000.0,
        dropout: float = 0.0,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        if features < 1 or max_length < 1:
            raise ValueError("features and max_length must be positive.")
        if base <= 1.0:
            raise ValueError("base must be greater than one.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, features, 2, dtype=torch.float32) * (-math.log(base) / features)
        )
        values = torch.zeros(max_length, features, dtype=torch.float32)
        values[:, 0::2] = torch.sin(positions * frequencies)
        if features > 1:
            values[:, 1::2] = torch.cos(positions * frequencies[: values[:, 1::2].shape[1]])
        self.encoding: torch.Tensor
        self.register_buffer("encoding", values, persistent=True)
        self.features = features
        self.max_length = max_length
        self.batch_first = bool(batch_first)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Add positions beginning at ``offset`` and apply optional dropout."""
        if x.ndim != 3 or x.shape[-1] != self.features:
            raise ValueError("x must have shape (B, T, features) or (T, B, features).")
        if offset < 0:
            raise ValueError("offset must be non-negative.")
        length = x.shape[1 if self.batch_first else 0]
        if offset + length > self.max_length:
            raise ValueError("Requested positions exceed max_length.")
        values = self.encoding[offset : offset + length].to(dtype=x.dtype)
        values = values.unsqueeze(0 if self.batch_first else 1)
        return self.dropout(x + values)
