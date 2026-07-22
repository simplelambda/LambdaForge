"""Learned absolute positional encoding."""

from __future__ import annotations

import torch
from torch import nn

from lambdaforge.nn.encodings.Encoding import Encoding


class LearnedPositionalEncoding(Encoding):
    """Add trainable absolute positions to a batch-first or sequence-first tensor."""

    def __init__(
        self,
        features: int,
        max_length: int = 4096,
        dropout: float = 0.0,
        batch_first: bool = True,
        initialization_standard_deviation: float = 0.02,
    ) -> None:
        super().__init__()
        if features < 1 or max_length < 1:
            raise ValueError("features and max_length must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if initialization_standard_deviation < 0.0:
            raise ValueError("initialization_standard_deviation must be non-negative.")
        self.positions = nn.Parameter(torch.empty(max_length, features))
        nn.init.normal_(self.positions, std=initialization_standard_deviation)
        self.features = features
        self.max_length = max_length
        self.batch_first = bool(batch_first)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Add the requested trainable slice and apply optional dropout."""
        if x.ndim != 3 or x.shape[-1] != self.features:
            raise ValueError("x must have shape (B, T, features) or (T, B, features).")
        if offset < 0:
            raise ValueError("offset must be non-negative.")
        length = x.shape[1 if self.batch_first else 0]
        if offset + length > self.max_length:
            raise ValueError("Requested positions exceed max_length.")
        values = self.positions[offset : offset + length]
        values = values.unsqueeze(0 if self.batch_first else 1)
        return self.dropout(x + values)
