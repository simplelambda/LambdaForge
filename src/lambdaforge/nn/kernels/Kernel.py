"""Abstract contract for pairwise kernel modules."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class Kernel(nn.Module, ABC):
    """Base class for kernels over batched sets of feature vectors.

    Inputs have shapes ``(B, N, F)`` and ``(B, M, F)``. The returned Gram
    tensor has shape ``(B, N, M)``.
    """

    def __init__(self, name: str | None = None) -> None:
        super().__init__()
        self.name = name or self.__class__.__name__

    @staticmethod
    def validate_inputs(x: torch.Tensor, y: torch.Tensor) -> None:
        """Validate the shared dense pairwise shape contract."""
        if x.ndim != 3 or y.ndim != 3:
            raise ValueError("Kernel inputs must have shapes (B, N, F) and (B, M, F).")
        if x.shape[0] != y.shape[0]:
            raise ValueError("Kernel inputs must have the same batch size.")
        if x.shape[-1] != y.shape[-1]:
            raise ValueError("Kernel inputs must have the same feature dimension.")
        if x.device != y.device:
            raise ValueError("Kernel inputs must be on the same device.")
        if x.dtype != y.dtype:
            raise ValueError("Kernel inputs must have the same dtype.")
        if not x.is_floating_point() or not y.is_floating_point():
            raise TypeError("Kernel inputs must be floating-point tensors.")

    @abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return a batched pairwise kernel tensor."""
        raise NotImplementedError
