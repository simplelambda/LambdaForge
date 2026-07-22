"""Abstract interface for regularization objects."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class Regularization(nn.Module, ABC):
    """Transform tensors stochastically during training and safely during evaluation."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a regularized tensor with the same shape as ``x``."""
        raise NotImplementedError
