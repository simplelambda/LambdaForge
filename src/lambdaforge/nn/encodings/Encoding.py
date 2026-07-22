"""Abstract interface for neural encoding objects."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class Encoding(nn.Module, ABC):
    """Convert or enrich tensor representations without task assumptions."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``x`` according to the concrete object's shape contract."""
        raise NotImplementedError
