"""Abstract pooling contract for sparse grouped rows."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class SparsePooling(nn.Module, ABC):
    """Reduce ``x[N, D]`` according to ``group_index[N]``."""

    @staticmethod
    def validate(
        x: torch.Tensor,
        group_index: torch.Tensor,
        num_groups: int | None,
    ) -> tuple[torch.Tensor, int]:
        """Validate sparse grouping inputs and infer the number of groups."""
        if x.ndim != 2:
            raise ValueError("x must have shape (N, D).")
        if group_index.ndim != 1 or group_index.shape[0] != x.shape[0]:
            raise ValueError("group_index must have shape (N,).")
        group_index = group_index.to(device=x.device, dtype=torch.long)
        if group_index.numel() and int(group_index.min()) < 0:
            raise IndexError("group_index cannot contain negative values.")
        inferred = int(group_index.max()) + 1 if group_index.numel() else 0
        if num_groups is None:
            num_groups = inferred
        if num_groups < inferred:
            raise ValueError("num_groups is smaller than the largest group index.")
        return group_index, int(num_groups)

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        group_index: torch.Tensor,
        num_groups: int | None = None,
    ) -> torch.Tensor:
        """Return one feature vector per sparse group."""
        raise NotImplementedError
