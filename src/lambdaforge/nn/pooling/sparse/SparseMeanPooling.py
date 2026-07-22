"""Sparse mean pooling."""

from __future__ import annotations

import torch

from lambdaforge.nn.models.Scatter import Scatter
from lambdaforge.nn.pooling.sparse.SparsePooling import SparsePooling


class SparseMeanPooling(SparsePooling):
    """Average rows in each sparse group; empty groups are zero."""

    def forward(
        self,
        x: torch.Tensor,
        group_index: torch.Tensor,
        num_groups: int | None = None,
    ) -> torch.Tensor:
        """Return shape ``(num_groups, D)``."""
        group_index, count = self.validate(x, group_index, num_groups)
        return Scatter.mean(x, group_index, count)
