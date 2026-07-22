"""Sparse maximum pooling."""

from __future__ import annotations

import torch

from lambdaforge.nn.pooling.sparse.SparsePooling import SparsePooling


class SparseMaxPooling(SparsePooling):
    """Take featurewise maxima per group and map empty groups to zero."""

    def forward(
        self,
        x: torch.Tensor,
        group_index: torch.Tensor,
        num_groups: int | None = None,
    ) -> torch.Tensor:
        """Return finite shape ``(num_groups, D)`` even with empty groups."""
        group_index, count = self.validate(x, group_index, num_groups)
        output = x.new_full((count, x.shape[1]), float("-inf"))
        expanded = group_index.unsqueeze(-1).expand_as(x)
        output = output.scatter_reduce(
            0,
            expanded,
            x,
            reduce="amax",
            include_self=True,
        )
        return torch.where(torch.isfinite(output), output, torch.zeros_like(output))
