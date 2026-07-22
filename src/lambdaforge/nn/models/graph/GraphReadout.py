"""Composable node-encoder, sparse-pooling and prediction model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from lambdaforge.nn.models.Model import Model
from lambdaforge.nn.pooling.sparse.SparsePooling import SparsePooling


class GraphReadout(Model):
    """Turn any node encoder into a graph-level model without task assumptions."""

    def __init__(
        self,
        encoder: nn.Module,
        pooling: SparsePooling,
        head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, nn.Module):
            raise TypeError("encoder must be a torch.nn.Module.")
        if not isinstance(pooling, SparsePooling):
            raise TypeError("pooling must subclass SparsePooling.")
        if head is not None and not isinstance(head, nn.Module):
            raise TypeError("head must be a torch.nn.Module or None.")
        self.encoder = encoder
        self.pooling = pooling
        self.head = head if head is not None else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        group_index: torch.Tensor,
        num_groups: int | None = None,
        **encoder_kwargs: Any,
    ) -> torch.Tensor:
        """Encode nodes, pool by graph/sample and apply the configured head."""
        encoded = self.encoder(x, edge_index, **encoder_kwargs)
        if not torch.is_tensor(encoded) or encoded.ndim != 2:
            raise TypeError("encoder must return a node tensor with shape (N, D).")
        return self.head(self.pooling(encoded, group_index, num_groups))
