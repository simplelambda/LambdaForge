"""Shape-safe normalization adapter for node-feature matrices."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from lambdaforge.nn.ComponentRegistry import ComponentRegistry
from lambdaforge.nn.normalizations.BatchNorm import BatchNorm
from lambdaforge.nn.normalizations.ChannelLayerNorm import ChannelLayerNorm
from lambdaforge.nn.normalizations.InstanceNorm import InstanceNorm
from lambdaforge.nn.normalizations.L2Norm import L2Norm
from lambdaforge.nn.normalizations.Normalization import Normalization
from lambdaforge.nn.normalizations.ScaleNorm import ScaleNorm


class GraphNormalization(nn.Module):
    """Apply a normalization that preserves `(nodes, features)` layout.

    Framework normalizations whose dimensional variants are unambiguous are
    checked eagerly. Custom `Normalization` subclasses remain extensible but
    must accept a two-dimensional node-feature matrix and preserve its shape.
    """

    def __init__(
        self,
        specification: type[Normalization] | str,
        num_features: int,
        options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        normalization_type = ComponentRegistry.resolve_normalization(specification)
        self.normalization = normalization_type(num_features, **(options or {}))
        self.num_features = int(num_features)
        self._validate_configuration()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize one node-feature matrix without changing its layout."""
        if x.ndim != 2 or x.shape[1] != self.num_features:
            raise ValueError(f"Graph normalization expects shape (N, {self.num_features}).")
        output = self.normalization(x)
        if not isinstance(output, torch.Tensor) or output.shape != x.shape:
            raise ValueError(
                "Graph normalizations must return a tensor with the same (nodes, features) shape."
            )
        return output

    def _validate_configuration(self) -> None:
        normalization = self.normalization
        if isinstance(normalization, InstanceNorm):
            raise ValueError(
                "InstanceNorm is incompatible with graph tensors shaped (nodes, features); "
                "use LayerNorm, RMSNorm, GroupNorm, BatchNorm(dim=1), or a custom graph-aware "
                "normalization."
            )
        if isinstance(normalization, BatchNorm) and not isinstance(
            normalization.norm, nn.BatchNorm1d
        ):
            raise ValueError("Graph tensors require BatchNorm(dim=1).")
        if isinstance(normalization, ChannelLayerNorm) and normalization.channel_dim not in (1, -1):
            raise ValueError("Graph tensors require ChannelLayerNorm(channel_dim=1 or -1).")
        if isinstance(normalization, (L2Norm, ScaleNorm)) and normalization.dim not in (1, -1):
            raise ValueError(
                "Graph tensors require vector normalization along the feature dimension "
                "(dim=1 or -1)."
            )
