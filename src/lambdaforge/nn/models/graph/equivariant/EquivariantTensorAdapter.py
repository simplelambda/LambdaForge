"""Validated injection boundary for optional higher-order equivariant providers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from lambdaforge.nn.models.Model import Model


class EquivariantTensorAdapter(Model):
    """Wrap e3nn-like modules without imposing their compiled dependency on users."""

    def __init__(self, module: nn.Module, output_key: str | None = None) -> None:
        super().__init__()
        if not isinstance(module, nn.Module):
            raise TypeError("module must be a torch.nn.Module.")
        if output_key is not None and not output_key:
            raise ValueError("output_key cannot be empty.")
        self.module = module
        self.output_key = output_key

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        coordinates: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Call an injected provider after validating graph and Cartesian shapes."""
        if node_features.ndim < 2:
            raise ValueError("node_features must start with a node dimension.")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, edges).")
        if coordinates.shape != (node_features.shape[0], 3):
            raise ValueError("coordinates must have shape (nodes, 3).")
        if coordinates.device != node_features.device:
            raise ValueError("coordinates and node_features must share a device.")
        result = self.module(node_features, edge_index, coordinates, **kwargs)
        if isinstance(result, Mapping):
            key = self.output_key or "features"
            if key not in result:
                raise KeyError(f"Equivariant provider output has no {key!r} tensor.")
            result = result[key]
        if not torch.is_tensor(result) or result.shape[0] != node_features.shape[0]:
            raise TypeError("Equivariant provider must return one tensor row per node.")
        return result
