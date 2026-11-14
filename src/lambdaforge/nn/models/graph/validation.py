"""Shared validation for reusable edge and geometric operators, not graph construction."""

from __future__ import annotations

import math

import torch

from lambdaforge.nn.models.graph.GraphEdgeData import GraphEdgeData
from lambdaforge.nn.models.graph.GraphEdgeIndex import GraphEdgeIndex


def channels(**values: int) -> None:
    """Reject non-positive or non-integer architectural widths eagerly."""
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")


def positive(value: float, name: str) -> float:
    """Validate a finite positive numerical scale."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be real.")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return float(value)


def graph(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    width: int,
    edge_features: torch.Tensor | None = None,
    edge_channels: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Validate floating node features and aligned directed edges; preserve multiplicity."""
    if x.ndim != 2 or x.shape[1] != width or not x.is_floating_point():
        raise ValueError(f"x must be floating point with shape (N, {width}).")
    routed = GraphEdgeIndex.normalize(edge_index, device=x.device, num_nodes=len(x))
    features = GraphEdgeData.normalize_features(
        edge_features, edge_channels=edge_channels, edge_count=routed.shape[1], reference=x
    )
    return routed, features


def aligned(
    value: torch.Tensor, shape: tuple[int, ...], reference: torch.Tensor, name: str
) -> torch.Tensor:
    """Reject malformed/nonfinite real geometric data and align dtype/device."""
    if value.shape != shape or not value.is_floating_point():
        raise ValueError(f"{name} must be floating point with shape {shape}.")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite.")
    return value.to(reference)


def edge_geometry(
    coordinates: torch.Tensor, edge_index: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return lengths and receiver-to-sender unit vectors on existing edges only.

    Coordinates are (N, 3); edges are source -> destination. Zero-length edges
    are invalid for directions and must be removed/handled explicitly by callers.
    No neighbours, coordinate updates or domain-specific distance units are inferred.
    """
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape (N, 3).")
    coordinates = aligned(coordinates, tuple(coordinates.shape), coordinates, "coordinates")
    routed = GraphEdgeIndex.normalize(
        edge_index, device=coordinates.device, num_nodes=len(coordinates)
    )
    displacement = coordinates[routed[0]] - coordinates[routed[1]]
    lengths = torch.linalg.vector_norm(displacement, dim=-1)
    if (lengths <= 0).any():
        raise ValueError("Zero-length directed edges have no defined direction.")
    return lengths, displacement / lengths.unsqueeze(-1)
