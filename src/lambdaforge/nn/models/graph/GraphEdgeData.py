"""Validation and alignment of optional data stored on sparse graph edges."""

from __future__ import annotations

import math
from numbers import Real

import torch

from lambdaforge.nn.models.graph.GraphEdgeIndex import GraphEdgeIndex
from lambdaforge.nn.models.graph.GraphSelfLoopFill import GraphSelfLoopFill
from lambdaforge.nn.models.Scatter import Scatter


class GraphEdgeData:
    """Centralize feature, relation and self-loop alignment contracts.

    Graph layers use a directed ``source -> destination`` edge list. Every
    edge-level tensor must keep exactly the same row order, including after
    existing self-loops are replaced by one canonical self-loop per node.
    """

    @classmethod
    def normalize_features(
        cls,
        edge_features: torch.Tensor | None,
        *,
        edge_channels: int,
        edge_count: int,
        reference: torch.Tensor,
        name: str = "edge_features",
    ) -> torch.Tensor | None:
        """Validate optional real-valued edge features and match node dtype/device."""
        if isinstance(edge_channels, bool) or not isinstance(edge_channels, int):
            raise TypeError("edge_channels must be an integer.")
        if edge_channels < 0:
            raise ValueError("edge_channels must be non-negative.")
        if edge_features is None:
            if edge_channels:
                raise ValueError(f"{name} is required because edge_channels={edge_channels}.")
            return None
        if not isinstance(edge_features, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor or None.")
        expected = (edge_count, edge_channels)
        if edge_features.ndim != 2 or tuple(edge_features.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}.")
        if edge_features.dtype == torch.bool or edge_features.is_complex():
            raise TypeError(f"{name} must contain real numeric values.")
        if edge_channels == 0:
            return None
        return edge_features.to(device=reference.device, dtype=reference.dtype)

    @classmethod
    def normalize_relation_types(
        cls,
        edge_types: torch.Tensor,
        *,
        edge_count: int,
        num_relations: int,
        device: torch.device,
        name: str = "edge_types",
    ) -> torch.Tensor:
        """Validate one integer relation identifier for every directed edge."""
        if isinstance(num_relations, bool) or not isinstance(num_relations, int):
            raise TypeError("num_relations must be an integer.")
        if num_relations < 1:
            raise ValueError("num_relations must be positive.")
        if not isinstance(edge_types, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if edge_types.ndim != 1 or edge_types.shape[0] != edge_count:
            raise ValueError(f"{name} must have shape ({edge_count},).")
        if edge_types.dtype not in GraphEdgeIndex.INTEGER_DTYPES:
            raise TypeError(f"{name} must use an integer dtype (Boolean is not accepted).")
        normalized = edge_types.to(device=device, dtype=torch.long)
        if normalized.numel() and (
            int(normalized.min()) < 0 or int(normalized.max()) >= num_relations
        ):
            raise IndexError(f"{name} contains a relation outside [0, num_relations).")
        return normalized

    @classmethod
    def replace_self_loops(
        cls,
        edge_index: torch.Tensor,
        *,
        device: torch.device,
        num_nodes: int,
        edge_features: torch.Tensor | None = None,
        fill: GraphSelfLoopFill | str | float = GraphSelfLoopFill.ZERO,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Replace all existing self-loops with one canonical loop per node."""
        normalized = GraphEdgeIndex.normalize(
            edge_index,
            device=device,
            num_nodes=num_nodes,
        )
        source, destination = normalized
        non_self = source != destination
        nodes = torch.arange(num_nodes, device=device)
        routed = torch.stack(
            (
                torch.cat((source[non_self], nodes)),
                torch.cat((destination[non_self], nodes)),
            )
        )
        if edge_features is None:
            return routed, None
        if edge_features.ndim != 2 or edge_features.shape[0] != source.shape[0]:
            raise ValueError("edge_features must have shape (E, edge_channels).")
        aligned = edge_features.to(device=device)
        self_features = cls._self_loop_features(
            aligned[non_self],
            destination[non_self],
            num_nodes,
            fill,
        )
        return routed, torch.cat((aligned[non_self], self_features))

    @staticmethod
    def _self_loop_features(
        edge_features: torch.Tensor,
        destination: torch.Tensor,
        num_nodes: int,
        fill: GraphSelfLoopFill | str | float,
    ) -> torch.Tensor:
        if isinstance(fill, bool):
            raise TypeError("self-loop fill must be 'zero', 'mean', or a real number.")
        if isinstance(fill, Real):
            if not math.isfinite(float(fill)):
                raise ValueError("Numeric self-loop fill must be finite.")
            return edge_features.new_full(
                (num_nodes, edge_features.shape[1]),
                float(fill),
            )
        policy = GraphSelfLoopFill(fill)
        if policy is GraphSelfLoopFill.ZERO:
            return edge_features.new_zeros((num_nodes, edge_features.shape[1]))
        return Scatter.mean(edge_features, destination, num_nodes)
