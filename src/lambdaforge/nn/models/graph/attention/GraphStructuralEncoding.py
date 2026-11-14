"""Explicit, tie-independent shortest-path codes on a supplied directed multigraph."""

from collections import deque
from dataclasses import dataclass

import torch

from lambdaforge.nn.models.graph.GraphEdgeIndex import GraphEdgeIndex
from lambdaforge.nn.models.graph.validation import channels


@dataclass(frozen=True)
class GraphStructuralEncoding:
    """Immutable tensor references: spatial (N,N) and directed incidence degrees (N,)."""

    spatial_indices: torch.Tensor
    in_degree: torch.Tensor
    out_degree: torch.Tensor

    @classmethod
    def from_edges(
        cls,
        edge_index: torch.Tensor,
        num_nodes: int,
        num_spatial_codes: int = 32,
        max_pairwise_bytes: int = 64 * 1024 * 1024,
    ) -> "GraphStructuralEncoding":
        """Compute CPU BFS distances, never select an arbitrary shortest path's attributes.

        Code 0=unreachable, d+1=hop distance saturated at last code. Degrees count
        directed incidences, including parallel relations. This is explicit offline
        O(N*(N+E)) preprocessing, not a hidden forward-time topology operation.
        """
        channels(num_spatial_codes=num_spatial_codes, max_pairwise_bytes=max_pairwise_bytes)
        if isinstance(num_nodes, bool) or not isinstance(num_nodes, int) or num_nodes < 0:
            raise ValueError("num_nodes must be a non-negative integer.")
        if num_spatial_codes < 2:
            raise ValueError("At least two spatial codes are required.")
        if num_nodes * num_nodes * 8 > max_pairwise_bytes:
            raise MemoryError("Shortest-path matrix exceeds max_pairwise_bytes.")
        edges = GraphEdgeIndex.normalize(
            edge_index, device=torch.device("cpu"), num_nodes=num_nodes
        )
        adjacency: list[set[int]] = [set() for _ in range(num_nodes)]
        indegree = torch.bincount(edges[1], minlength=num_nodes)
        outdegree = torch.bincount(edges[0], minlength=num_nodes)
        for source, target in edges.t().tolist():
            adjacency[source].add(target)
        spatial = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
        for source in range(num_nodes):
            hops = {source: 0}
            pending = deque([source])
            while pending:
                current = pending.popleft()
                for target in adjacency[current]:
                    if target not in hops:
                        hops[target] = hops[current] + 1
                        pending.append(target)
            for target, distance in hops.items():
                spatial[target, source] = min(distance + 1, num_spatial_codes - 1)
        return cls(spatial, indegree, outdegree)
