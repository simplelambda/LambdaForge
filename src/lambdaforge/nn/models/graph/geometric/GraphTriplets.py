"""Bounded typed-edge paths without immediate returns or implicit new neighbours."""

from dataclasses import dataclass

import torch

from lambdaforge.nn.models.graph.GraphEdgeIndex import GraphEdgeIndex
from lambdaforge.nn.models.graph.validation import channels, edge_geometry


@dataclass(frozen=True)
class GraphTriplets:
    """Aligned incoming k->j and outgoing j->i edge IDs; parallel relations are retained."""

    incoming: torch.Tensor
    outgoing: torch.Tensor

    @classmethod
    def from_edges(
        cls, edge_index: torch.Tensor, num_nodes: int, max_triplets: int = 1_000_000
    ) -> "GraphTriplets":
        """Explicit CPU preprocessing; reject self-loops, preserve edge order and multiplicity."""
        channels(max_triplets=max_triplets)
        if isinstance(num_nodes, bool) or not isinstance(num_nodes, int) or num_nodes < 0:
            raise ValueError("num_nodes must be non-negative integer.")
        edges = GraphEdgeIndex.normalize(
            edge_index, device=torch.device("cpu"), num_nodes=num_nodes
        )
        if (edges[0] == edges[1]).any():
            raise ValueError("Directional triplets require edges without self-loops.")
        arrivals: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
        for edge, (source, target) in enumerate(edges.t().tolist()):
            arrivals[target].append((edge, source))
        incoming: list[int] = []
        outgoing: list[int] = []
        for edge, (center, target) in enumerate(edges.t().tolist()):
            for first, source in arrivals[center]:
                if source == target:
                    continue
                if len(incoming) >= max_triplets:
                    raise MemoryError("Directed triplet count exceeds max_triplets.")
                incoming.append(first)
                outgoing.append(edge)
        return cls(
            torch.tensor(incoming, dtype=torch.long, device=edge_index.device),
            torch.tensor(outgoing, dtype=torch.long, device=edge_index.device),
        )

    def validate(self, edge_count: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate public edge-ID tensors before indexing a message/basis tensor."""
        indices = torch.stack((self.incoming, self.outgoing))
        normalized = GraphEdgeIndex.normalize(indices, device=device, num_nodes=edge_count)
        return normalized[0], normalized[1]

    def angles(self, coordinates: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Angle at j between j->k and j->i, stable atan2 on nonzero geometry."""
        _, directions = edge_geometry(coordinates, edge_index)
        incoming, outgoing = self.validate(edge_index.shape[1], coordinates.device)
        edges = GraphEdgeIndex.normalize(
            edge_index, device=coordinates.device, num_nodes=len(coordinates)
        )
        if (
            (edges[1, incoming] != edges[0, outgoing]) | (edges[0, incoming] == edges[1, outgoing])
        ).any():
            raise ValueError("Triplets must be connected k->j->i with k != i.")
        to_k, to_i = directions[incoming], -directions[outgoing]
        cross = torch.linalg.cross(to_k, to_i, dim=-1)
        return torch.atan2(torch.linalg.vector_norm(cross, dim=-1), (to_k * to_i).sum(-1))
