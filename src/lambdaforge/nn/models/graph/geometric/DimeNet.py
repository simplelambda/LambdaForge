"""Directional edge encoder with bilinear or DimeNet++ factorized interactions."""

from __future__ import annotations

import torch
from torch import nn

from lambdaforge.nn.models.graph.geometric.DimeNetBasis import DimeNetBasis
from lambdaforge.nn.models.graph.geometric.DimeNetInteraction import DimeNetInteraction
from lambdaforge.nn.models.graph.geometric.DimeNetPlusPlusInteraction import (
    DimeNetPlusPlusInteraction,
)
from lambdaforge.nn.models.graph.geometric.GraphTriplets import GraphTriplets
from lambdaforge.nn.models.graph.GraphEdgeData import GraphEdgeData
from lambdaforge.nn.models.graph.validation import aligned, channels, graph
from lambdaforge.nn.models.Model import Model
from lambdaforge.nn.models.Scatter import Scatter


class DimeNet(Model):
    """Generic directional node representation, not a pretrained molecular energy model.

    Explicit distances, triplets and angles prevent hidden graph rebuilding. Arbitrary
    node/edge scalar descriptors replace chemical-number embeddings. interaction_type
    selects the bilinear equation or the DimeNet++ factorized equation. Per-block
    radial edge readouts sum by destination; an explicit node root retains isolated
    descriptors. Standard Torch initialization and this readout are generic adaptations,
    not published QM9 training/pretraining parity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        num_radial: int = 6,
        num_spherical: int = 7,
        cutoff: float = 5.0,
        edge_channels: int = 0,
        interaction_type: str = "factorized",
        interaction_channels: int = 32,
        basis_channels: int = 8,
        max_triplet_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        super().__init__()
        channels(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            max_triplet_bytes=max_triplet_bytes,
        )
        if interaction_type not in {"bilinear", "factorized"}:
            raise ValueError("interaction_type must be bilinear or factorized.")
        if (
            isinstance(edge_channels, bool)
            or not isinstance(edge_channels, int)
            or edge_channels < 0
        ):
            raise ValueError("edge_channels must be non-negative integer.")
        self.in_channels, self.edge_channels = in_channels, edge_channels
        self.hidden_channels, self.max_triplet_bytes = hidden_channels, max_triplet_bytes
        self.basis = DimeNetBasis(num_radial, num_spherical, cutoff)
        self.radial_embedding = nn.Linear(num_radial, hidden_channels)
        self.embedding = nn.Sequential(
            nn.Linear(2 * in_channels + hidden_channels + edge_channels, hidden_channels), nn.SiLU()
        )
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if interaction_type == "bilinear":
                self.layers.append(
                    DimeNetInteraction(hidden_channels, num_radial, num_spherical, basis_channels)
                )
            else:
                self.layers.append(
                    DimeNetPlusPlusInteraction(
                        hidden_channels,
                        num_radial,
                        num_spherical,
                        interaction_channels,
                        basis_channels,
                    )
                )
        self.radial_readouts = nn.ModuleList(
            nn.Linear(num_radial, hidden_channels, bias=False) for _ in range(num_layers + 1)
        )
        self.node_readouts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, out_channels, bias=False),
            )
            for _ in range(num_layers + 1)
        )
        self.root = nn.Linear(in_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        distances: torch.Tensor,
        triplets: GraphTriplets,
        angles: torch.Tensor,
        edge_features: torch.Tensor | None = None,
        *,
        distance_weights: torch.Tensor | None = None,
        message_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return invariant scalar (N,out_channels) directed-triplet representations."""
        self._graph_inputs(x, edge_index, triplets, edge_features)
        if distances.shape != (edge_index.shape[1],):
            raise ValueError("distances must align with directed edges.")
        radial, spherical = self.basis(
            distances.to(x), angles.to(x), triplets, distance_weights=distance_weights
        )
        return self.forward_from_basis(
            x,
            edge_index,
            radial,
            spherical,
            triplets,
            edge_features,
            message_weights=message_weights,
        )

    def _graph_inputs(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        triplets: GraphTriplets,
        edge_features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        edges, features = graph(x, edge_index, self.in_channels, edge_features, self.edge_channels)
        incoming, outgoing = triplets.validate(edges.shape[1], x.device)
        if (
            (edges[1, incoming] != edges[0, outgoing]) | (edges[0, incoming] == edges[1, outgoing])
        ).any():
            raise ValueError("Triplets must follow supplied edges k->j->i, k != i.")
        if (edges[0] == edges[1]).any():
            raise ValueError("Directional edges must not include self-loops.")
        # Lower bound on main triplet tensors, not total autograd/optimizer memory.
        width = max(self.hidden_channels, self.basis.num_radial * self.basis.num_spherical)
        if len(incoming) * width * x.element_size() > self.max_triplet_bytes:
            raise MemoryError("Directional triplet activations exceed max_triplet_bytes.")
        return edges, features

    def forward_from_basis(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        radial: torch.Tensor,
        spherical: torch.Tensor,
        triplets: GraphTriplets,
        edge_features: torch.Tensor | None = None,
        *,
        message_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Use explicit aligned native bases, e.g. distinct per-relation support policies.

        Radial (E,R), spherical (T,L*R). g_e multiplies initial states, each
        interaction's final state and each edge readout; triplets additionally
        carry g_in*g_out. Fractional weights therefore compound with path/depth.
        No detach or topology changes. At all-zero weights, only root(x) plus
        the legacy zero-input node-readout constants remain, not relation messages.
        """
        edges, features = self._graph_inputs(x, edge_index, triplets, edge_features)
        radial = aligned(radial, (edges.shape[1], self.basis.num_radial), x, "radial")
        spherical = aligned(
            spherical,
            (len(triplets.incoming), self.basis.num_radial * self.basis.num_spherical),
            x,
            "spherical",
        )
        weights = GraphEdgeData.normalize_message_weights(
            message_weights, edge_count=edges.shape[1], reference=x
        )
        parts = [x[edges[1]], x[edges[0]], torch.nn.functional.silu(self.radial_embedding(radial))]
        if features is not None:
            parts.append(features)
        states = self.embedding(torch.cat(parts, -1))
        if weights is not None:
            states = states * weights.to(states)[:, None]
        output = self.root(x)
        for index in range(len(self.layers) + 1):
            messages = states * self.radial_readouts[index](radial)
            if weights is not None:
                messages = messages * weights.to(messages)[:, None]
            gathered = Scatter.sum(messages, edges[1], len(x))
            output = output + self.node_readouts[index](gathered)
            if index < len(self.layers):
                states = self.layers[index](
                    states, radial, spherical, triplets, message_weights=message_weights
                )
        return output
