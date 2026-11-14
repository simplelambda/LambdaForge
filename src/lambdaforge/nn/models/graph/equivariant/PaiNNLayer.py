"""PaiNN radial interaction followed by invariant scalar/vector mixing."""

from __future__ import annotations

import torch
from torch import nn

from lambdaforge.nn.models.graph.geometric.CosineCutoff import CosineCutoff
from lambdaforge.nn.models.graph.geometric.SineRadialBasis import SineRadialBasis
from lambdaforge.nn.models.graph.GraphEdgeData import GraphEdgeData
from lambdaforge.nn.models.graph.validation import aligned, graph, positive
from lambdaforge.nn.models.graph.validation import channels as validate_channels
from lambdaforge.nn.models.Scatter import Scatter


class PaiNNLayer(nn.Module):
    """Update (s[N,C], V[N,C,3]) with explicit distances and unit polar directions.

    Uses sinusoidal radial features and optional cosine cutoff. Vectors are not
    coordinates and no atomic positions are modified. Edge features are optional
    authored scalar conditioning; relation/gate policy belongs to the caller.
    """

    def __init__(
        self,
        channels: int,
        num_radial: int = 20,
        radius: float = 5.0,
        cutoff: float | None = None,
        edge_channels: int = 0,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        validate_channels(channels=channels)
        if (
            isinstance(edge_channels, bool)
            or not isinstance(edge_channels, int)
            or edge_channels < 0
        ):
            raise ValueError("edge_channels must be a non-negative integer.")
        self.channels, self.edge_channels = channels, edge_channels
        self.epsilon = positive(epsilon, "epsilon")
        self.radial_basis = SineRadialBasis(num_radial, radius)
        self.cutoff = CosineCutoff(cutoff) if cutoff is not None else None
        self.filter = nn.Linear(num_radial + edge_channels, 3 * channels)
        self.context = nn.Sequential(
            nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, 3 * channels)
        )
        self.vector_mix = nn.Linear(channels, 2 * channels, bias=False)
        self.scalar_mix = nn.Sequential(
            nn.Linear(2 * channels, channels), nn.SiLU(), nn.Linear(channels, 3 * channels)
        )

    def forward(
        self,
        scalars: torch.Tensor,
        vectors: torch.Tensor,
        edge_index: torch.Tensor,
        distances: torch.Tensor,
        directions: torch.Tensor,
        edge_features: torch.Tensor | None = None,
        *,
        distance_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return updated invariant scalars and equivariant polar vectors."""
        edges, features = graph(
            scalars, edge_index, self.channels, edge_features, self.edge_channels
        )
        vectors = aligned(vectors, (len(scalars), self.channels, 3), scalars, "vectors")
        distances = aligned(distances, (edges.shape[1],), scalars, "distances")
        radial, support = self.radial_inputs(distances, distance_weights=distance_weights)
        return self.forward_from_radial(
            scalars,
            vectors,
            edges,
            radial,
            directions,
            support if self.cutoff is not None or distance_weights is not None else None,
            features,
        )

    def forward_from_radial(
        self,
        scalars: torch.Tensor,
        vectors: torch.Tensor,
        edge_index: torch.Tensor,
        radial: torch.Tensor,
        directions: torch.Tensor,
        support: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Consume explicit radial (E,R), directions (E,3), optional support (E,).

        No cutoff is applied again; per-relation native bases/supports can be
        combined before one interaction/local update. Zero support removes the
        whole edge's scalar/vector messages, not the node's own local mixing.
        """
        edges, features = graph(
            scalars, edge_index, self.channels, edge_features, self.edge_channels
        )
        vectors = aligned(vectors, (len(scalars), self.channels, 3), scalars, "vectors")
        radial = aligned(radial, (edges.shape[1], self.radial_basis.num_radial), scalars, "radial")
        if support is not None:
            support = aligned(support, (edges.shape[1],), scalars, "support")
        directions = aligned(directions, (edges.shape[1], 3), scalars, "directions")
        # Unit vectors rounded to fp16/bf16 are not unit to 1e-4. Accumulate
        # norms in at least fp32 and bound the check by input precision.
        tolerance = max(1e-4, 2 * torch.finfo(directions.dtype).eps)
        squared_norm = (
            (
                directions.float()
                if directions.dtype in {torch.float16, torch.bfloat16}
                else directions
            )
            .square()
            .sum(-1)
        )
        if not torch.allclose(
            squared_norm, torch.ones_like(squared_norm), atol=tolerance, rtol=tolerance
        ):
            raise ValueError("directions must be unit vectors; zero-length geometry is invalid.")
        filters = self.filter(radial if features is None else torch.cat((radial, features), -1))
        if support is not None:
            filters = filters * support[:, None]
        source, target = edges
        ds, dv_direction, dv_state = (self.context(scalars)[source] * filters).chunk(3, -1)
        scalars = scalars + Scatter.sum(ds, target, len(scalars))
        vectors = vectors + Scatter.sum(
            dv_direction[..., None] * directions[:, None, :]
            + dv_state[..., None] * vectors[source],
            target,
            len(scalars),
        )
        a, b = self.vector_mix(vectors.transpose(-1, -2)).transpose(-1, -2).chunk(2, -2)
        norms = (a.square().sum(-1) + self.epsilon).sqrt()
        delta_s, delta_v, coupling = self.scalar_mix(torch.cat((scalars, norms), -1)).chunk(3, -1)
        return scalars + delta_s + coupling * (a * b).sum(-1), vectors + delta_v[..., None] * b

    def radial_inputs(
        self, distances: torch.Tensor, *, distance_weights: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return sinusoidal features/support with distance-only source control.

        g=0 gives radial zero/support one; scalar filter biases/edge attributes
        remain, and separately supplied polar directions are NOT disabled.
        g=1 is the original basis/cutoff. Geometry is validated even when closed.
        """
        radial = self.radial_basis(distances)
        support = self.cutoff(distances) if self.cutoff is not None else torch.ones_like(distances)
        weights = GraphEdgeData.normalize_message_weights(
            distance_weights,
            edge_count=len(distances),
            reference=distances,
            name="distance_weights",
            allow_scalar=True,
        )
        if weights is not None:
            radial = radial * weights[:, None]
            support = (1 - weights) + weights * support
        return radial, support
