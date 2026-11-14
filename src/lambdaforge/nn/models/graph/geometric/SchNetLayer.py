"""SchNet-style continuous-filter residual interaction on a supplied graph."""

from __future__ import annotations

import torch
from torch import nn

from lambdaforge.nn.models.graph.geometric.CosineCutoff import CosineCutoff
from lambdaforge.nn.models.graph.geometric.GaussianRadialBasis import GaussianRadialBasis
from lambdaforge.nn.models.graph.GraphEdgeData import GraphEdgeData
from lambdaforge.nn.models.graph.validation import aligned, graph
from lambdaforge.nn.models.Scatter import Scatter


class SchNetLayer(nn.Module):
    """Learn radial filters, multiply projected sender channels, sum and update.

    Takes explicit edge distances, not hidden coordinates: a consumer controls which
    distance source/gate is used. Optional scalar edge attributes condition filters.
    cutoff=None preserves all supplied edges (including long-range relations).
    This generic node interaction uses SiLU; it is not the molecular energy readout
    or an exact reproduction of the original shifted-softplus training recipe.
    """

    def __init__(
        self,
        channels: int,
        num_radial: int = 32,
        radius: float = 5.0,
        cutoff: float | None = None,
        edge_channels: int = 0,
    ) -> None:
        super().__init__()
        from lambdaforge.nn.models.graph.validation import channels as validate_channels

        validate_channels(channels=channels)
        if (
            isinstance(edge_channels, bool)
            or not isinstance(edge_channels, int)
            or edge_channels < 0
        ):
            raise ValueError("edge_channels must be a non-negative integer.")
        self.channels, self.edge_channels = channels, edge_channels
        self.radial_basis = GaussianRadialBasis(num_radial, radius)
        self.cutoff = CosineCutoff(cutoff) if cutoff is not None else None
        self.input_projection = nn.Linear(channels, channels, bias=False)
        self.filter = nn.Sequential(
            nn.Linear(num_radial + edge_channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        self.update = nn.Sequential(
            nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, channels)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        distances: torch.Tensor,
        edge_features: torch.Tensor | None = None,
        *,
        distance_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return residual scalar node states (N, channels)."""
        edges, features = graph(x, edge_index, self.channels, edge_features, self.edge_channels)
        distances = aligned(distances, (edges.shape[1],), x, "distances")
        radial, support = self.radial_inputs(distances, distance_weights=distance_weights)
        return self.forward_from_radial(
            x,
            edges,
            radial,
            support if self.cutoff is not None or distance_weights is not None else None,
            features,
        )

    def forward_from_radial(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        radial: torch.Tensor,
        support: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Use (E,R) features and optional (E,) complete filter multipliers.

        Supports mixing native per-relation bases/cutoffs before ONE node update.
        No hidden cutoff is reapplied. None means identity, not an inferred mask.
        Zero support removes edge messages; root/zero-input update bias remains.
        """
        edges, features = graph(x, edge_index, self.channels, edge_features, self.edge_channels)
        radial = aligned(radial, (edges.shape[1], self.radial_basis.num_radial), x, "radial")
        if support is not None:
            support = aligned(support, (edges.shape[1],), x, "support")
        filters = self.filter(radial if features is None else torch.cat((radial, features), -1))
        if support is not None:
            filters = filters * support[:, None]
        messages = self.input_projection(x)[edges[0]] * filters
        return x + self.update(Scatter.sum(messages, edges[1], len(x)))

    def radial_inputs(
        self, distances: torch.Tensor, *, distance_weights: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return radial features/support, jointly controlled without falsifying geometry.

        For scalar/(E,) g: radial=g*R(d), support=1+g*(cutoff(d)-1).
        At g=0 no distance dependence remains; filter biases/other edge attributes
        and node/root transformations remain. This is not relation removal.
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
