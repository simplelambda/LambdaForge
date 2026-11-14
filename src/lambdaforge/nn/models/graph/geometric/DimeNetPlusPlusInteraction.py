"""Factorized directional interactions with DimeNet++ down/up projections."""

import torch
from torch import nn
from torch.nn import functional as F

from lambdaforge.nn.models.graph.geometric.DimeNetInteraction import DimeNetInteraction
from lambdaforge.nn.models.graph.geometric.GraphTriplets import GraphTriplets
from lambdaforge.nn.models.graph.GraphEdgeData import GraphEdgeData
from lambdaforge.nn.models.graph.validation import channels
from lambdaforge.nn.models.Scatter import Scatter


class DimeNetPlusPlusInteraction(DimeNetInteraction):
    """Replace DimeNet's bilinear tensor with basis and interaction bottlenecks.

    The different equation is explicit; this is not an alias for the bilinear block.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_radial: int,
        num_spherical: int,
        interaction_channels: int = 32,
        basis_channels: int = 8,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
    ) -> None:
        super().__init__(
            hidden_channels,
            num_radial,
            num_spherical,
            basis_channels,
            num_before_skip,
            num_after_skip,
        )
        channels(interaction_channels=interaction_channels, basis_channels=basis_channels)
        del self.bilinear
        self.radial = nn.Sequential(
            nn.Linear(num_radial, basis_channels, bias=False),
            nn.Linear(basis_channels, hidden_channels, bias=False),
        )
        self.spherical = nn.Sequential(
            nn.Linear(num_radial * num_spherical, basis_channels, bias=False),
            nn.Linear(basis_channels, interaction_channels, bias=False),
        )
        self.down = nn.Linear(hidden_channels, interaction_channels, bias=False)
        self.up = nn.Linear(interaction_channels, hidden_channels, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        radial: torch.Tensor,
        spherical: torch.Tensor,
        triplets: GraphTriplets,
        *,
        message_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Filter, down-project, multiply spherical features, aggregate then up-project."""
        first, second, r, s = self._validate(x, radial, spherical, triplets)
        send = F.silu(self.down(F.silu(self.incoming(x)) * self.radial(r)))
        messages = send[first] * self.spherical(s)
        weights = GraphEdgeData.normalize_message_weights(
            message_weights, edge_count=len(x), reference=x
        )
        if weights is not None:
            messages = messages * (weights[first] * weights[second])[:, None]
        interaction = F.silu(self.up(Scatter.sum(messages, second, len(x))))
        result = self._finish(x, interaction)
        return result if weights is None else result * weights[:, None]
