"""Bilinear directional edge-state interaction from DimeNet."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from lambdaforge.nn.models.graph.geometric.GraphTriplets import GraphTriplets
from lambdaforge.nn.models.graph.GraphEdgeData import GraphEdgeData
from lambdaforge.nn.models.graph.validation import aligned, channels
from lambdaforge.nn.models.Scatter import Scatter


class DimeNetInteraction(nn.Module):
    """Bilinearly combine spherical features and incoming edge messages.

    This block consumes precomputed aligned radial/spherical bases. Its output
    lives on directed edges, not Cartesian vector channels or graph-level energies.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_radial: int,
        num_spherical: int,
        bilinear_channels: int = 8,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
    ) -> None:
        super().__init__()
        channels(
            hidden_channels=hidden_channels,
            num_radial=num_radial,
            num_spherical=num_spherical,
            bilinear_channels=bilinear_channels,
        )
        for count in (num_before_skip, num_after_skip):
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("Residual block counts must be non-negative integers.")
        self.hidden_channels, self.num_radial, self.num_spherical = (
            hidden_channels,
            num_radial,
            num_spherical,
        )
        self.radial: nn.Module = nn.Linear(num_radial, hidden_channels, bias=False)
        self.spherical: nn.Module = nn.Linear(
            num_radial * num_spherical, bilinear_channels, bias=False
        )
        self.incoming = nn.Linear(hidden_channels, hidden_channels)
        self.outgoing = nn.Linear(hidden_channels, hidden_channels)
        self.bilinear = nn.Parameter(
            torch.empty(hidden_channels, bilinear_channels, hidden_channels)
        )
        nn.init.xavier_uniform_(self.bilinear)
        self.before = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
            )
            for _ in range(num_before_skip)
        )
        self.skip = nn.Linear(hidden_channels, hidden_channels)
        self.after = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
            )
            for _ in range(num_after_skip)
        )

    def _validate(
        self,
        x: torch.Tensor,
        radial: torch.Tensor,
        spherical: torch.Tensor,
        triplets: GraphTriplets,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or x.shape[1] != self.hidden_channels or not x.is_floating_point():
            raise ValueError("edge states must be floating (E,hidden_channels).")
        first, second = triplets.validate(len(x), x.device)
        r = aligned(radial, (len(x), self.num_radial), x, "radial")
        s = aligned(spherical, (len(first), self.num_radial * self.num_spherical), x, "spherical")
        return first, second, r, s

    def _finish(self, x: torch.Tensor, interaction: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.outgoing(x)) + interaction
        for block in self.before:
            hidden = hidden + block(hidden)
        hidden = x + F.silu(self.skip(hidden))
        for block in self.after:
            hidden = hidden + block(hidden)
        return hidden

    def forward(
        self,
        x: torch.Tensor,
        radial: torch.Tensor,
        spherical: torch.Tensor,
        triplets: GraphTriplets,
        *,
        message_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Weight each triplet by g_in*g_out and the final edge state by g_out."""
        first, second, r, s = self._validate(x, radial, spherical, triplets)
        send = F.silu(self.incoming(x)) * self.radial(r)
        interactions = torch.einsum("tb,tc,obc->to", self.spherical(s), send[first], self.bilinear)
        weights = GraphEdgeData.normalize_message_weights(
            message_weights, edge_count=len(x), reference=x
        )
        if weights is not None:
            interactions = interactions * (weights[first] * weights[second])[:, None]
        result = self._finish(x, Scatter.sum(interactions, second, len(x)))
        return result if weights is None else result * weights[:, None]
