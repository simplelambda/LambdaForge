"""Generic continuous-filter node encoder, independent of atom identities or energies."""

from __future__ import annotations

import torch
from torch import nn

from lambdaforge.nn.models.graph.geometric.SchNetLayer import SchNetLayer
from lambdaforge.nn.models.graph.validation import channels
from lambdaforge.nn.models.Model import Model


class SchNet(Model):
    """Project arbitrary floating node descriptors, interact, return one embedding per node."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        num_radial: int = 32,
        radius: float = 5.0,
        cutoff: float | None = None,
        edge_channels: int = 0,
    ) -> None:
        super().__init__()
        channels(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
        )
        self.input = nn.Linear(in_channels, hidden_channels)
        self.layers = nn.ModuleList(
            SchNetLayer(hidden_channels, num_radial, radius, cutoff, edge_channels)
            for _ in range(num_layers)
        )
        self.output = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        distances: torch.Tensor,
        edge_features: torch.Tensor | None = None,
        *,
        distance_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return (N, out_channels); never create radius neighbours or mutate positions."""
        hidden = self.input(x)
        for layer in self.layers:
            hidden = layer(
                hidden, edge_index, distances, edge_features, distance_weights=distance_weights
            )
        return self.output(hidden)
