"""Node-wise PaiNN representation for arbitrary scalar inputs on a supplied graph."""

from __future__ import annotations

import torch
from torch import nn

from lambdaforge.nn.models.graph.equivariant.PaiNNLayer import PaiNNLayer
from lambdaforge.nn.models.graph.validation import channels
from lambdaforge.nn.models.Model import Model


class PaiNN(Model):
    """Stack radial/vector interactions; omit molecular embedding/energy/domain policy."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        num_radial: int = 20,
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
        self.hidden_channels = hidden_channels
        self.input = nn.Linear(in_channels, hidden_channels)
        self.layers = nn.ModuleList(
            PaiNNLayer(hidden_channels, num_radial, radius, cutoff, edge_channels)
            for _ in range(num_layers)
        )
        self.output = nn.Linear(hidden_channels, out_channels)

    def forward_with_vectors(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        distances: torch.Tensor,
        directions: torch.Tensor,
        edge_features: torch.Tensor | None = None,
        *,
        distance_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return final latent scalar and polar-vector states."""
        scalars = self.input(x)
        vectors = scalars.new_zeros((len(x), self.hidden_channels, 3))
        for layer in self.layers:
            scalars, vectors = layer(
                scalars,
                vectors,
                edge_index,
                distances,
                directions,
                edge_features,
                distance_weights=distance_weights,
            )
        return scalars, vectors

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        distances: torch.Tensor,
        directions: torch.Tensor,
        edge_features: torch.Tensor | None = None,
        *,
        distance_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return invariant (N,out_channels) node embeddings, with no graph-level pooling."""
        return self.output(
            self.forward_with_vectors(
                x,
                edge_index,
                distances,
                directions,
                edge_features,
                distance_weights=distance_weights,
            )[0]
        )
