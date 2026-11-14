"""Generic GVP encoder with a final invariant node readout."""

from __future__ import annotations

import torch
from torch import nn

from lambdaforge.nn.models.graph.equivariant.GVP import GVP
from lambdaforge.nn.models.graph.equivariant.GVPGNNLayer import GVPGNNLayer
from lambdaforge.nn.models.graph.validation import channels
from lambdaforge.nn.models.Model import Model


class GVPGNN(Model):
    """Keep polar vectors between blocks; expose scalars for downstream generic encoders."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scalar_channels: int = 64,
        vector_channels: int = 16,
        num_layers: int = 3,
        edge_channels: int = 0,
        edge_vector_channels: int = 1,
        aggregation: str = "mean",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        channels(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            scalar_channels=scalar_channels,
            vector_channels=vector_channels,
        )
        self.vector_channels = vector_channels
        self.input = nn.Linear(in_channels, scalar_channels)
        self.layers = nn.ModuleList(
            GVPGNNLayer(
                scalar_channels,
                vector_channels,
                edge_channels,
                edge_vector_channels,
                aggregation,
                dropout,
            )
            for _ in range(num_layers)
        )
        self.readout = GVP(scalar_channels, vector_channels, out_channels, 0, activate=False)

    def forward_with_vectors(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return final latent states; vectors initialize to exact zero."""
        scalars = self.input(x)
        vectors = scalars.new_zeros((len(x), self.vector_channels, 3))
        for layer in self.layers:
            scalars, vectors = layer(scalars, vectors, edge_index, edge_vectors, edge_features)
        return scalars, vectors

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return (N,out_channels) invariant features; no graph pooling is implied."""
        scalars, vectors = self.forward_with_vectors(x, edge_index, edge_vectors, edge_features)
        return self.readout(scalars, vectors)[0]
