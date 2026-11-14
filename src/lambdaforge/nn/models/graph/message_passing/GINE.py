"""Node-wise stack of explicit edge-aware GIN operators."""

from __future__ import annotations

import torch
from torch import nn

from lambdaforge.nn.models.graph.message_passing.GINELayer import GINELayer
from lambdaforge.nn.models.graph.validation import channels
from lambdaforge.nn.models.Model import Model


class GINE(Model):
    """Compose GINE layers with ReLU between hidden layers and linear final MLP output."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_channels: int,
        hidden_channels: tuple[int, ...] = (64, 64),
        epsilon: float = 0.0,
        trainable_epsilon: bool = False,
    ) -> None:
        super().__init__()
        widths = (in_channels, *hidden_channels, out_channels)
        for width in widths:
            channels(width=width)
        self.layers = nn.ModuleList(
            GINELayer(
                left, right, edge_channels, epsilon=epsilon, trainable_epsilon=trainable_epsilon
            )
            for left, right in zip(widths[:-1], widths[1:], strict=True)
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return (N,out_channels); no neighbour inference or graph pooling."""
        for index, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_features)
            if index < len(self.layers) - 1:
                x = x.relu()
        return x
