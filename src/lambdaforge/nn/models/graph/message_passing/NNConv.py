"""Node encoder composed from full or basis-decomposed edge-conditioned filters."""

from __future__ import annotations

import torch
from torch import nn

from lambdaforge.nn.models.graph.message_passing.NNConvLayer import NNConvLayer
from lambdaforge.nn.models.graph.validation import channels
from lambdaforge.nn.models.Model import Model


class NNConv(Model):
    """Stack NNConv layers with SiLU only between layers, preserving node order."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_channels: int,
        hidden_channels: tuple[int, ...] = (64, 64),
        conditioner_channels: int = 64,
        num_bases: int | None = None,
        aggregation: str = "mean",
        root_weight: bool = True,
        center_conditioner: bool = False,
        message_chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        widths = (in_channels, *hidden_channels, out_channels)
        for width in widths:
            channels(width=width)
        self.layers = nn.ModuleList(
            NNConvLayer(
                left,
                right,
                edge_channels,
                conditioner_channels,
                num_bases,
                aggregation,
                root_weight,
                center_conditioner,
                message_chunk_size,
            )
            for left, right in zip(widths[:-1], widths[1:], strict=True)
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor
    ) -> torch.Tensor:
        """Return (N,out_channels); edge metadata stay aligned at every layer."""
        for index, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_features)
            if index < len(self.layers) - 1:
                x = torch.nn.functional.silu(x)
        return x
