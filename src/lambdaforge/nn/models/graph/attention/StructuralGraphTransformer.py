"""Generic node-wise global structural Transformer stack."""

from __future__ import annotations

import torch
from torch import nn

from lambdaforge.nn.models.graph.attention.StructuralGraphTransformerLayer import (
    StructuralGraphTransformerLayer,
)
from lambdaforge.nn.models.graph.validation import channels
from lambdaforge.nn.models.Model import Model


class StructuralGraphTransformer(Model):
    """Compose structural blocks without flattening graphs into a shared attention domain."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        max_degree: int = 128,
        num_spatial_codes: int = 32,
        dropout: float = 0.0,
        max_attention_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        super().__init__()
        channels(in_channels=in_channels, out_channels=out_channels, num_layers=num_layers)
        self.input = nn.Linear(in_channels, hidden_channels)
        self.layers = nn.ModuleList(
            StructuralGraphTransformerLayer(
                hidden_channels,
                num_heads,
                max_degree,
                num_spatial_codes,
                dropout,
                max_attention_bytes,
            )
            for _ in range(num_layers)
        )
        self.output = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        spatial_indices: torch.Tensor,
        in_degree: torch.Tensor,
        out_degree: torch.Tensor,
        valid_nodes: torch.Tensor | None = None,
        edge_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return padded (B,N,out_channels) embeddings with padding exactly zero."""
        hidden = self.input(x)
        for layer in self.layers:
            hidden = layer(hidden, spatial_indices, in_degree, out_degree, valid_nodes, edge_bias)
        result = self.output(hidden)
        return (
            result
            if valid_nodes is None
            else result.masked_fill(~valid_nodes.to(x.device)[..., None], 0)
        )
