"""Permutation-invariant Set Transformer model."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from lambdaforge.nn.activations.base import Activation
from lambdaforge.nn.ComponentRegistry import ComponentRegistry
from lambdaforge.nn.models.Model import Model


class SetTransformer(Model):
    """Self-attention set encoder followed by learned seed attention pooling."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        num_seeds: int = 1,
        feedforward_dim: int | None = None,
        dropout: float = 0.1,
        activation: type[Activation] | str = "gelu",
        activation_kwargs: dict[str, Any] | None = None,
        norm_first: bool = True,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        squeeze_single_seed: bool = True,
        enable_nested_tensor: bool = False,
    ) -> None:
        super().__init__()
        if min(in_features, out_features, d_model, num_heads, num_layers, num_seeds) < 1:
            raise ValueError("Feature sizes, heads, layers and seeds must be positive.")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if feedforward_dim is not None and feedforward_dim < 1:
            raise ValueError("feedforward_dim must be positive when provided.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if layer_norm_eps <= 0:
            raise ValueError("layer_norm_eps must be positive.")
        activation_cls = ComponentRegistry.resolve_activation(activation)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            num_heads,
            dim_feedforward=feedforward_dim or 4 * d_model,
            dropout=dropout,
            activation=activation_cls(**(activation_kwargs or {})),
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=norm_first,
            bias=bias,
        )
        self.input = nn.Linear(in_features, d_model, bias=bias)
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers,
            norm=nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias),
            enable_nested_tensor=enable_nested_tensor,
        )
        self.seeds = nn.Parameter(torch.empty(1, num_seeds, d_model))
        self.pooling_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, bias=bias, batch_first=True
        )
        self.output_normalization = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.output = nn.Linear(d_model, out_features, bias=bias)
        self.num_seeds = num_seeds
        self.squeeze_single_seed = bool(squeeze_single_seed)
        nn.init.normal_(self.seeds, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode a set; ``True`` entries in ``mask`` denote valid elements."""
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, elements, features).")
        padding_mask = None
        if mask is not None:
            if mask.shape != x.shape[:2]:
                raise ValueError("mask must have shape (batch, elements).")
            valid = mask.to(dtype=torch.bool, device=x.device)
            if bool((valid.sum(dim=1) == 0).any()):
                raise ValueError("Every set must contain at least one valid element.")
            padding_mask = ~valid
        encoded = self.encoder(self.input(x), src_key_padding_mask=padding_mask)
        queries = self.seeds.expand(x.shape[0], -1, -1)
        pooled, _ = self.pooling_attention(
            queries,
            encoded,
            encoded,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        output = self.output(self.output_normalization(pooled))
        if self.num_seeds == 1 and self.squeeze_single_seed:
            return output[:, 0]
        return output
