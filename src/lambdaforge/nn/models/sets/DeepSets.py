"""Permutation-invariant Deep Sets model."""

from __future__ import annotations

from typing import Any

import torch

from lambdaforge.nn.activations.Activation import Activation
from lambdaforge.nn.activations.ReLU import ReLU
from lambdaforge.nn.models.MLP import MLP
from lambdaforge.nn.models.Model import Model
from lambdaforge.nn.normalizations.IdentityNorm import IdentityNorm
from lambdaforge.nn.normalizations.Normalization import Normalization
from lambdaforge.nn.pooling.MeanPooling import MeanPooling
from lambdaforge.nn.pooling.Pooling import Pooling


class DeepSets(Model):
    """Encode elements independently, pool them, then decode the set representation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        embedding_dim: int = 128,
        encoder_hidden: int | list[int] | tuple[int, ...] | None = (128,),
        decoder_hidden: int | list[int] | tuple[int, ...] | None = (128,),
        pooling: Pooling | None = None,
        activation: type[Activation] | str = ReLU,
        normalization: type[Normalization] | str = IdentityNorm,
        dropout: float = 0.0,
        residual: bool = False,
        activation_kwargs: dict[str, Any] | None = None,
        normalization_kwargs: dict[str, Any] | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1 or embedding_dim < 1:
            raise ValueError("in_features, out_features and embedding_dim must be positive.")
        self.element_encoder = MLP(
            in_features,
            embedding_dim,
            hidden=list(encoder_hidden) if isinstance(encoder_hidden, tuple) else encoder_hidden,
            activation=activation,
            normalization=normalization,
            dropout=dropout,
            residual=residual,
            activation_kwargs=activation_kwargs,
            normalization_kwargs=normalization_kwargs,
            bias=bias,
        )
        self.pooling = pooling if pooling is not None else MeanPooling()
        self.set_decoder = MLP(
            embedding_dim,
            out_features,
            hidden=list(decoder_hidden) if isinstance(decoder_hidden, tuple) else decoder_hidden,
            activation=activation,
            normalization=normalization,
            dropout=dropout,
            residual=residual,
            activation_kwargs=activation_kwargs,
            normalization_kwargs=normalization_kwargs,
            bias=bias,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Map ``x`` shaped ``(batch, elements, features)`` to one output per set."""
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, elements, features).")
        if mask is not None and mask.shape != x.shape[:2]:
            raise ValueError("mask must have shape (batch, elements).")
        encoded = self.element_encoder(x)
        pooled = self.pooling(encoded, mask)
        return self.set_decoder(pooled)
