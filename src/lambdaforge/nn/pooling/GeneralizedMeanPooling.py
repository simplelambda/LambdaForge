"""Configurable generalized-mean pooling."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from lambdaforge.nn.pooling.Pooling import Pooling


class GeneralizedMeanPooling(Pooling):
    r"""Pool non-negative features with a generalized mean.

    ``p=1`` is mean pooling and larger exponents approach max pooling. Inputs
    are clamped to ``eps`` as in common GeM vision implementations. Set
    ``signed=True`` to use a sign-preserving experimental extension for
    unbounded embeddings. Empty masked sets return zeros.
    """

    def __init__(
        self,
        p: float = 3.0,
        learnable: bool = True,
        per_feature: bool = False,
        in_features: int | None = None,
        min_p: float = 1e-3,
        max_p: float | None = None,
        eps: float = 1e-6,
        signed: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if p <= 0 or min_p <= 0 or min_p >= p:
            raise ValueError("p must be positive and greater than min_p.")
        if max_p is not None and max_p < p:
            raise ValueError("max_p must be greater than or equal to the initial p.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if per_feature and (in_features is None or in_features < 1):
            raise ValueError("in_features must be positive when per_feature=True.")
        parameter_count = int(in_features) if per_feature and in_features is not None else 1
        raw_value = math.log(math.expm1(p - min_p))
        raw = torch.full((parameter_count,), raw_value)
        if learnable:
            self.raw_p = nn.Parameter(raw)
        else:
            self.register_buffer("raw_p", raw)
        self.per_feature = bool(per_feature)
        self.in_features = in_features
        self.min_p = float(min_p)
        self.max_p = None if max_p is None else float(max_p)
        self.eps = float(eps)
        self.signed = bool(signed)

    @property
    def p(self) -> torch.Tensor:
        """Return the positive effective generalized-mean exponent."""
        value = F.softplus(self.raw_p) + self.min_p
        return value if self.max_p is None else value.clamp(max=self.max_p)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] < 1:
            raise ValueError("GeneralizedMeanPooling expects non-empty x shaped (B, N, D).")
        if self.per_feature and x.shape[-1] != self.in_features:
            raise ValueError(f"Expected {self.in_features} features, got {x.shape[-1]}.")
        if mask is not None and tuple(mask.shape) != tuple(x.shape[:2]):
            raise ValueError("mask must have shape (B, N).")
        exponent = self.p.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
        magnitude = x.abs().clamp_min(self.eps) if self.signed else x.clamp_min(self.eps)
        powered = magnitude.pow(exponent)
        if self.signed:
            powered = powered * x.sign()
        if mask is None:
            mean_powered = powered.mean(dim=1)
            has_values = torch.ones((x.shape[0], 1), dtype=torch.bool, device=x.device)
        else:
            valid = mask.bool()
            count = valid.sum(dim=1, keepdim=True)
            mean_powered = (powered * valid.unsqueeze(-1)).sum(dim=1) / count.clamp_min(1)
            has_values = count > 0
        inverse = exponent.squeeze(1).reciprocal()
        if self.signed:
            pooled = mean_powered.sign() * mean_powered.abs().clamp_min(self.eps).pow(inverse)
        else:
            pooled = mean_powered.clamp_min(self.eps).pow(inverse)
        return torch.where(has_values, pooled, torch.zeros_like(pooled))
