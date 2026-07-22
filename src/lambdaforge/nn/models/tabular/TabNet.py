"""Dependency-light attentive tabular network inspired by TabNet."""

from __future__ import annotations

import math

import torch
from torch import nn

from lambdaforge.nn.models.Model import Model


class TabNet(Model):
    """Sequentially attend to features and aggregate decision representations."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 64,
        num_steps: int = 3,
        relaxation: float = 1.5,
        mask_temperature: float = 1.0,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if min(in_features, out_features, hidden_features, num_steps) < 1:
            raise ValueError("Feature sizes and num_steps must be positive.")
        if not math.isfinite(relaxation) or relaxation < 1.0:
            raise ValueError("relaxation must be finite and at least 1.")
        if not math.isfinite(mask_temperature) or mask_temperature <= 0:
            raise ValueError("mask_temperature must be finite and positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.in_features = in_features
        self.num_steps = num_steps
        self.relaxation = float(relaxation)
        self.mask_temperature = float(mask_temperature)
        self.input_normalization = nn.LayerNorm(in_features)
        self.initial_attention = nn.Linear(in_features, in_features, bias=bias)
        self.feature_transformers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_features, 2 * hidden_features, bias=bias),
                    nn.GLU(dim=-1),
                    nn.LayerNorm(hidden_features),
                    nn.Linear(hidden_features, 2 * hidden_features, bias=bias),
                    nn.GLU(dim=-1),
                    nn.Dropout(dropout),
                )
                for _ in range(num_steps)
            ]
        )
        self.attention_layers = nn.ModuleList(
            [nn.Linear(hidden_features, in_features, bias=bias) for _ in range(num_steps - 1)]
        )
        self.head = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions for a floating tensor shaped ``(batch, features)``."""
        features, _ = self.forward_with_masks(x)
        return self.head(features)

    def forward_with_masks(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Return decision features and each normalized attentive feature mask."""
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError("x must have shape (batch, in_features).")
        if not torch.is_floating_point(x):
            raise TypeError("x must use a floating-point dtype.")
        normalized = self.input_normalization(x)
        prior = torch.ones_like(normalized)
        attention = self.initial_attention(normalized)
        decision: torch.Tensor | None = None
        masks: list[torch.Tensor] = []
        for index, transformer in enumerate(self.feature_transformers):
            mask = torch.softmax(
                attention / self.mask_temperature + torch.log(prior.clamp_min(1e-12)),
                dim=-1,
            )
            transformed = transformer(normalized * mask)
            decision = transformed.relu() if decision is None else decision + transformed.relu()
            masks.append(mask)
            prior = prior * (self.relaxation - mask).clamp_min(0.0)
            if index < len(self.attention_layers):
                attention = self.attention_layers[index](transformed)
        if decision is None:
            raise RuntimeError("TabNet has no decision step.")
        return decision, tuple(masks)
