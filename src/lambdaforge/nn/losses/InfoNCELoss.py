"""In-batch InfoNCE contrastive objective."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from lambdaforge.nn.losses.Loss import Loss
from lambdaforge.nn.losses.Reduction import Reduction


class InfoNCELoss(Loss):
    """Contrast aligned rows of two embedding matrices using in-batch negatives.

    Inputs must both have shape ``(B, D)`` and matching rows are positives.
    Distributed negatives are deliberately not gathered implicitly; every DDP
    rank therefore uses its local batch, avoiding hidden communication.
    """

    def __init__(
        self,
        first_key: str = "embedding_a",
        second_key: str = "embedding_b",
        weight: float = 1.0,
        temperature: float = 0.07,
        learnable_temperature: bool = False,
        min_temperature: float = 1e-4,
        normalize: bool = True,
        symmetric: bool = True,
        reduction: Reduction | str = Reduction.MEAN,
        name: str = "info_nce",
    ) -> None:
        super().__init__(name=name, weight=weight)
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if min_temperature <= 0 or min_temperature >= temperature:
            raise ValueError("min_temperature must be positive and smaller than temperature.")
        raw_value = math.log(math.expm1(temperature - min_temperature))
        raw = torch.tensor(raw_value, dtype=torch.float32)
        if learnable_temperature:
            self.raw_temperature = nn.Parameter(raw)
        else:
            self.register_buffer("raw_temperature", raw)
        self.first_key = first_key
        self.second_key = second_key
        self.min_temperature = float(min_temperature)
        self.normalize = bool(normalize)
        self.symmetric = bool(symmetric)
        self.reduction = Reduction.from_value(reduction)

    @property
    def temperature(self) -> torch.Tensor:
        """Return the positive effective temperature."""
        return F.softplus(self.raw_temperature) + self.min_temperature

    def forward(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        context: object | None = None,
    ) -> torch.Tensor:
        del batch, context
        first = outputs[self.first_key]
        second = outputs[self.second_key]
        if first.ndim != 2 or second.ndim != 2 or first.shape != second.shape:
            raise ValueError("InfoNCELoss inputs must have the same shape (B, D).")
        if first.shape[0] < 1 or first.shape[1] < 1:
            raise ValueError("InfoNCELoss inputs cannot have empty batch or feature dimensions.")
        if self.normalize:
            first = F.normalize(first, dim=-1)
            second = F.normalize(second, dim=-1)
        temperature = self.temperature.to(device=first.device, dtype=first.dtype)
        logits = torch.matmul(first, second.transpose(0, 1)) / temperature
        target = torch.arange(first.shape[0], device=first.device)
        forward_loss = F.cross_entropy(logits, target, reduction=self.reduction.value)
        if not self.symmetric:
            return self.weight * forward_loss
        reverse_loss = F.cross_entropy(
            logits.transpose(0, 1), target, reduction=self.reduction.value
        )
        return self.weight * 0.5 * (forward_loss + reverse_loss)
