"""Mapping-based Huber regression loss."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from lambdaforge.nn.losses.Loss import Loss
from lambdaforge.nn.losses.Reduction import Reduction


class HuberLoss(Loss):
    """Robust Huber loss with a configurable positive ``delta``."""

    def __init__(
        self,
        output_key: str = "prediction",
        target_key: str = "target",
        weight: float = 1.0,
        reduction: Reduction | str = Reduction.MEAN,
        delta: float = 1.0,
        name: str = "huber",
    ) -> None:
        super().__init__(name=name, weight=weight)
        if delta <= 0:
            raise ValueError("delta must be positive.")
        self.output_key = output_key
        self.target_key = target_key
        self.reduction = Reduction.from_value(reduction)
        self.delta = float(delta)

    def forward(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        context: object | None = None,
    ) -> torch.Tensor:
        del context
        prediction = outputs[self.output_key]
        target = batch[self.target_key].to(device=prediction.device, dtype=prediction.dtype)
        loss = F.huber_loss(prediction, target, reduction=self.reduction.value, delta=self.delta)
        return self.weight * loss
