"""Mapping-based smooth-L1 regression loss."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from lambdaforge.nn.losses.Loss import Loss
from lambdaforge.nn.losses.Reduction import Reduction


class SmoothL1Loss(Loss):
    """Quadratic-to-linear robust loss with configurable transition ``beta``."""

    def __init__(
        self,
        output_key: str = "prediction",
        target_key: str = "target",
        weight: float = 1.0,
        reduction: Reduction | str = Reduction.MEAN,
        beta: float = 1.0,
        name: str = "smooth_l1",
    ) -> None:
        super().__init__(name=name, weight=weight)
        if beta < 0:
            raise ValueError("beta must be non-negative.")
        self.output_key = output_key
        self.target_key = target_key
        self.reduction = Reduction.from_value(reduction)
        self.beta = float(beta)

    def forward(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        context: object | None = None,
    ) -> torch.Tensor:
        del context
        prediction = outputs[self.output_key]
        target = batch[self.target_key].to(device=prediction.device, dtype=prediction.dtype)
        loss = F.smooth_l1_loss(prediction, target, reduction=self.reduction.value, beta=self.beta)
        return self.weight * loss
