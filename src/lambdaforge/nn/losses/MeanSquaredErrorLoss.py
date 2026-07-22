"""Mapping-based mean-squared-error loss."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from lambdaforge.nn.losses.Loss import Loss
from lambdaforge.nn.losses.Reduction import Reduction


class MeanSquaredErrorLoss(Loss):
    """Measure squared regression error between configurable mapping keys."""

    def __init__(
        self,
        output_key: str = "prediction",
        target_key: str = "target",
        weight: float = 1.0,
        reduction: Reduction | str = Reduction.MEAN,
        name: str = "mean_squared_error",
    ) -> None:
        super().__init__(name=name, weight=weight)
        self.output_key = output_key
        self.target_key = target_key
        self.reduction = Reduction.from_value(reduction)

    def forward(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        context: object | None = None,
    ) -> torch.Tensor:
        del context
        prediction = outputs[self.output_key]
        target = batch[self.target_key].to(device=prediction.device, dtype=prediction.dtype)
        return self.weight * F.mse_loss(prediction, target, reduction=self.reduction.value)
