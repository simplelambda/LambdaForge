"""Focal loss for binary or multilabel logits."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from lambdaforge.nn.losses.Loss import Loss
from lambdaforge.nn.losses.Reduction import Reduction


class BinaryFocalLoss(Loss):
    r"""Binary focal loss over logits.

    The unreduced term is ``alpha_t * (1 - p_t) ** gamma * BCE``. Set
    ``alpha=None`` to disable class balancing. Targets may be binary or soft
    values in ``[0, 1]``.
    """

    def __init__(
        self,
        output_key: str = "logits",
        target_key: str = "target",
        weight: float = 1.0,
        reduction: Reduction | str = Reduction.MEAN,
        alpha: float | None = 0.25,
        gamma: float = 2.0,
        name: str = "binary_focal",
    ) -> None:
        super().__init__(name=name, weight=weight)
        if alpha is not None and not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1] or None.")
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        self.output_key = output_key
        self.target_key = target_key
        self.reduction = Reduction.from_value(reduction)
        self.alpha = None if alpha is None else float(alpha)
        self.gamma = float(gamma)

    def forward(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        context: object | None = None,
    ) -> torch.Tensor:
        del context
        logits = outputs[self.output_key]
        target = batch[self.target_key].to(device=logits.device, dtype=logits.dtype)
        if logits.shape != target.shape:
            raise ValueError("BinaryFocalLoss logits and targets must have identical shapes.")
        cross_entropy = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        probability = torch.sigmoid(logits)
        probability_true = probability * target + (1.0 - probability) * (1.0 - target)
        focal_weight = (1.0 - probability_true).pow(self.gamma)
        if self.alpha is not None:
            alpha_true = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
            focal_weight = focal_weight * alpha_true
        loss = focal_weight * cross_entropy
        return self.weight * self.reduction.reduce(loss)
