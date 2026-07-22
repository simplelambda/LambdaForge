"""Pairwise contrastive embedding loss."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from lambdaforge.nn.losses.Loss import Loss
from lambdaforge.nn.losses.Reduction import Reduction


class ContrastiveLoss(Loss):
    r"""Classic contrastive loss for paired embeddings.

    Targets use ``1`` for a similar pair and ``0`` for a dissimilar pair. The
    per-example objective is ``y*d^2 + (1-y)*max(margin-d, 0)^2``.
    """

    def __init__(
        self,
        first_key: str = "embedding_a",
        second_key: str = "embedding_b",
        target_key: str = "target",
        weight: float = 1.0,
        margin: float = 1.0,
        p: float = 2.0,
        eps: float = 1e-6,
        reduction: Reduction | str = Reduction.MEAN,
        name: str = "contrastive",
    ) -> None:
        super().__init__(name=name, weight=weight)
        if margin < 0:
            raise ValueError("margin must be non-negative.")
        if p <= 0 or eps <= 0:
            raise ValueError("p and eps must be positive.")
        self.first_key = first_key
        self.second_key = second_key
        self.target_key = target_key
        self.margin = float(margin)
        self.p = float(p)
        self.eps = float(eps)
        self.reduction = Reduction.from_value(reduction)

    def forward(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        context: object | None = None,
    ) -> torch.Tensor:
        del context
        first = outputs[self.first_key]
        second = outputs[self.second_key]
        if first.shape != second.shape:
            raise ValueError("ContrastiveLoss embeddings must have identical shapes.")
        if first.ndim != 2:
            raise ValueError("ContrastiveLoss embeddings must be matrices shaped (B, D).")
        target = batch[self.target_key].to(device=first.device, dtype=first.dtype)
        distance = F.pairwise_distance(first, second, p=self.p, eps=self.eps)
        if target.shape != distance.shape:
            try:
                target = torch.broadcast_to(target, distance.shape)
            except RuntimeError as error:
                raise ValueError(
                    "ContrastiveLoss targets must match the embedding leading dimensions."
                ) from error
        loss = target * distance.square() + (1.0 - target) * F.relu(self.margin - distance).square()
        return self.weight * self.reduction.reduce(loss)
