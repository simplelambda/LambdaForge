"""Mapping-based triplet margin loss."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from lambdaforge.nn.losses.Loss import Loss
from lambdaforge.nn.losses.Reduction import Reduction


class TripletMarginLoss(Loss):
    """Keep anchors closer to positives than negatives by a margin."""

    def __init__(
        self,
        anchor_key: str = "anchor",
        positive_key: str = "positive",
        negative_key: str = "negative",
        weight: float = 1.0,
        margin: float = 1.0,
        p: float = 2.0,
        eps: float = 1e-6,
        swap: bool = False,
        reduction: Reduction | str = Reduction.MEAN,
        name: str = "triplet_margin",
    ) -> None:
        super().__init__(name=name, weight=weight)
        if margin <= 0:
            raise ValueError("margin must be positive.")
        if p <= 0 or eps <= 0:
            raise ValueError("p and eps must be positive.")
        self.anchor_key = anchor_key
        self.positive_key = positive_key
        self.negative_key = negative_key
        self.margin = float(margin)
        self.p = float(p)
        self.eps = float(eps)
        self.swap = bool(swap)
        self.reduction = Reduction.from_value(reduction)

    def forward(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        context: object | None = None,
    ) -> torch.Tensor:
        del batch, context
        anchor = outputs[self.anchor_key]
        positive = outputs[self.positive_key]
        negative = outputs[self.negative_key]
        if anchor.shape != positive.shape or anchor.shape != negative.shape:
            raise ValueError("TripletMarginLoss embeddings must have identical shapes.")
        if anchor.ndim != 2:
            raise ValueError("TripletMarginLoss embeddings must be matrices shaped (B, D).")
        loss = F.triplet_margin_loss(
            anchor,
            positive,
            negative,
            margin=self.margin,
            p=self.p,
            eps=self.eps,
            swap=self.swap,
            reduction=self.reduction.value,
        )
        return self.weight * loss
