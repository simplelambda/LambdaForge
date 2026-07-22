"""Tversky overlap loss for binary, multiclass and multilabel outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from lambdaforge.nn.losses.Loss import Loss
from lambdaforge.nn.losses.Reduction import Reduction


class TverskyLoss(Loss):
    r"""Compute ``1 - Tversky`` with configurable false-positive/negative costs."""

    def __init__(
        self,
        output_key: str = "logits",
        target_key: str = "target",
        weight: float = 1.0,
        mode: str = "binary",
        from_logits: bool = True,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 0.0,
        eps: float = 1e-7,
        include_background: bool = True,
        class_weight: torch.Tensor | Sequence[float] | None = None,
        reduction: Reduction | str = Reduction.MEAN,
        name: str = "tversky",
    ) -> None:
        super().__init__(name=name, weight=weight)
        if mode not in {"binary", "multiclass", "multilabel"}:
            raise ValueError("mode must be 'binary', 'multiclass' or 'multilabel'.")
        if alpha < 0 or beta < 0 or alpha + beta <= 0:
            raise ValueError("alpha and beta must be non-negative and not both zero.")
        if smooth < 0 or eps <= 0:
            raise ValueError("smooth must be non-negative and eps must be positive.")
        resolved_weight = (
            None
            if class_weight is None
            else torch.as_tensor(class_weight, dtype=torch.float32).detach().clone()
        )
        if resolved_weight is not None and (
            resolved_weight.ndim != 1 or resolved_weight.numel() < 1
        ):
            raise ValueError("class_weight must be a non-empty one-dimensional tensor or sequence.")
        self.class_weight: torch.Tensor | None
        self.register_buffer("class_weight", resolved_weight)
        self.output_key = output_key
        self.target_key = target_key
        self.mode = mode
        self.from_logits = bool(from_logits)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)
        self.eps = float(eps)
        self.include_background = bool(include_background)
        self.reduction = Reduction.from_value(reduction)

    def forward(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        context: object | None = None,
    ) -> torch.Tensor:
        del context
        prediction = outputs[self.output_key]
        target = batch[self.target_key].to(device=prediction.device)
        if prediction.ndim < 2:
            raise ValueError("TverskyLoss predictions must include a batch dimension.")
        if self.mode == "multiclass":
            class_count = prediction.shape[1]
            expected_target_shape = (prediction.shape[0], *prediction.shape[2:])
            if tuple(target.shape) != expected_target_shape:
                raise ValueError(
                    f"Expected multiclass target shape {expected_target_shape}, "
                    f"got {tuple(target.shape)}."
                )
            probabilities = torch.softmax(prediction, dim=1) if self.from_logits else prediction
            one_hot = F.one_hot(target.to(torch.long), num_classes=class_count)
            target_values = one_hot.movedim(-1, 1).to(dtype=prediction.dtype)
        else:
            probabilities = torch.sigmoid(prediction) if self.from_logits else prediction
            target_values = target.to(dtype=prediction.dtype)
            if probabilities.shape != target_values.shape:
                raise ValueError("TverskyLoss predictions and targets must have identical shapes.")
        if self.mode == "binary":
            probabilities = probabilities.reshape(probabilities.shape[0], 1, -1)
            target_values = target_values.reshape(target_values.shape[0], 1, -1)
        else:
            probabilities = probabilities.reshape(
                probabilities.shape[0], probabilities.shape[1], -1
            )
            target_values = target_values.reshape(
                target_values.shape[0], target_values.shape[1], -1
            )
        if not self.include_background and probabilities.shape[1] > 1:
            probabilities = probabilities[:, 1:]
            target_values = target_values[:, 1:]
        true_positive = (probabilities * target_values).sum(dim=-1)
        false_positive = (probabilities * (1.0 - target_values)).sum(dim=-1)
        false_negative = ((1.0 - probabilities) * target_values).sum(dim=-1)
        score = (true_positive + self.smooth) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.smooth
            + self.eps
        )
        loss = 1.0 - score
        if self.class_weight is not None:
            if self.class_weight.numel() != loss.shape[1]:
                raise ValueError(
                    f"class_weight has {self.class_weight.numel()} values for "
                    f"{loss.shape[1]} channels."
                )
            loss = loss * self.class_weight.to(dtype=loss.dtype).view(1, -1)
        return self.weight * self.reduction.reduce(loss)
