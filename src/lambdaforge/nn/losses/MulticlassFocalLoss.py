"""Focal loss for mutually exclusive classes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from lambdaforge.nn.losses.Loss import Loss
from lambdaforge.nn.losses.Reduction import Reduction


class MulticlassFocalLoss(Loss):
    """Focal cross entropy for logits shaped ``(B, C, ...)``.

    ``class_weight`` supplies one alpha value per class and accepts a YAML
    sequence or tensor. Targets are integer class indices shaped ``(B, ...)``.
    """

    def __init__(
        self,
        output_key: str = "logits",
        target_key: str = "target",
        weight: float = 1.0,
        reduction: Reduction | str = Reduction.MEAN,
        gamma: float = 2.0,
        class_weight: torch.Tensor | Sequence[float] | None = None,
        ignore_index: int = -100,
        name: str = "multiclass_focal",
    ) -> None:
        super().__init__(name=name, weight=weight)
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
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
        self.reduction = Reduction.from_value(reduction)
        self.gamma = float(gamma)
        self.ignore_index = int(ignore_index)

    def forward(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        context: object | None = None,
    ) -> torch.Tensor:
        del context
        logits = outputs[self.output_key]
        target = batch[self.target_key].to(device=logits.device, dtype=torch.long)
        if logits.ndim < 2:
            raise ValueError("MulticlassFocalLoss logits must have shape (B, C, ...).")
        expected_target_shape = (logits.shape[0], *logits.shape[2:])
        if tuple(target.shape) != expected_target_shape:
            raise ValueError(
                f"Expected target shape {expected_target_shape}, got {tuple(target.shape)}."
            )

        class_count = logits.shape[1]
        flat_logits = logits.movedim(1, -1).reshape(-1, class_count)
        flat_target = target.reshape(-1)
        valid = flat_target != self.ignore_index
        if not torch.any(valid):
            return logits.sum() * 0.0
        valid_logits = flat_logits[valid]
        valid_target = flat_target[valid]
        log_probability = F.log_softmax(valid_logits, dim=-1)
        selected_log_probability = log_probability.gather(1, valid_target.unsqueeze(1)).squeeze(1)
        selected_probability = selected_log_probability.exp()
        loss = -(1.0 - selected_probability).pow(self.gamma) * selected_log_probability
        if self.class_weight is not None:
            if self.class_weight.numel() != class_count:
                raise ValueError(
                    f"class_weight has {self.class_weight.numel()} values for "
                    f"{class_count} classes."
                )
            weights = self.class_weight.to(dtype=logits.dtype)[valid_target]
            loss = loss * weights
        return self.weight * self.reduction.reduce(loss)
