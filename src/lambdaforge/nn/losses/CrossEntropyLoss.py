"""Mapping-based multiclass cross-entropy loss."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from lambdaforge.nn.losses.Loss import Loss
from lambdaforge.nn.losses.Reduction import Reduction


class CrossEntropyLoss(Loss):
    """Cross entropy over raw class logits.

    Parameters mirror :func:`torch.nn.functional.cross_entropy`, while
    ``output_key`` and ``target_key`` connect the objective to arbitrary task
    mappings. Only scalar reductions are accepted by LambdaForge training.
    """

    def __init__(
        self,
        output_key: str = "logits",
        target_key: str = "target",
        weight: float = 1.0,
        reduction: Reduction | str = Reduction.MEAN,
        class_weight: torch.Tensor | Sequence[float] | None = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        name: str = "cross_entropy",
    ) -> None:
        super().__init__(name=name, weight=weight)
        if not 0.0 <= label_smoothing <= 1.0:
            raise ValueError("label_smoothing must be in [0, 1].")
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
        self.ignore_index = int(ignore_index)
        self.label_smoothing = float(label_smoothing)

    def forward(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        context: object | None = None,
    ) -> torch.Tensor:
        del context
        logits = outputs[self.output_key]
        target = batch[self.target_key].to(device=logits.device, dtype=torch.long)
        loss = F.cross_entropy(
            logits,
            target,
            weight=(
                None if self.class_weight is None else self.class_weight.to(dtype=logits.dtype)
            ),
            ignore_index=self.ignore_index,
            reduction=self.reduction.value,
            label_smoothing=self.label_smoothing,
        )
        return self.weight * loss
