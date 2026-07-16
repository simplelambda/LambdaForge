"""Shared state accumulator for multiclass metrics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import torch

from lambdaforge.metrics.Metric import Metric


class MulticlassMetric(Metric):
    """Base metric that accumulates multiclass scores and targets on the CPU."""

    def __init__(
        self,
        name: str,
        pred_key: str = "logits",
        target_key: str = "y",
        num_classes: int | None = None,
        average: Literal["macro", "weighted", "none"] | None = "macro",
    ) -> None:
        super().__init__(name=name, higher_is_better=True)
        if num_classes is not None and num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        self.pred_key = pred_key
        self.target_key = target_key
        self.num_classes = num_classes
        self.average = average
        self.reset()

    def update(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        context: object | None = None,
    ) -> None:
        """Append one batch of class scores and integer targets."""
        del context
        predictions = outputs[self.pred_key].detach().float().cpu()
        targets = batch[self.target_key].detach().long().view(-1).cpu()
        if predictions.ndim != 2:
            raise ValueError("Multiclass predictions must have shape (N, C).")
        if predictions.shape[0] != targets.shape[0]:
            raise ValueError("Predictions and targets must have the same batch size.")
        if self.num_classes is not None and predictions.shape[1] != self.num_classes:
            raise ValueError("Prediction class dimension does not match num_classes.")
        self._predictions.append(predictions)
        self._targets.append(targets)

    def reset(self) -> None:
        """Remove all accumulated samples."""
        self._predictions: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []

    def values(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return concatenated class scores and targets."""
        if not self._predictions:
            return torch.empty((0, 0)), torch.empty(0, dtype=torch.long)
        return torch.cat(self._predictions), torch.cat(self._targets)

    def classes(self, predictions: torch.Tensor) -> int:
        """Resolve and validate the class count."""
        return self.num_classes or int(predictions.shape[1])

    def distributed_state(self) -> dict[str, torch.Tensor]:
        """Return accumulated scores and targets for DDP merging."""
        predictions, targets = self.values()
        return {"predictions": predictions, "targets": targets}

    def merge_distributed_state(self, state: Mapping[str, Any]) -> None:
        """Append one worker's accumulated scores and targets."""
        targets = state["targets"]
        if targets.numel() == 0:
            return
        self._predictions.append(state["predictions"])
        self._targets.append(targets)

    def compute(self) -> float:
        """Require concrete metrics to implement their reduction."""
        raise NotImplementedError
