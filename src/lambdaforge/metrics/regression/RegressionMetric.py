"""Shared accumulator for regression metrics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from lambdaforge.metrics.Metric import Metric


class RegressionMetric(Metric):
    """Base metric that accumulates prediction-target pairs on the CPU."""

    def __init__(
        self,
        name: str,
        pred_key: str = "pred",
        target_key: str = "y",
        higher_is_better: bool = False,
    ) -> None:
        super().__init__(name=name, higher_is_better=higher_is_better)
        self.pred_key = pred_key
        self.target_key = target_key
        self.reset()

    def update(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        context: object | None = None,
    ) -> None:
        """Append one batch of flattened predictions and targets."""
        del context
        prediction = outputs[self.pred_key].detach().view(-1).float().cpu()
        target = batch[self.target_key].detach().view(-1).float().cpu()
        if prediction.shape != target.shape:
            raise ValueError("Prediction and target shapes must match.")
        self._predictions.append(prediction)
        self._targets.append(target)

    def reset(self) -> None:
        """Remove all accumulated samples."""
        self._predictions: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []

    def values(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return concatenated predictions and targets."""
        if not self._predictions:
            return torch.empty(0), torch.empty(0)
        return torch.cat(self._predictions), torch.cat(self._targets)

    def distributed_state(self) -> dict[str, torch.Tensor]:
        """Return accumulated pairs for DDP merging."""
        predictions, targets = self.values()
        return {"predictions": predictions, "targets": targets}

    def merge_distributed_state(self, state: Mapping[str, Any]) -> None:
        """Append one worker's accumulated pairs."""
        self._predictions.append(state["predictions"])
        self._targets.append(state["targets"])

    def compute(self) -> float:
        """Require concrete metrics to implement their reduction."""
        raise NotImplementedError
