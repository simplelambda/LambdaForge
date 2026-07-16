"""Base class for binary metrics that accumulate TP/TN/FP/FN counts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lambdaforge.metrics.Metric import Metric


class BinaryConfusionMetric(Metric):
    """Base for binary metrics built on top of a confusion matrix.

    Accumulates true positives, true negatives, false positives and false
    negatives across batches.  Subclasses only need to implement ``compute``.

    Parameters
    ----------
    name : str
        Metric name used in logs and reports.
    pred_key : str
        Key in ``outputs`` that holds predicted probabilities (or logits).
        Default: ``"probs"``.
    target_key : str
        Key in ``batch`` that holds ground-truth labels (0 or 1).
        Default: ``"y"``.
    threshold : float
        Decision threshold applied to probabilities.  Predictions >= threshold
        are considered positive.  Default: ``0.5``.
    higher_is_better : bool
        Whether larger values are better.  Default: ``True``.
    """

    def __init__(
        self,
        name: str,
        pred_key: str = "probs",
        target_key: str = "y",
        threshold: float = 0.5,
        higher_is_better: bool = True,
    ) -> None:
        super().__init__(name=name, higher_is_better=higher_is_better)
        self.pred_key = pred_key
        self.target_key = target_key
        self.threshold = threshold
        self.reset()

    # ------------------------------------------------------------------
    def update(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        context: object | None = None,
    ) -> None:
        probs = outputs[self.pred_key]
        target = batch[self.target_key]

        probs = probs.detach().view(-1).float().cpu()
        target = target.detach().view(-1).long().cpu()

        pred = (probs >= self.threshold).long()

        self.tp += int(((pred == 1) & (target == 1)).sum().item())
        self.tn += int(((pred == 0) & (target == 0)).sum().item())
        self.fp += int(((pred == 1) & (target == 0)).sum().item())
        self.fn += int(((pred == 0) & (target == 1)).sum().item())

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    # ------------------------------------------------------------------
    def confusion(self) -> dict[str, int]:
        """Return the accumulated confusion matrix counts."""
        return {"tp": self.tp, "tn": self.tn, "fp": self.fp, "fn": self.fn}

    def distributed_state(self) -> dict[str, int]:
        """Return confusion counts for DDP merging."""
        return self.confusion()

    def merge_distributed_state(self, state: Mapping[str, Any]) -> None:
        """Add one worker's confusion counts."""
        self.tp += int(state["tp"])
        self.tn += int(state["tn"])
        self.fp += int(state["fp"])
        self.fn += int(state["fn"])
