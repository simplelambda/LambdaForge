"""Binary AUROC (Area Under the ROC Curve) metric.

Uses ``torchmetrics`` when available and falls back to ``scikit-learn``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from lambdaforge.metrics.Metric import Metric


class BinaryAUROC(Metric):
    """Area Under the Receiver Operating Characteristic curve.

    Formula:
        AUROC = integral of TPR vs FPR as the decision threshold varies.

    An AUROC of 1.0 means perfect separation; 0.5 means random guessing.

    Requires either ``torchmetrics`` or ``scikit-learn`` installed.

    Parameters
    ----------
    pred_key : str
        Key in ``outputs`` for prediction scores, logits or probabilities.
        Default: ``"probs"``.
    target_key : str
        Key in ``batch`` for ground-truth labels (0 or 1). Default: ``"y"``.
    """

    def __init__(
        self,
        pred_key: str = "probs",
        target_key: str = "y",
    ) -> None:
        super().__init__(name="auroc", higher_is_better=True)
        self.pred_key = pred_key
        self.target_key = target_key
        self.reset()

    # ------------------------------------------------------------------
    def update(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        context: object | None = None,
    ) -> None:
        probs = outputs[self.pred_key].detach().view(-1).float().cpu()
        target = batch[self.target_key].detach().view(-1).long().cpu()
        self._preds.append(probs)
        self._targets.append(target)

    def reset(self) -> None:
        self._preds: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []

    def _cat(self):
        if not self._preds:
            return torch.empty(0), torch.empty(0)
        return torch.cat(self._preds), torch.cat(self._targets)

    def distributed_state(self) -> dict[str, torch.Tensor]:
        """Return accumulated predictions and targets for DDP merging."""
        predictions, targets = self._cat()
        return {"predictions": predictions, "targets": targets}

    def merge_distributed_state(self, state: Mapping[str, Any]) -> None:
        """Append one worker's predictions and targets."""
        self._preds.append(state["predictions"])
        self._targets.append(state["targets"])

    # ------------------------------------------------------------------
    def compute(self) -> float:
        preds, targets = self._cat()
        if len(preds) == 0:
            return float("nan")

        unique = targets.unique()
        if len(unique) < 2:
            return float("nan")

        try:
            from torchmetrics import AUROC as TM_AUROC

            metric = TM_AUROC(task="binary")
            return float(metric(preds, targets).item())
        except ImportError:
            pass

        try:
            from sklearn.metrics import roc_auc_score

            return float(roc_auc_score(targets.numpy(), preds.numpy()))
        except ImportError as error:
            raise ImportError(
                "BinaryAUROC requires either torchmetrics or scikit-learn. "
                "Install with: pip install torchmetrics  or  pip install scikit-learn"
            ) from error
