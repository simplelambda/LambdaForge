"""Binary Matthews Correlation Coefficient metric."""

import math

from lambdaforge.metrics.classification.binary.BinaryConfusionMetric import (
    BinaryConfusionMetric,
)


class BinaryMCC(BinaryConfusionMetric):
    """Matthews Correlation Coefficient for binary classification.

    Formula:
        MCC = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

    Returns a value between -1 (total disagreement) and +1 (perfect
    prediction).  Unlike accuracy or F1, MCC is balanced even when
    classes have very different sizes.

    Parameters
    ----------
    pred_key : str
        Key in ``outputs`` for predicted probabilities. Default: ``"probs"``.
    target_key : str
        Key in ``batch`` for ground-truth labels. Default: ``"y"``.
    threshold : float
        Decision threshold. Default: ``0.5``.
    """

    def __init__(
        self,
        pred_key: str = "probs",
        target_key: str = "y",
        threshold: float = 0.5,
    ) -> None:
        super().__init__(
            name="mcc",
            pred_key=pred_key,
            target_key=target_key,
            threshold=threshold,
        )

    def compute(self) -> float:
        num = self.tp * self.tn - self.fp * self.fn
        denom = math.sqrt(
            (self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn)
        )
        return num / denom if denom > 0 else 0.0
