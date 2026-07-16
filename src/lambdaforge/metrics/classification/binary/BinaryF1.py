"""Binary F1-score metric."""

from lambdaforge.metrics.classification.binary.BinaryConfusionMetric import (
    BinaryConfusionMetric,
)


class BinaryF1(BinaryConfusionMetric):
    """Binary classification F1 score (harmonic mean of precision and recall).

    Formula:
        F1 = 2 * TP / (2 * TP + FP + FN)

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
            name="f1",
            pred_key=pred_key,
            target_key=target_key,
            threshold=threshold,
        )

    def compute(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn

        if denom == 0:
            return 0.0

        return 2 * self.tp / denom
