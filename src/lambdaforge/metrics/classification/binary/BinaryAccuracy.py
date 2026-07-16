"""Binary accuracy metric."""

from lambdaforge.metrics.classification.binary.BinaryConfusionMetric import (
    BinaryConfusionMetric,
)


class BinaryAccuracy(BinaryConfusionMetric):
    """Binary classification accuracy.

    Formula:
        accuracy = (TP + TN) / (TP + TN + FP + FN)

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
            name="accuracy",
            pred_key=pred_key,
            target_key=target_key,
            threshold=threshold,
        )

    def compute(self) -> float:
        total = self.tp + self.tn + self.fp + self.fn

        if total == 0:
            return 0.0

        return (self.tp + self.tn) / total
