"""Binary precision metric."""

from lambdaforge.metrics.classification.binary.BinaryConfusionMetric import (
    BinaryConfusionMetric,
)


class BinaryPrecision(BinaryConfusionMetric):
    """Binary classification precision (positive predictive value).

    Formula:
        precision = TP / (TP + FP)

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
            name="precision",
            pred_key=pred_key,
            target_key=target_key,
            threshold=threshold,
        )

    def compute(self) -> float:
        denom = self.tp + self.fp

        if denom == 0:
            return 0.0

        return self.tp / denom
