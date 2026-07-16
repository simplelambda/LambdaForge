"""Binary specificity metric."""

from lambdaforge.metrics.classification.binary.BinaryConfusionMetric import (
    BinaryConfusionMetric,
)


class BinarySpecificity(BinaryConfusionMetric):
    """Binary classification specificity (true negative rate).

    Formula:
        specificity = TN / (TN + FP)

    Measures the proportion of actual negatives correctly identified.

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
            name="specificity",
            pred_key=pred_key,
            target_key=target_key,
            threshold=threshold,
        )

    def compute(self) -> float:
        denom = self.tn + self.fp
        return self.tn / denom if denom > 0 else 0.0
