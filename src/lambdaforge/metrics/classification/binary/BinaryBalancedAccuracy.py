"""Binary balanced accuracy metric."""

from lambdaforge.metrics.classification.binary.BinaryConfusionMetric import (
    BinaryConfusionMetric,
)


class BinaryBalancedAccuracy(BinaryConfusionMetric):
    """Balanced accuracy — average of sensitivity and specificity.

    Formula:
        balanced_accuracy = (TPR + TNR) / 2
        TPR = TP / (TP + FN)
        TNR = TN / (TN + FP)

    Useful when classes are imbalanced, as it weights each class equally
    regardless of its support.

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
            name="balanced_accuracy",
            pred_key=pred_key,
            target_key=target_key,
            threshold=threshold,
        )

    def compute(self) -> float:
        tpr_denom = self.tp + self.fn
        tnr_denom = self.tn + self.fp
        tpr = self.tp / tpr_denom if tpr_denom > 0 else 0.0
        tnr = self.tn / tnr_denom if tnr_denom > 0 else 0.0
        return (tpr + tnr) / 2.0
