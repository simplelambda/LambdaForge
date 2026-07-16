"""Binary Cohen's kappa metric."""

from lambdaforge.metrics.classification.binary.BinaryConfusionMetric import (
    BinaryConfusionMetric,
)


class BinaryCohenKappa(BinaryConfusionMetric):
    """Cohen's kappa for binary classification.

    Kappa measures agreement between predictions and targets corrected by the
    agreement expected by chance. It is useful alongside accuracy when class
    imbalance makes always predicting the majority class look deceptively good.
    """

    def __init__(
        self,
        pred_key: str = "probs",
        target_key: str = "y",
        threshold: float = 0.5,
    ) -> None:
        super().__init__(
            name="kappa",
            pred_key=pred_key,
            target_key=target_key,
            threshold=threshold,
        )

    def compute(self) -> float:
        total = self.tp + self.tn + self.fp + self.fn
        if total == 0:
            return 0.0

        observed = (self.tp + self.tn) / total
        pred_pos = self.tp + self.fp
        pred_neg = self.tn + self.fn
        true_pos = self.tp + self.fn
        true_neg = self.tn + self.fp
        expected = ((pred_pos * true_pos) + (pred_neg * true_neg)) / (total * total)

        denom = 1.0 - expected
        if denom <= 0:
            return 0.0

        return (observed - expected) / denom
