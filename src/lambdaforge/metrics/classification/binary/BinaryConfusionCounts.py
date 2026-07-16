"""Binary confusion counts metric."""

from lambdaforge.metrics.classification.binary.BinaryConfusionMetric import (
    BinaryConfusionMetric,
)


class BinaryConfusionCounts(BinaryConfusionMetric):
    """Accumulates TP, TN, FP, FN counts for inspection.

    ``compute()`` returns accuracy as a convenience; use ``counts()`` to
    retrieve the raw confusion matrix values.

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
            name="confusion_counts",
            pred_key=pred_key,
            target_key=target_key,
            threshold=threshold,
        )

    def compute(self) -> float:
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    def counts(self) -> dict[str, int]:
        """Return the accumulated confusion matrix as a dict."""
        return self.confusion()
