"""Multiclass accuracy."""

from lambdaforge.metrics.classification.multiclass.MulticlassMetric import MulticlassMetric


class MulticlassAccuracy(MulticlassMetric):
    """Compute the fraction of samples assigned to the correct class."""

    def __init__(self, pred_key: str = "logits", target_key: str = "y") -> None:
        super().__init__("multiclass_accuracy", pred_key, target_key)

    def compute(self) -> float:
        predictions, targets = self.values()
        if targets.numel() == 0:
            return float("nan")
        return float((predictions.argmax(dim=-1) == targets).float().mean().item())
