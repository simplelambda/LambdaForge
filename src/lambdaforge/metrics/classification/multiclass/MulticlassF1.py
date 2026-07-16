"""Multiclass F1 score."""

from lambdaforge.metrics.classification.multiclass.MulticlassMetric import MulticlassMetric


class MulticlassF1(MulticlassMetric):
    """Compute the macro-averaged one-vs-rest F1 score."""

    def __init__(self, pred_key: str = "logits", target_key: str = "y") -> None:
        super().__init__("multiclass_f1", pred_key, target_key)

    def compute(self) -> float:
        predictions, targets = self.values()
        if targets.numel() == 0:
            return float("nan")
        labels = predictions.argmax(dim=-1)
        scores: list[float] = []
        for class_id in range(self.classes(predictions)):
            true_positive = int(((labels == class_id) & (targets == class_id)).sum())
            false_positive = int(((labels == class_id) & (targets != class_id)).sum())
            false_negative = int(((labels != class_id) & (targets == class_id)).sum())
            denominator = 2 * true_positive + false_positive + false_negative
            scores.append(2 * true_positive / denominator if denominator else 0.0)
        return sum(scores) / len(scores)
