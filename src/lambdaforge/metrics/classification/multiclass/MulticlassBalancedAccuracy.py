"""Multiclass balanced accuracy."""

import torch

from lambdaforge.metrics.classification.multiclass.MulticlassMetric import MulticlassMetric


class MulticlassBalancedAccuracy(MulticlassMetric):
    """Compute macro-averaged recall across classes present in the targets."""

    def __init__(self, pred_key: str = "logits", target_key: str = "y") -> None:
        super().__init__("multiclass_balanced_accuracy", pred_key, target_key)

    def compute(self) -> float:
        predictions, targets = self.values()
        if targets.numel() == 0:
            return float("nan")
        labels = predictions.argmax(dim=-1)
        recalls = torch.stack(
            [
                (labels[targets == class_id] == class_id).float().mean()
                for class_id in targets.unique()
            ]
        )
        return float(recalls.mean().item())
