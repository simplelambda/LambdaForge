"""Multiclass area under the precision-recall curve."""

import torch

from lambdaforge.metrics.classification.multiclass.MulticlassMetric import MulticlassMetric


class MulticlassAUPRC(MulticlassMetric):
    """Compute macro multiclass average precision with torchmetrics."""

    def __init__(
        self,
        pred_key: str = "logits",
        target_key: str = "y",
        num_classes: int | None = None,
    ) -> None:
        super().__init__("multiclass_auprc", pred_key, target_key, num_classes)

    def compute(self) -> float:
        predictions, targets = self.values()
        if targets.numel() == 0:
            return float("nan")
        try:
            from torchmetrics.classification import MulticlassAveragePrecision
        except ImportError as error:
            raise ImportError("MulticlassAUPRC requires torchmetrics.") from error
        metric = MulticlassAveragePrecision(
            num_classes=self.classes(predictions),
            average=self.average,
        )
        return float(metric(torch.softmax(predictions, dim=-1), targets).item())
