"""Multiclass area under the ROC curve."""

import torch

from lambdaforge.metrics.classification.multiclass.MulticlassMetric import MulticlassMetric


class MulticlassAUROC(MulticlassMetric):
    """Compute macro one-vs-rest AUROC with torchmetrics."""

    def __init__(
        self,
        pred_key: str = "logits",
        target_key: str = "y",
        num_classes: int | None = None,
    ) -> None:
        super().__init__("multiclass_auroc", pred_key, target_key, num_classes)

    def compute(self) -> float:
        predictions, targets = self.values()
        if targets.numel() == 0:
            return float("nan")
        try:
            from torchmetrics.classification import MulticlassAUROC as TorchMetric
        except ImportError as error:
            raise ImportError("MulticlassAUROC requires torchmetrics.") from error
        metric = TorchMetric(num_classes=self.classes(predictions), average=self.average)
        return float(metric(torch.softmax(predictions, dim=-1), targets).item())
