"""Fixed-memory histogram approximation of multiclass average precision."""

from __future__ import annotations

import torch

from lambdaforge.metrics.classification.multiclass.MulticlassCurveAverage import (
    MulticlassCurveAverage,
)
from lambdaforge.metrics.classification.multiclass.StreamingMulticlassCurveMetric import (
    StreamingMulticlassCurveMetric,
)
from lambdaforge.metrics.classification.multiclass.UndefinedClassPolicy import (
    UndefinedClassPolicy,
)


class StreamingMulticlassAUPRC(StreamingMulticlassCurveMetric):
    """Approximate one-vs-rest multiclass average precision with bounded state."""

    def __init__(
        self,
        *,
        num_classes: int,
        num_bins: int = 4096,
        average: MulticlassCurveAverage | str = MulticlassCurveAverage.MACRO,
        undefined_class_policy: UndefinedClassPolicy | str = UndefinedClassPolicy.IGNORE,
        pred_key: str = "logits",
        target_key: str = "y",
        from_logits: bool = True,
        validate_probability_sum: bool = True,
        probability_tolerance: float = 1e-5,
    ) -> None:
        super().__init__(
            name="streaming_multiclass_auprc",
            num_classes=num_classes,
            num_bins=num_bins,
            average=average,
            undefined_class_policy=undefined_class_policy,
            pred_key=pred_key,
            target_key=target_key,
            from_logits=from_logits,
            validate_probability_sum=validate_probability_sum,
            probability_tolerance=probability_tolerance,
        )

    def _score_counts(
        self,
        positive_counts: torch.Tensor,
        negative_counts: torch.Tensor,
    ) -> float:
        positive = positive_counts.flip(0).to(dtype=torch.float64)
        negative = negative_counts.flip(0).to(dtype=torch.float64)
        total_positive = positive.sum()
        total_negative = negative.sum()
        if total_positive.item() == 0 or total_negative.item() == 0:
            return float("nan")
        cumulative_positive = torch.cumsum(positive, dim=0)
        cumulative_total = cumulative_positive + torch.cumsum(negative, dim=0)
        precision = torch.zeros_like(cumulative_positive)
        defined = cumulative_total > 0
        precision[defined] = cumulative_positive[defined] / cumulative_total[defined]
        recall_increment = positive / total_positive
        return float((precision * recall_increment).sum().item())
