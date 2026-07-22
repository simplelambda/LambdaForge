"""Fixed-memory histogram approximation of multiclass AUROC."""

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


class StreamingMulticlassAUROC(StreamingMulticlassCurveMetric):
    """Approximate one-vs-rest multiclass AUROC with bounded state."""

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
            name="streaming_multiclass_auroc",
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
        positive = positive_counts.to(dtype=torch.float64)
        negative = negative_counts.to(dtype=torch.float64)
        total_positive = positive.sum()
        total_negative = negative.sum()
        if total_positive.item() == 0 or total_negative.item() == 0:
            return float("nan")
        negatives_below = torch.cumsum(negative, dim=0) - negative
        concordant = (positive * (negatives_below + 0.5 * negative)).sum()
        return float((concordant / (total_positive * total_negative)).item())
