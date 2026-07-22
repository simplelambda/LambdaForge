"""Fixed-memory histogram approximation of binary average precision."""

from __future__ import annotations

import torch

from lambdaforge.metrics.classification.binary.StreamingBinaryCurveMetric import (
    StreamingBinaryCurveMetric,
)


class StreamingBinaryAUPRC(StreamingBinaryCurveMetric):
    """Approximate binary AUPRC as binned average precision.

    Bins are traversed from the highest score to the lowest and precision is
    weighted by each bin's recall increment. This matches the non-trapezoidal
    average-precision semantics used by BinaryAUPRC.
    """

    def __init__(
        self,
        pred_key: str = "probs",
        target_key: str = "y",
        num_bins: int = 4096,
        from_logits: bool = False,
    ) -> None:
        super().__init__(
            name="streaming_auprc",
            pred_key=pred_key,
            target_key=target_key,
            num_bins=num_bins,
            from_logits=from_logits,
        )

    def compute(self) -> float:
        """Return precision weighted by recall increments over descending bins."""
        positive = self._positive_counts.flip(0).to(dtype=torch.float64)
        negative = self._negative_counts.flip(0).to(dtype=torch.float64)
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
