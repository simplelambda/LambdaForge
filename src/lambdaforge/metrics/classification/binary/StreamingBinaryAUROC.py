"""Fixed-memory histogram approximation of binary AUROC."""

from __future__ import annotations

import torch

from lambdaforge.metrics.classification.binary.StreamingBinaryCurveMetric import (
    StreamingBinaryCurveMetric,
)


class StreamingBinaryAUROC(StreamingBinaryCurveMetric):
    """Approximate binary AUROC using a bounded pair of score histograms.

    Samples within one probability bin are treated as tied. The result is
    exact whenever no positive-negative pair with different scores shares a
    bin. Increasing num_bins improves score resolution but does not imply a
    universal numerical error bound.
    """

    def __init__(
        self,
        pred_key: str = "probs",
        target_key: str = "y",
        num_bins: int = 4096,
        from_logits: bool = False,
    ) -> None:
        super().__init__(
            name="streaming_auroc",
            pred_key=pred_key,
            target_key=target_key,
            num_bins=num_bins,
            from_logits=from_logits,
        )

    def compute(self) -> float:
        """Return concordant positive-negative pairs, with half credit for ties."""
        positive = self._positive_counts.to(dtype=torch.float64)
        negative = self._negative_counts.to(dtype=torch.float64)
        total_positive = positive.sum()
        total_negative = negative.sum()
        if total_positive.item() == 0 or total_negative.item() == 0:
            return float("nan")

        negatives_below = torch.cumsum(negative, dim=0) - negative
        concordant = (positive * (negatives_below + 0.5 * negative)).sum()
        return float((concordant / (total_positive * total_negative)).item())
