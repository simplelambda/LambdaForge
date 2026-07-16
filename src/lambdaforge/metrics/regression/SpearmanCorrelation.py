"""Spearman rank correlation coefficient."""

import torch

from lambdaforge.metrics.regression.RegressionMetric import RegressionMetric


class SpearmanCorrelation(RegressionMetric):
    """Compute rank correlation, including average ranks for tied values."""

    def __init__(self, pred_key: str = "pred", target_key: str = "y") -> None:
        super().__init__("spearman", pred_key, target_key, higher_is_better=True)

    def compute(self) -> float:
        predictions, targets = self.values()
        if predictions.numel() < 2:
            return float("nan")
        prediction_ranks = self._rank(predictions)
        target_ranks = self._rank(targets)
        prediction_ranks -= prediction_ranks.mean()
        target_ranks -= target_ranks.mean()
        denominator = prediction_ranks.square().sum().sqrt() * target_ranks.square().sum().sqrt()
        if denominator == 0:
            return float("nan")
        return float(((prediction_ranks * target_ranks).sum() / denominator).item())

    @staticmethod
    def _rank(values: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(values, stable=True)
        sorted_values = values[order]
        ranks = torch.empty_like(values, dtype=torch.float32)
        start = 0
        while start < values.numel():
            end = start + 1
            while end < values.numel() and sorted_values[end] == sorted_values[start]:
                end += 1
            average_rank = (start + 1 + end) / 2.0
            ranks[order[start:end]] = average_rank
            start = end
        return ranks
