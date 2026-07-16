"""Pearson correlation coefficient."""

from lambdaforge.metrics.regression.RegressionMetric import RegressionMetric


class PearsonCorrelation(RegressionMetric):
    """Compute linear correlation between predictions and targets."""

    def __init__(self, pred_key: str = "pred", target_key: str = "y") -> None:
        super().__init__("pearson", pred_key, target_key, higher_is_better=True)

    def compute(self) -> float:
        predictions, targets = self.values()
        if predictions.numel() < 2:
            return float("nan")
        predictions = predictions - predictions.mean()
        targets = targets - targets.mean()
        denominator = predictions.square().sum().sqrt() * targets.square().sum().sqrt()
        if denominator == 0:
            return float("nan")
        return float(((predictions * targets).sum() / denominator).item())
