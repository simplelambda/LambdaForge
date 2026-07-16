"""Coefficient of determination."""

from lambdaforge.metrics.regression.RegressionMetric import RegressionMetric


class R2Score(RegressionMetric):
    """Compute the coefficient of determination, R squared."""

    def __init__(self, pred_key: str = "pred", target_key: str = "y") -> None:
        super().__init__("r2", pred_key, target_key, higher_is_better=True)

    def compute(self) -> float:
        predictions, targets = self.values()
        if predictions.numel() == 0:
            return float("nan")
        residual = ((targets - predictions) ** 2).sum()
        total = ((targets - targets.mean()) ** 2).sum()
        if total == 0:
            return float("nan")
        return float((1.0 - residual / total).item())
