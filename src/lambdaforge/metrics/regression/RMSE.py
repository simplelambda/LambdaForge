"""Root mean squared error."""

from lambdaforge.metrics.regression.RegressionMetric import RegressionMetric


class RMSE(RegressionMetric):
    """Compute the square root of the mean squared error."""

    def __init__(self, pred_key: str = "pred", target_key: str = "y") -> None:
        super().__init__("rmse", pred_key, target_key)

    def compute(self) -> float:
        predictions, targets = self.values()
        if predictions.numel() == 0:
            return float("nan")
        return float(((predictions - targets) ** 2).mean().sqrt().item())
