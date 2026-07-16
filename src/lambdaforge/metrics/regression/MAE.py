"""Mean absolute error."""

from lambdaforge.metrics.regression.RegressionMetric import RegressionMetric


class MAE(RegressionMetric):
    """Compute the mean absolute difference between predictions and targets."""

    def __init__(self, pred_key: str = "pred", target_key: str = "y") -> None:
        super().__init__("mae", pred_key, target_key)

    def compute(self) -> float:
        predictions, targets = self.values()
        if predictions.numel() == 0:
            return float("nan")
        return float((predictions - targets).abs().mean().item())
