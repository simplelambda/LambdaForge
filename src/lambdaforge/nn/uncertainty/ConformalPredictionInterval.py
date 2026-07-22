"""Split-conformal prediction intervals for tensor regression outputs."""

from __future__ import annotations

import math

import torch
from torch import nn


class ConformalPredictionInterval(nn.Module):
    """Calibrate absolute residual quantiles and produce distribution-free bands."""

    def __init__(self, miscoverage: float = 0.1) -> None:
        super().__init__()
        if not math.isfinite(miscoverage) or not 0 < miscoverage < 1:
            raise ValueError("miscoverage must be finite and in (0, 1).")
        self.miscoverage = float(miscoverage)
        self.register_buffer("residual_quantile", None, persistent=True)

    @property
    def is_calibrated(self) -> bool:
        """Report whether held-out residuals have been fitted."""
        return self.residual_quantile is not None

    def fit(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> ConformalPredictionInterval:
        """Replace calibration state using finite-sample corrected residual ranks."""
        if predictions.shape != targets.shape or predictions.ndim < 1:
            raise ValueError("predictions and targets must have identical non-scalar shapes.")
        if predictions.shape[0] < 1:
            raise ValueError("At least one calibration sample is required.")
        residuals = (targets.detach() - predictions.detach()).abs()
        if not torch.isfinite(residuals).all():
            raise ValueError("Calibration residuals must be finite.")
        sample_count = predictions.shape[0]
        quantile = min(
            1.0,
            math.ceil((sample_count + 1) * (1.0 - self.miscoverage)) / sample_count,
        )
        self.residual_quantile = torch.quantile(
            residuals,
            quantile,
            dim=0,
            interpolation="higher",
        )
        return self

    def forward(self, predictions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return prediction, lower and upper tensors with unchanged shape."""
        if self.residual_quantile is None:
            raise RuntimeError("ConformalPredictionInterval must be fitted before use.")
        width = self.residual_quantile.to(device=predictions.device, dtype=predictions.dtype)
        return {
            "prediction": predictions,
            "lower": predictions - width,
            "upper": predictions + width,
        }
