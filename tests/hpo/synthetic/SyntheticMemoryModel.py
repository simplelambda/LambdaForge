"""Exact synthetic candidate-memory model."""

from __future__ import annotations

from lambdaforge.hpo.AdaptiveAction import AdaptiveAction
from lambdaforge.hpo.AdaptiveOptimizerState import AdaptiveOptimizerState
from lambdaforge.hpo.PredictiveEstimate import PredictiveEstimate


class SyntheticMemoryModel:
    """Scale memory with a generic numeric resource feature."""

    headroom_bytes = 0

    def __init__(self, *, bytes_per_unit: float = 1.0, relative_uncertainty: float = 0.0) -> None:
        self.bytes_per_unit = float(bytes_per_unit)
        self.relative_uncertainty = float(relative_uncertainty)

    def predict(self, action: AdaptiveAction, state: AdaptiveOptimizerState) -> PredictiveEstimate:
        """Return a Gaussian prediction from ``resource_features['size']``."""
        del state
        mean = float(action.resource_features.get("size", 0)) * self.bytes_per_unit
        return PredictiveEstimate.normal(
            mean,
            mean * self.relative_uncertainty,
            quantile=0.99,
            source="synthetic_memory",
        )
