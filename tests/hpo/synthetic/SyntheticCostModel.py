"""Exact synthetic action-cost model."""

from __future__ import annotations

from collections.abc import Mapping

from lambdaforge.hpo.AdaptiveAction import AdaptiveAction
from lambdaforge.hpo.AdaptiveOptimizerState import AdaptiveOptimizerState
from lambdaforge.hpo.PredictiveEstimate import PredictiveEstimate


class SyntheticCostModel:
    """Return configured deterministic costs by action identifier."""

    def __init__(self, costs: Mapping[str, float]) -> None:
        self.costs = {str(key): float(value) for key, value in costs.items()}

    def predict(self, action: AdaptiveAction, state: AdaptiveOptimizerState) -> PredictiveEstimate:
        """Return an exact cost and ignore unrelated controller state."""
        del state
        return PredictiveEstimate.normal(
            self.costs[action.action_id],
            0.0,
            quantile=0.9,
            source="synthetic_cost",
        )
