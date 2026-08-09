"""Lightweight uncertainty-aware action cost model."""

from __future__ import annotations

from statistics import fmean, pstdev

from lambdaforge.hpo.AdaptiveAction import AdaptiveAction
from lambdaforge.hpo.AdaptiveOptimizerState import AdaptiveOptimizerState
from lambdaforge.hpo.PredictiveEstimate import PredictiveEstimate


class EmpiricalCostModel:
    """Predict seconds from observed per-epoch rates with conservative cold start."""

    def __init__(self, *, cold_seconds_per_budget: float = 1.0) -> None:
        if cold_seconds_per_budget <= 0:
            raise ValueError("Cold-start cost rate must be positive.")
        self.cold_seconds_per_budget = float(cold_seconds_per_budget)

    def predict(self, action: AdaptiveAction, state: AdaptiveOptimizerState) -> PredictiveEstimate:
        """Estimate incremental wall time; prefer same-configuration evidence."""
        same = [
            observation
            for observation in state.observations_for(action.config_id)
            if observation.seconds > 0 and observation.budget > 0
        ]
        population = same or [
            observation
            for observation in state.observations
            if observation.seconds > 0 and observation.budget > 0
        ]
        current_budgets = {
            action.action_id: action.current_budget for action in state.completed_actions
        }
        rates = [
            observation.seconds
            / max(1, observation.budget - current_budgets.get(observation.action_id, 0))
            for observation in population
        ]
        source = "same_configuration" if same else "global" if rates else "cold_start"
        mean_rate = fmean(rates) if rates else self.cold_seconds_per_budget
        std_rate = pstdev(rates) if len(rates) > 1 else mean_rate * 0.5
        increment = max(1, action.target_budget - action.current_budget)
        return PredictiveEstimate.normal(
            mean_rate * increment,
            std_rate * increment,
            quantile=0.9,
            source=source,
        )
