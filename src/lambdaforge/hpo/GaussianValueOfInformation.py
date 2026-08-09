"""One-step Gaussian Value of Information for heterogeneous HPO actions."""

from __future__ import annotations

import math
from statistics import NormalDist

from lambdaforge.hpo.AdaptiveAction import AdaptiveAction
from lambdaforge.hpo.AdaptiveActionKind import AdaptiveActionKind
from lambdaforge.hpo.AdaptiveOptimizerState import AdaptiveOptimizerState
from lambdaforge.hpo.LearningCurveModel import LearningCurveModel
from lambdaforge.hpo.PredictiveEstimate import PredictiveEstimate


class GaussianValueOfInformation:
    """Approximate one-step Knowledge Gradient by Gaussian moment matching.

    For each action the future posterior mean of its configuration is approximated by a Normal
    random variable whose variance equals the action's expected reduction in posterior variance.
    The expected increase in the best posterior mean then has a closed form. This is a documented
    KG approximation over START/RESUME/ADD_SEED, not ``improvement + uncertainty``.
    """

    def __init__(self, *, max_budget: int, exploration_weight: float = 1.0) -> None:
        if max_budget < 1 or exploration_weight <= 0:
            raise ValueError("Value-of-information settings must be positive.")
        self.max_budget = int(max_budget)
        self.exploration_weight = float(exploration_weight)

    def estimate(
        self,
        action: AdaptiveAction,
        state: AdaptiveOptimizerState,
        model: LearningCurveModel,
        predictions: dict[str, PredictiveEstimate],
        *,
        direction: str,
        risk_type: str = "mean",
        risk_lambda: float = 0.0,
    ) -> float:
        """Return expected terminal-value improvement under the moment model."""
        prediction = predictions[action.config_id]
        sign = 1.0 if direction == "maximize" else -1.0
        candidate_mean = sign * prediction.mean
        if risk_type == "mean_minus_std":
            candidate_mean -= risk_lambda * prediction.standard_deviation
        alternatives = [
            (sign * estimate.mean)
            - (risk_lambda * estimate.standard_deviation if risk_type == "mean_minus_std" else 0.0)
            for config_id, estimate in predictions.items()
            if config_id != action.config_id
        ]
        alternative_best = max(alternatives, default=candidate_mean)
        current_value = max(alternative_best, candidate_mean)
        shift_deviation = self._posterior_mean_shift(action, state, model, prediction)
        if shift_deviation <= 0:
            return 0.0
        expected_value = self._expected_maximum(
            candidate_mean,
            shift_deviation * self.exploration_weight,
            alternative_best,
        )
        return max(0.0, expected_value - current_value)

    def _posterior_mean_shift(
        self,
        action: AdaptiveAction,
        state: AdaptiveOptimizerState,
        model: LearningCurveModel,
        prediction: PredictiveEstimate,
    ) -> float:
        variance = prediction.standard_deviation**2
        if variance <= 0:
            return 0.0
        if action.kind is AdaptiveActionKind.ADD_SEED:
            seed_count = max(
                1,
                len({item.seed for item in state.observations_for(action.config_id)}),
            )
            future_variance = variance * seed_count / (seed_count + 1)
        elif action.kind is AdaptiveActionKind.CONFIRM:
            future_variance = variance * 0.5
        else:
            target = model.predict_seed_at_budget(
                state,
                action.config_id,
                action.seed,
                target_budget=action.target_budget,
                max_budget=self.max_budget,
            )
            fidelity = max(0.0, min(1.0, action.target_budget / self.max_budget))
            prior_fidelity = max(0.0, min(1.0, action.current_budget / self.max_budget))
            incremental_information = max(1e-6, math.sqrt(fidelity) - math.sqrt(prior_fidelity))
            signal_fraction = variance / max(
                variance + target.standard_deviation**2,
                1e-12,
            )
            reduction = min(0.95, incremental_information * signal_fraction)
            future_variance = variance * (1.0 - reduction)
        return math.sqrt(max(0.0, variance - future_variance))

    @staticmethod
    def _expected_maximum(mean: float, deviation: float, threshold: float) -> float:
        if deviation <= 0:
            return max(mean, threshold)
        z = (mean - threshold) / deviation
        normal = NormalDist()
        density = math.exp(-(z**2) / 2.0) / math.sqrt(2.0 * math.pi)
        return threshold + (mean - threshold) * normal.cdf(z) + deviation * density
