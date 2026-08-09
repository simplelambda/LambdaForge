"""Conservative final-performance model from complete learning curves."""

from __future__ import annotations

import math
from statistics import fmean, pstdev

from lambdaforge.hpo.AdaptiveOptimizerState import AdaptiveOptimizerState
from lambdaforge.hpo.PredictiveEstimate import PredictiveEstimate


class LearningCurveModel:
    """Project progress while preserving uncertainty for sparse slow starters."""

    def __init__(self, *, exploration_weight: float = 1.0) -> None:
        if exploration_weight <= 0:
            raise ValueError("Learning-curve exploration weight must be positive.")
        self.exploration_weight = float(exploration_weight)

    def predict_seed(
        self,
        state: AdaptiveOptimizerState,
        config_id: str,
        seed: int,
        *,
        max_budget: int,
    ) -> PredictiveEstimate:
        """Predict one seed's full-budget score from every recorded curve point."""
        observations = state.observations_for(config_id, seed=seed)
        points: dict[int, float] = {}
        for observation in observations:
            points.update(observation.curve)
            if observation.score is not None:
                points[observation.budget] = observation.score
        ordered = sorted(points.items())
        global_scores = [
            observation.score for observation in state.observations if observation.score is not None
        ]
        global_mean = fmean(global_scores) if global_scores else 0.0
        global_std = pstdev(global_scores) if len(global_scores) > 1 else 1.0
        if not ordered:
            return PredictiveEstimate.normal(
                global_mean,
                max(global_std, 1e-6),
                quantile=0.95,
                source="cold_start_global",
            )
        last_budget, last_score = ordered[-1]
        if last_budget >= max_budget:
            return PredictiveEstimate.normal(
                last_score,
                max(global_std * 0.05, 1e-6),
                quantile=0.95,
                source="observed_full_budget",
            )
        slopes = [
            (right_score - left_score) / (right_budget - left_budget)
            for (left_budget, left_score), (right_budget, right_score) in zip(
                ordered, ordered[1:], strict=False
            )
            if right_budget > left_budget
        ]
        recent = slopes[-3:]
        slope = fmean(recent) if recent else 0.0
        remaining = max_budget - last_budget
        # Positive progress is deliberately allowed to extrapolate farther than decline. This
        # gives slow starters time while uncertainty still shrinks with accumulated fidelity.
        decay = 0.65 if slope > 0 else 0.25
        projected = last_score + slope * remaining * decay
        slope_noise = pstdev(recent) * remaining if len(recent) > 1 else abs(slope) * remaining
        sparsity = global_std / math.sqrt(max(1, len(ordered)))
        deviation = max(1e-6, slope_noise, sparsity * self.exploration_weight)
        return PredictiveEstimate.normal(
            projected,
            deviation,
            quantile=0.95,
            source="learning_curve_projection",
        )

    def predict_configuration(
        self,
        state: AdaptiveOptimizerState,
        config_id: str,
        *,
        max_budget: int,
    ) -> PredictiveEstimate:
        """Combine seed-level final predictions without penalizing variance in the mean."""
        seeds = sorted({observation.seed for observation in state.observations_for(config_id)})
        if not seeds:
            return self.predict_seed(state, config_id, 0, max_budget=max_budget)
        estimates = [
            self.predict_seed(state, config_id, seed, max_budget=max_budget) for seed in seeds
        ]
        means = [estimate.mean for estimate in estimates]
        between = pstdev(means) if len(means) > 1 else 0.0
        within = math.sqrt(sum(estimate.standard_deviation**2 for estimate in estimates)) / len(
            estimates
        )
        standard_error = math.sqrt(between**2 + within**2) / math.sqrt(len(estimates))
        return PredictiveEstimate.normal(
            fmean(means),
            max(standard_error, 1e-6),
            quantile=0.95,
            source="hierarchical_seed_mean",
        )

    def probability_competitive(
        self,
        candidate: PredictiveEstimate,
        incumbent: PredictiveEstimate,
        *,
        margin: float,
        direction: str,
    ) -> float:
        """Approximate posterior probability of being within the incumbent margin."""
        candidate_mean = candidate.mean if direction == "maximize" else -candidate.mean
        incumbent_mean = incumbent.mean if direction == "maximize" else -incumbent.mean
        deviation = math.sqrt(candidate.standard_deviation**2 + incumbent.standard_deviation**2)
        difference = PredictiveEstimate.normal(
            candidate_mean - incumbent_mean,
            deviation,
            quantile=0.95,
            source="competitive_difference",
        )
        return difference.probability_at_least(-margin)
