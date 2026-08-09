"""Probabilistic learning-curve and hierarchical seed model."""

from __future__ import annotations

import math
from statistics import fmean, variance

import numpy as np

from lambdaforge.hpo.AdaptiveOptimizerState import AdaptiveOptimizerState
from lambdaforge.hpo.PredictiveEstimate import PredictiveEstimate


class LearningCurveModel:
    """Fit deterministic Bayesian basis posteriors to partial curves.

    The model is deliberately provider-neutral. Each seed curve uses Bayesian linear regression
    over smooth fidelity bases capable of trend, curvature, warm-up and plateau behaviour. The
    posterior variance is propagated into a random-effects seed mean. BoTorch's search surrogate
    separately models the population response jointly over configuration and fidelity.
    """

    def __init__(
        self,
        *,
        exploration_weight: float = 1.0,
        prior_precision: float = 1.0,
        noise_floor: float = 1e-3,
    ) -> None:
        if exploration_weight <= 0 or prior_precision <= 0 or noise_floor <= 0:
            raise ValueError("Probabilistic learning-curve settings must be positive.")
        self.exploration_weight = float(exploration_weight)
        self.prior_precision = float(prior_precision)
        self.noise_floor = float(noise_floor)

    def predict_seed(
        self,
        state: AdaptiveOptimizerState,
        config_id: str,
        seed: int,
        *,
        max_budget: int,
    ) -> PredictiveEstimate:
        """Return the full-budget posterior for one configuration and seed."""
        return self.predict_seed_at_budget(
            state,
            config_id,
            seed,
            target_budget=max_budget,
            max_budget=max_budget,
        )

    def predict_seed_at_budget(
        self,
        state: AdaptiveOptimizerState,
        config_id: str,
        seed: int,
        *,
        target_budget: int,
        max_budget: int,
    ) -> PredictiveEstimate:
        """Return a posterior at an explicit cumulative fidelity."""
        if max_budget < 1 or not 0 <= target_budget <= max_budget:
            raise ValueError("Learning-curve budgets require 0 <= target <= max and max > 0.")
        points = self._curve_points(state, config_id, seed)
        global_points: dict[tuple[str, int, int], float] = {}
        for observation in state.observations:
            for budget, score in observation.curve:
                if math.isfinite(score):
                    global_points[(observation.config_id, observation.seed, budget)] = score
        global_scores = list(global_points.values())
        global_mean = fmean(global_scores) if global_scores else 0.0
        global_deviation = math.sqrt(variance(global_scores)) if len(global_scores) > 1 else 1.0
        if not points:
            return PredictiveEstimate.normal(
                global_mean,
                max(global_deviation, self.noise_floor),
                quantile=0.95,
                source="probabilistic_curve_prior",
            )

        budgets = np.asarray([budget / max_budget for budget, _ in points], dtype=np.float64)
        scores = np.asarray([score for _, score in points], dtype=np.float64)
        design = self._basis(budgets)
        target = self._basis(np.asarray([target_budget / max_budget], dtype=np.float64))[0]
        scale = max(global_deviation, float(np.std(scores)), self.noise_floor)
        noise = max(self.noise_floor, scale / math.sqrt(max(4, len(points) * 4)))
        prior_mean = np.asarray([global_mean, 0.0, 0.0, 0.0], dtype=np.float64)
        precision = np.diag(
            np.asarray([0.1, 1.0, 2.0, 1.5], dtype=np.float64) * self.prior_precision
        )
        posterior_precision = precision + (design.T @ design) / (noise**2)
        rhs = precision @ prior_mean + (design.T @ scores) / (noise**2)
        try:
            posterior_mean = np.linalg.solve(posterior_precision, rhs)
            posterior_covariance = np.linalg.inv(posterior_precision)
        except np.linalg.LinAlgError:
            posterior_mean = np.linalg.pinv(posterior_precision) @ rhs
            posterior_covariance = np.linalg.pinv(posterior_precision)
        mean = float(target @ posterior_mean)
        latent_variance = max(0.0, float(target @ posterior_covariance @ target))
        extrapolation = max(0.0, target_budget - points[-1][0]) / max_budget
        deviation = math.sqrt(latent_variance + noise**2 + (scale * extrapolation * 0.15) ** 2)
        deviation *= self.exploration_weight
        return PredictiveEstimate.normal(
            mean,
            max(deviation, self.noise_floor),
            quantile=0.95,
            source="bayesian_learning_curve",
        )

    def predict_configuration(
        self,
        state: AdaptiveOptimizerState,
        config_id: str,
        *,
        max_budget: int,
    ) -> PredictiveEstimate:
        """Propagate within-seed and between-seed uncertainty exactly once."""
        seeds = sorted({observation.seed for observation in state.observations_for(config_id)})
        if not seeds:
            return self.predict_seed(state, config_id, 0, max_budget=max_budget)
        estimates = [
            self.predict_seed(state, config_id, seed, max_budget=max_budget) for seed in seeds
        ]
        return self.combine_seed_estimates(estimates)

    def combine_seed_estimates(
        self,
        estimates: list[PredictiveEstimate],
        *,
        between_seed_variance: float | None = None,
    ) -> PredictiveEstimate:
        """Combine seed posteriors with ``tau²/n + sum(v_s)/n²`` variance.

        When ``tau²`` is not supplied, a method-of-moments estimate removes average within-seed
        estimation variance from the observed sample variance and truncates at zero.
        """
        if not estimates:
            raise ValueError("At least one seed estimate is required.")
        means = [estimate.mean for estimate in estimates]
        within_variances = [estimate.standard_deviation**2 for estimate in estimates]
        count = len(estimates)
        if between_seed_variance is None:
            observed = variance(means) if count > 1 else 0.0
            between_seed_variance = max(0.0, observed - fmean(within_variances))
        if not math.isfinite(between_seed_variance) or between_seed_variance < 0:
            raise ValueError("Between-seed variance must be finite and non-negative.")
        mean_variance = between_seed_variance / count + sum(within_variances) / (count**2)
        return PredictiveEstimate.normal(
            fmean(means),
            max(math.sqrt(mean_variance), self.noise_floor),
            quantile=0.95,
            source="hierarchical_random_effects_mean",
        )

    def probability_competitive(
        self,
        candidate: PredictiveEstimate,
        incumbent: PredictiveEstimate,
        *,
        margin: float,
        direction: str,
    ) -> float:
        """Return Gaussian posterior probability of practical competitiveness."""
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

    def probability_configuration_competitive(
        self,
        state: AdaptiveOptimizerState,
        candidate_id: str,
        incumbent_id: str,
        *,
        max_budget: int,
        margin: float,
        direction: str,
    ) -> float:
        """Prefer paired shared-seed differences, falling back to independent posteriors."""
        candidate_seeds = {item.seed for item in state.observations_for(candidate_id)}
        incumbent_seeds = {item.seed for item in state.observations_for(incumbent_id)}
        shared = sorted(candidate_seeds & incumbent_seeds)
        if shared:
            differences: list[PredictiveEstimate] = []
            sign = 1.0 if direction == "maximize" else -1.0
            for seed in shared:
                candidate = self.predict_seed(state, candidate_id, seed, max_budget=max_budget)
                incumbent = self.predict_seed(state, incumbent_id, seed, max_budget=max_budget)
                differences.append(
                    PredictiveEstimate.normal(
                        sign * (candidate.mean - incumbent.mean),
                        math.sqrt(
                            candidate.standard_deviation**2 + incumbent.standard_deviation**2
                        ),
                        quantile=0.95,
                        source="paired_seed_difference",
                    )
                )
            paired = self.combine_seed_estimates(differences)
            return paired.probability_at_least(-margin)
        return self.probability_competitive(
            self.predict_configuration(state, candidate_id, max_budget=max_budget),
            self.predict_configuration(state, incumbent_id, max_budget=max_budget),
            margin=margin,
            direction=direction,
        )

    @staticmethod
    def _curve_points(
        state: AdaptiveOptimizerState, config_id: str, seed: int
    ) -> list[tuple[int, float]]:
        points: dict[int, float] = {}
        for observation in state.observations_for(config_id, seed=seed):
            points.update(observation.curve)
            if observation.score is not None:
                points[observation.budget] = observation.score
        return sorted(points.items())

    @staticmethod
    def _basis(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, 0.0, 1.0)
        return np.column_stack(
            (
                np.ones_like(clipped),
                clipped,
                clipped**2,
                np.log1p(9.0 * clipped) / math.log(10.0),
            )
        )
