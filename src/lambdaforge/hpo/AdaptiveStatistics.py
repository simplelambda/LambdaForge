"""Small probabilistic primitives for adaptive seed allocation."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from lambdaforge.hpo.ScientificDesign import SeedNoiseModel


@dataclass(frozen=True, slots=True)
class CandidateEstimate:
    """Normal approximation for one configuration's mean over random seeds."""

    mean: float
    standard_error: float
    samples: int
    seed_noise_calibrated: bool = False

    def probability_at_least(self, threshold: float) -> float:
        """Return ``P(mean >= threshold)`` without requiring SciPy."""
        if self.standard_error <= 0:
            return 1.0 if self.mean >= threshold else 0.0
        z = (self.mean - threshold) / self.standard_error
        return statistics.NormalDist().cdf(z)


@dataclass(frozen=True, slots=True)
class SeedRaceDecision:
    """Evidence used to decide whether another seed can change the ranking."""

    trial: int
    probability_competitive: float
    expected_uncertainty_reduction: float
    completed_seeds: int
    current_standard_error: float
    purpose: str = "ADD_SEED"
    recommended_seed: int | None = None
    comparison_trials: tuple[int, ...] = ()
    information_value: float = 0.0


class AdaptiveSeedRacer:
    """Allocate shared seeds using probability of practical competitiveness.

    Completed seed outcomes estimate a hierarchical random-effects mean. Between-candidate spread
    is never used as seed noise: before within-candidate evidence exists, uncertainty remains
    explicitly uncalibrated and replication receives calibration value. Shared seeds are compared
    as paired differences whenever possible.
    """

    def __init__(
        self,
        *,
        mode: str,
        margin: float = 0.0,
        probability_threshold: float = 0.1,
        variance_floor: float = 1e-12,
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("Seed racer mode must be min or max.")
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("Seed racer margin must be finite and non-negative.")
        if not 0 <= probability_threshold <= 1:
            raise ValueError("Seed racer probability threshold must be in [0, 1].")
        if not math.isfinite(variance_floor) or variance_floor <= 0:
            raise ValueError("Seed racer variance floor must be finite and positive.")
        self.mode = mode
        self.margin = float(margin)
        self.probability_threshold = float(probability_threshold)
        self.variance_floor = float(variance_floor)

    def estimates(
        self, outcomes: Mapping[int, Mapping[int | None, float]]
    ) -> dict[int, CandidateEstimate]:
        """Estimate candidates with pooled *within-candidate* seed variance only."""
        noise = SeedNoiseModel.fit(outcomes)
        estimates: dict[int, CandidateEstimate] = {}
        for trial, values_by_seed in outcomes.items():
            values = tuple(float(value) for value in values_by_seed.values())
            if not values:
                continue
            variance = statistics.variance(values) if len(values) > 1 else noise.variance
            estimates[int(trial)] = CandidateEstimate(
                statistics.fmean(values),
                (
                    math.sqrt(max(variance, self.variance_floor) / len(values))
                    if variance is not None
                    else math.inf
                ),
                len(values),
                noise.calibrated,
            )
        return estimates

    def decisions(
        self,
        outcomes: Mapping[int, Mapping[int | None, float]],
        *,
        eligible: Sequence[int] | None = None,
        available_seeds: Mapping[int, Sequence[int]] | None = None,
    ) -> tuple[SeedRaceDecision, ...]:
        """Rank extra seeds by uncertainty reduction in incumbent/challenger comparisons."""
        estimates = self.estimates(outcomes)
        if not estimates:
            return ()
        incumbent = (
            max(estimates, key=lambda trial: estimates[trial].mean)
            if self.mode == "max"
            else min(estimates, key=lambda trial: estimates[trial].mean)
        )
        allowed = set(estimates) if eligible is None else set(eligible)
        competitor_probabilities: dict[int, float] = {}
        for trial, estimate in estimates.items():
            competitor_probabilities[trial] = (
                1.0
                if trial == incumbent
                else self._probability_competitive(
                    outcomes.get(trial, {}),
                    outcomes.get(incumbent, {}),
                    estimate,
                    estimates[incumbent],
                )
            )
        decisions: list[SeedRaceDecision] = []
        for trial in sorted(allowed & estimates.keys()):
            probability = competitor_probabilities[trial]
            if probability < self.probability_threshold:
                continue
            estimate = estimates[trial]
            comparisons = tuple(
                other
                for other in sorted(estimates)
                if other != trial
                and (
                    trial == incumbent
                    or other == incumbent
                    or competitor_probabilities[other] >= self.probability_threshold
                )
            )
            pair_values = [
                self._comparison_information(
                    trial,
                    other,
                    outcomes,
                    estimates,
                    competitor_probabilities,
                    incumbent,
                )
                for other in comparisons
            ]
            information = 1.0 - math.prod(1.0 - min(1.0, max(0.0, value)) for value in pair_values)
            reduction = information * (
                estimate.standard_error if math.isfinite(estimate.standard_error) else 1.0
            )
            recommended_seed = self._recommended_shared_seed(
                trial,
                comparisons,
                outcomes,
                available_seeds or {},
                competitor_probabilities,
            )
            purpose = (
                "CALIBRATE_SEED_NOISE"
                if not estimate.seed_noise_calibrated
                else "ADD_SHARED_SEED"
                if recommended_seed is not None
                else "REPLICATE_INCUMBENT"
                if trial == incumbent
                else "ADD_SEED"
            )
            decisions.append(
                SeedRaceDecision(
                    trial,
                    probability,
                    reduction,
                    estimate.samples,
                    estimate.standard_error,
                    purpose,
                    recommended_seed,
                    comparisons,
                    information,
                )
            )
        return tuple(
            sorted(
                decisions,
                key=lambda item: (
                    -item.expected_uncertainty_reduction,
                    -item.probability_competitive,
                    item.trial,
                ),
            )
        )

    def _probability_competitive(
        self,
        candidate_values: Mapping[int | None, float],
        incumbent_values: Mapping[int | None, float],
        candidate: CandidateEstimate,
        incumbent: CandidateEstimate,
    ) -> float:
        shared = [
            seed for seed in candidate_values if seed is not None and seed in incumbent_values
        ]
        sign = 1.0 if self.mode == "max" else -1.0
        if len(shared) >= 2:
            differences = [
                sign * (float(candidate_values[seed]) - float(incumbent_values[seed]))
                for seed in shared
            ]
            mean = statistics.fmean(differences)
            error = math.sqrt(
                max(statistics.variance(differences), self.variance_floor) / len(differences)
            )
        else:
            mean = sign * (candidate.mean - incumbent.mean)
            error = math.sqrt(candidate.standard_error**2 + incumbent.standard_error**2)
        if not math.isfinite(error):
            return 0.5
        if error <= 0:
            return 1.0 if mean >= -self.margin else 0.0
        return 1.0 - statistics.NormalDist(mean, error).cdf(-self.margin)

    def _comparison_information(
        self,
        trial: int,
        other: int,
        outcomes: Mapping[int, Mapping[int | None, float]],
        estimates: Mapping[int, CandidateEstimate],
        probabilities: Mapping[int, float],
        incumbent: int,
    ) -> float:
        left, right = estimates[trial], estimates[other]
        if not left.seed_noise_calibrated or not right.seed_noise_calibrated:
            uncertainty = 1.0
        else:
            variance = left.standard_error**2 + right.standard_error**2
            shared = set(outcomes.get(trial, {})) & set(outcomes.get(other, {})) - {None}
            if len(shared) >= 2:
                differences = [
                    float(outcomes[trial][seed]) - float(outcomes[other][seed]) for seed in shared
                ]
                variance = max(
                    statistics.variance(differences) / len(differences), self.variance_floor
                )
            uncertainty = 1.0 - math.exp(
                -math.sqrt(variance) / max(self.margin, math.sqrt(variance), 1e-12)
            )
        # Candidate weights vanish naturally as practical competitiveness vanishes.  The incumbent
        # is valuable against every plausible challenger rather than being penalized for p~=1.
        weight = probabilities[other] if trial == incumbent else probabilities[trial]
        sample_reduction = 1.0 - math.sqrt(left.samples / (left.samples + 1.0))
        return min(
            1.0, max(0.0, weight * uncertainty * max(sample_reduction, 1.0 / (left.samples + 1.0)))
        )

    @staticmethod
    def _recommended_shared_seed(
        trial: int,
        comparisons: Sequence[int],
        outcomes: Mapping[int, Mapping[int | None, float]],
        available: Mapping[int, Sequence[int]],
        probabilities: Mapping[int, float],
    ) -> int | None:
        attempted = set(outcomes.get(trial, {}))
        allowed = set(available.get(trial, ())) - attempted
        if not allowed:
            return None
        scores: dict[int, float] = {}
        for other in comparisons:
            for seed in set(outcomes.get(other, {})) & allowed - {None}:
                assert isinstance(seed, int)
                scores[seed] = scores.get(seed, 0.0) + probabilities.get(other, 0.0)
        return max(scores, key=lambda seed: (scores[seed], -seed)) if scores else None


__all__ = ["AdaptiveSeedRacer", "CandidateEstimate", "SeedRaceDecision"]
