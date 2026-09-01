"""Small probabilistic primitives for adaptive seed allocation."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CandidateEstimate:
    """Normal approximation for one configuration's mean over random seeds."""

    mean: float
    standard_error: float
    samples: int

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


class AdaptiveSeedRacer:
    """Allocate shared seeds using probability of practical competitiveness.

    The model deliberately stays small: completed seed outcomes estimate a random-effects mean,
    while a pooled between-seed variance prevents one-observation candidates from becoming
    spuriously certain.  Shared seeds are compared as paired differences whenever possible.
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
        """Estimate every candidate using one shared pooled seed-variance prior."""
        pooled = self._pooled_variance(outcomes)
        estimates: dict[int, CandidateEstimate] = {}
        for trial, values_by_seed in outcomes.items():
            values = tuple(float(value) for value in values_by_seed.values())
            if not values:
                continue
            variance = statistics.variance(values) if len(values) > 1 else pooled
            estimates[int(trial)] = CandidateEstimate(
                statistics.fmean(values),
                math.sqrt(max(variance, self.variance_floor) / len(values)),
                len(values),
            )
        return estimates

    def decisions(
        self,
        outcomes: Mapping[int, Mapping[int | None, float]],
        *,
        eligible: Sequence[int] | None = None,
    ) -> tuple[SeedRaceDecision, ...]:
        """Rank candidates for an additional seed by expected uncertainty reduction."""
        estimates = self.estimates(outcomes)
        if not estimates:
            return ()
        incumbent = (
            max(estimates, key=lambda trial: estimates[trial].mean)
            if self.mode == "max"
            else min(estimates, key=lambda trial: estimates[trial].mean)
        )
        allowed = set(estimates) if eligible is None else set(eligible)
        decisions: list[SeedRaceDecision] = []
        for trial in sorted(allowed & estimates.keys()):
            probability = (
                1.0
                if trial == incumbent
                else self._probability_competitive(
                    outcomes.get(trial, {}),
                    outcomes.get(incumbent, {}),
                    estimates[trial],
                    estimates[incumbent],
                )
            )
            if probability < self.probability_threshold:
                continue
            estimate = estimates[trial]
            next_error = estimate.standard_error * math.sqrt(
                estimate.samples / (estimate.samples + 1)
            )
            reduction = max(0.0, estimate.standard_error - next_error)
            # Uncertainty matters most close to the decision boundary.  The incumbent remains
            # eligible so its own mean can be estimated rather than treating it as known truth.
            boundary_weight = max(1e-9, 4.0 * probability * (1.0 - probability))
            if trial == incumbent:
                boundary_weight = max(boundary_weight, 0.25)
            decisions.append(
                SeedRaceDecision(
                    trial,
                    probability,
                    reduction * boundary_weight,
                    estimate.samples,
                    estimate.standard_error,
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
        if error <= 0:
            return 1.0 if mean >= -self.margin else 0.0
        return 1.0 - statistics.NormalDist(mean, error).cdf(-self.margin)

    def _pooled_variance(self, outcomes: Mapping[int, Mapping[int | None, float]]) -> float:
        variances: list[tuple[int, float]] = []
        all_values: list[float] = []
        for values_by_seed in outcomes.values():
            values = [float(value) for value in values_by_seed.values()]
            all_values.extend(values)
            if len(values) > 1:
                variances.append((len(values) - 1, statistics.variance(values)))
        weight = sum(item[0] for item in variances)
        if weight:
            return max(
                sum(degrees * variance for degrees, variance in variances) / weight,
                self.variance_floor,
            )
        if len(all_values) > 1:
            return max(statistics.variance(all_values), self.variance_floor)
        scale = max(abs(all_values[0]) * 0.1, 1.0) if all_values else 1.0
        return max(scale**2, self.variance_floor)


__all__ = ["AdaptiveSeedRacer", "CandidateEstimate", "SeedRaceDecision"]
