"""Candidate-level censored-performance risk over a mixed finite search space."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SurvivalObservation:
    """One candidate outcome; operational failures and preemptions do not belong here."""

    trial: int
    survived: bool
    fidelity: float = 1.0
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class SurvivalEstimate:
    """Smoothed probability that a candidate reaches the next decision boundary."""

    trial: int
    probability: float
    lower: float
    upper: float
    effective_samples: float


class SurvivalModel:
    """Joint mixed-space k-NN Beta model for censored performance evidence.

    This deliberately predicts survival rather than inventing a final objective for a pruned
    curve.  A Beta(1, 1) prior keeps sparse regions uncertain and makes the signal a soft
    acquisition modifier rather than a hard exclusion rule.
    """

    def __init__(self, candidates: Mapping[int, Mapping[str, Any]]) -> None:
        self.candidates = {
            int(trial): {str(name): value for name, value in parameters.items()}
            for trial, parameters in candidates.items()
        }
        self._numeric_bounds = self._bounds(tuple(self.candidates.values()))

    def predict(
        self,
        trial: int,
        observations: Sequence[SurvivalObservation],
        *,
        neighbours: int = 8,
    ) -> SurvivalEstimate:
        """Return a local posterior and conservative 90% uncertainty interval."""
        if trial not in self.candidates:
            raise KeyError(f"Unknown candidate {trial}.")
        eligible = [value for value in observations if value.trial in self.candidates]
        local = sorted(
            eligible,
            key=lambda value: (
                self._distance(self.candidates[trial], self.candidates[value.trial]),
                value.trial,
            ),
        )[: max(1, neighbours)]
        alpha = beta = 1.0
        effective = 0.0
        for observation in local:
            distance = self._distance(self.candidates[trial], self.candidates[observation.trial])
            fidelity = min(1.0, max(0.05, float(observation.fidelity)))
            weight = max(0.0, float(observation.weight)) * fidelity * math.exp(-3.0 * distance)
            effective += weight
            if observation.survived:
                alpha += weight
            else:
                beta += weight
        total = alpha + beta
        probability = alpha / total
        deviation = math.sqrt(alpha * beta / (total * total * (total + 1.0)))
        return SurvivalEstimate(
            trial,
            probability,
            max(0.0, probability - 1.645 * deviation),
            min(1.0, probability + 1.645 * deviation),
            effective,
        )

    def predict_all(
        self, observations: Sequence[SurvivalObservation]
    ) -> dict[int, SurvivalEstimate]:
        return {trial: self.predict(trial, observations) for trial in sorted(self.candidates)}

    @staticmethod
    def _bounds(
        candidates: Sequence[Mapping[str, Any]],
    ) -> dict[str, tuple[float, float]]:
        names = {str(name) for candidate in candidates for name in candidate}
        output: dict[str, tuple[float, float]] = {}
        for name in names:
            values = [
                float(candidate[name])
                for candidate in candidates
                if isinstance(candidate.get(name), int | float)
                and not isinstance(candidate.get(name), bool)
                and math.isfinite(float(candidate[name]))
            ]
            if values:
                output[name] = (min(values), max(values))
        return output

    def _distance(self, left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
        names = sorted(set(left) | set(right))
        if not names:
            return 0.0
        squared = 0.0
        for name in names:
            first, second = left.get(name), right.get(name)
            bounds = self._numeric_bounds.get(name)
            if (
                bounds is not None
                and isinstance(first, int | float)
                and not isinstance(first, bool)
                and isinstance(second, int | float)
                and not isinstance(second, bool)
            ):
                low, high = bounds
                width = high - low
                delta = 0.0 if width <= 0 else (float(first) - float(second)) / width
            else:
                delta = 0.0 if first == second else 1.0
            squared += delta * delta
        return math.sqrt(squared / len(names))


__all__ = ["SurvivalEstimate", "SurvivalModel", "SurvivalObservation"]
