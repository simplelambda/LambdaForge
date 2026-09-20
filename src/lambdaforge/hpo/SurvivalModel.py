"""Candidate-level censored-performance risk over a mixed finite search space."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lambdaforge.hpo.ParameterSpace import ParameterSpace


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


@dataclass(frozen=True, slots=True)
class SurvivalAcquisitionPolicy:
    """Interpretable bounded adjustment of objective acquisition by censored evidence."""

    minimum_survival_factor: float = 0.5
    uncertainty_exploration_weight: float = 0.1

    def adjust(self, objective_score: float, estimate: SurvivalEstimate) -> float:
        """Discount likely-pruned regions softly while retaining uncertainty exploration."""
        probability = min(1.0, max(0.0, estimate.probability))
        factor = self.minimum_survival_factor + (1.0 - self.minimum_survival_factor) * probability
        uncertainty = max(0.0, estimate.upper - estimate.lower)
        return float(objective_score) * factor + self.uncertainty_exploration_weight * uncertainty

    def to_dict(self) -> dict[str, float | str]:
        return {
            "method": "objective-score-times-survival-factor-plus-uncertainty",
            "minimum_survival_factor": self.minimum_survival_factor,
            "uncertainty_exploration_weight": self.uncertainty_exploration_weight,
        }


class SurvivalModel:
    """Joint mixed-space k-NN Beta model for censored performance evidence.

    This deliberately predicts survival rather than inventing a final objective for a pruned
    curve.  A Beta(1, 1) prior keeps sparse regions uncertain and makes the signal a soft
    acquisition modifier rather than a hard exclusion rule.
    """

    def __init__(
        self,
        candidates: Mapping[int, Mapping[str, Any]],
        *,
        parameter_space: ParameterSpace | Mapping[str, Any] | None = None,
    ) -> None:
        self.candidates = {
            int(trial): {str(name): value for name, value in parameters.items()}
            for trial, parameters in candidates.items()
        }
        self.parameter_space = (
            parameter_space
            if isinstance(parameter_space, ParameterSpace)
            else ParameterSpace.from_schema(parameter_space, tuple(self.candidates.values()))
        )

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
                self.parameter_space.distance(self.candidates[trial], self.candidates[value.trial]),
                value.trial,
            ),
        )[: max(1, neighbours)]
        alpha = beta = 1.0
        effective = 0.0
        for observation in local:
            distance = self.parameter_space.distance(
                self.candidates[trial], self.candidates[observation.trial]
            )
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


__all__ = [
    "SurvivalAcquisitionPolicy",
    "SurvivalEstimate",
    "SurvivalModel",
    "SurvivalObservation",
]
