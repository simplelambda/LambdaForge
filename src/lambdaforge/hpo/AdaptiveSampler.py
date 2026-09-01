"""Dependency-light sequential selection over a reproducible candidate pool."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lambdaforge.hpo.SurvivalModel import SurvivalEstimate


@dataclass(frozen=True, slots=True)
class CandidateObservation:
    """One completed candidate score available to the proposal controller."""

    trial: int
    value: float
    standard_error: float | None = None
    fidelity: float = 1.0


class AdaptiveSampler:
    """Choose candidates using a small k-nearest-neighbour acquisition function.

    The complete candidate pool is deterministic scientific planning state, but only selected
    candidates become Trials. Selection balances the predicted objective near good observations
    with distance from everything already observed. This keeps the base install free of a large
    optimizer dependency while making proposal order depend on actual results.
    """

    def __init__(self, candidates: Mapping[int, Mapping[str, Any]], *, mode: str) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("Adaptive sampler mode must be min or max.")
        self.candidates = {
            int(trial): {str(name): value for name, value in parameters.items()}
            for trial, parameters in candidates.items()
        }
        self.mode = mode
        self._numeric_bounds = self._bounds(tuple(self.candidates.values()))

    def initial(self, count: int) -> tuple[int, ...]:
        """Return a deterministic space-filling startup set."""
        available = sorted(self.candidates)
        if count >= len(available):
            return tuple(available)
        selected = [available[0]]
        while len(selected) < count:
            remaining = [trial for trial in available if trial not in selected]
            selected.append(
                max(
                    remaining,
                    key=lambda trial: (
                        min(
                            self._distance(self.candidates[trial], self.candidates[chosen])
                            for chosen in selected
                        ),
                        -trial,
                    ),
                )
            )
        return tuple(selected)

    def propose(
        self,
        observations: Sequence[CandidateObservation],
        *,
        selected: Sequence[int],
        pending: Sequence[int] = (),
        pending_fidelity: Mapping[int, float] | None = None,
        censored: Sequence[int] = (),
        survival: Mapping[int, SurvivalEstimate] | None = None,
        count: int,
    ) -> tuple[int, ...]:
        """Select the next result-dependent batch without exposing unselected candidates."""
        selected_set = set(selected)
        available = [trial for trial in sorted(self.candidates) if trial not in selected_set]
        if count < 1 or not available:
            return ()
        observed = [value for value in observations if value.trial in self.candidates]
        censored_trials = [trial for trial in censored if trial in self.candidates]
        if not observed:
            return tuple(available[:count])
        values = [value.value for value in observed]
        center = sum(values) / len(values)
        scale = max(max(values) - min(values), 1e-12)
        proposed: list[int] = []
        while available and len(proposed) < count:

            def acquisition(trial: int) -> tuple[float, float, int]:
                parameters = self.candidates[trial]
                neighbours = sorted(
                    (
                        (
                            math.sqrt(
                                (
                                    self._distance(parameters, self.candidates[item.trial]) ** 2
                                    + (1.0 - min(1.0, max(0.0, item.fidelity))) ** 2
                                )
                                / 2.0
                            ),
                            item.value,
                            min(1.0, max(0.05, item.fidelity)),
                        )
                        for item in observed
                    ),
                    key=lambda item: item[0],
                )[: min(5, len(observed))]
                weights = [
                    fidelity / max(distance, 1e-6) for distance, _value, fidelity in neighbours
                ]
                prediction = sum(
                    weight * value
                    for weight, (_distance, value, _fidelity) in zip(
                        weights, neighbours, strict=True
                    )
                ) / sum(weights)
                exploitation = (prediction - center) / scale
                if self.mode == "min":
                    exploitation = -exploitation
                references = (
                    [(item.trial, min(1.0, max(0.0, item.fidelity))) for item in observed]
                    + [
                        (
                            reference,
                            min(
                                1.0,
                                max(
                                    0.0,
                                    float((pending_fidelity or {}).get(reference, 1.0)),
                                ),
                            ),
                        )
                        for reference in pending
                    ]
                    + [(reference, 1.0) for reference in [*censored_trials, *proposed]]
                )
                exploration = min(
                    math.sqrt(
                        (
                            self._distance(parameters, self.candidates[reference]) ** 2
                            + (1.0 - fidelity) ** 2
                        )
                        / 2.0
                    )
                    for reference, fidelity in references
                )
                censored_penalty = (
                    0.2
                    * (
                        1
                        - min(
                            self._distance(parameters, self.candidates[reference])
                            for reference in censored_trials
                        )
                    )
                    if censored_trials
                    else 0.0
                )
                survival_adjustment = 0.0
                if survival is not None and trial in survival:
                    estimate = survival[trial]
                    uncertainty = estimate.upper - estimate.lower
                    survival_adjustment = 0.25 * (estimate.probability - 0.5) + 0.1 * uncertainty
                # Exploitation dominates once evidence exists, while the distance bonus keeps
                # unexplored regions alive and avoids proposing near-duplicates in one batch.
                # A pruned-only neighbour is censored negative evidence: discourage its immediate
                # vicinity mildly, but never invent an exact full-budget objective for it.
                return (
                    exploitation
                    + 0.35 * exploration
                    - (0.0 if survival is not None else censored_penalty)
                    + survival_adjustment,
                    exploration,
                    -trial,
                )

            chosen = max(available, key=acquisition)
            proposed.append(chosen)
            available.remove(chosen)
        return tuple(proposed)

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


__all__ = ["AdaptiveSampler", "CandidateObservation"]
