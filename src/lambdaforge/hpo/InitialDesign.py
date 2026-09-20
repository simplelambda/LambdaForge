"""Deterministic scientific initial design and bounded coverage accounting.

The authored candidate pool is immutable.  This module decides which evidence should exist; it
does not decide when a process can run.  Resource admission therefore never changes the selected
anchors or their scientific obligations.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from lambdaforge.hpo.ParameterSpace import INACTIVE, ParameterSpace

AnchorState = Literal[
    "RESERVED",
    "WAITING_FOR_RESOURCES",
    "DISPATCHED",
    "OBSERVED",
    "CENSORED",
    "INFEASIBLE",
    "REPLACED",
]


@dataclass(frozen=True, slots=True)
class CoverageObligation:
    """One authored support condition that an initial anchor should represent."""

    key: str
    kind: str
    target: Mapping[str, Any]
    candidate_trials: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "target": dict(self.target),
            "candidate_trials": list(self.candidate_trials),
        }


@dataclass(frozen=True, slots=True)
class InitialDesignAnchor:
    """One protected candidate and the obligations it helps satisfy."""

    trial: int
    obligations: tuple[str, ...]
    selection_reason: str

    def to_dict(self, *, state: AnchorState = "RESERVED") -> dict[str, Any]:
        return {
            "pool_trial": self.trial,
            "state": state,
            "obligations": list(self.obligations),
            "selection_reason": self.selection_reason,
        }


@dataclass(frozen=True, slots=True)
class InitialDesignPlan:
    """Geometry-derived initial evidence plan, independent of execution parallelism."""

    mode: Literal["automatic", "explicit"]
    authored_dimensions: int
    feature_count: int
    full_rank: int
    effective_rank: int
    anchors: tuple[InitialDesignAnchor, ...]
    obligations: tuple[CoverageObligation, ...]
    uncovered_obligations: tuple[str, ...]

    @property
    def trials(self) -> tuple[int, ...]:
        return tuple(anchor.trial for anchor in self.anchors)

    def to_dict(self, states: Mapping[int, AnchorState] | None = None) -> dict[str, Any]:
        states = states or {}
        observed = sum(states.get(anchor.trial) == "OBSERVED" for anchor in self.anchors)
        censored = sum(states.get(anchor.trial) == "CENSORED" for anchor in self.anchors)
        pending = (
            len(self.anchors)
            - observed
            - censored
            - sum(states.get(anchor.trial) in {"INFEASIBLE", "REPLACED"} for anchor in self.anchors)
        )
        return {
            "initial_design_version": 1,
            "mode": self.mode,
            "authored_dimensions": self.authored_dimensions,
            "feature_count": self.feature_count,
            "full_rank": self.full_rank,
            "effective_rank": self.effective_rank,
            "required_anchors": len(self.anchors),
            "anchors_observed": observed,
            "anchors_pending": max(0, pending),
            "anchors_censored": censored,
            "anchors": [
                anchor.to_dict(state=states.get(anchor.trial, "RESERVED"))
                for anchor in self.anchors
            ],
            "coverage_obligations": [value.to_dict() for value in self.obligations],
            "uncovered_obligations": list(self.uncovered_obligations),
        }

    def replacement(
        self,
        trial: int,
        *,
        candidates: Mapping[int, Mapping[str, Any]],
        parameter_space: ParameterSpace,
        unavailable: Sequence[int] = (),
        obligation_keys: Sequence[str] | None = None,
    ) -> int | None:
        """Choose a feasible substitute preserving the failed anchor's obligations."""
        anchor = next((value for value in self.anchors if value.trial == trial), None)
        required = (
            tuple(obligation_keys)
            if obligation_keys is not None
            else (anchor.obligations if anchor is not None else ())
        )
        if not required:
            return None
        obligations = {value.key: value for value in self.obligations}
        occupied = set(self.trials) | set(unavailable)
        available = sorted(set(candidates) - occupied)
        if not available:
            return None
        selected = [candidates[value] for value in self.trials if value != trial]

        def score(candidate: int) -> tuple[int, float, int]:
            preserved = sum(
                candidate in obligations[key].candidate_trials
                for key in required
                if key in obligations
            )
            separation = min(
                (
                    parameter_space.distance(candidates[candidate], existing)
                    for existing in selected
                ),
                default=1.0,
            )
            return preserved, separation, -candidate

        replacement = max(available, key=score)
        return replacement if score(replacement)[0] > 0 else None


class InitialDesignPlanner:
    """Build a bounded rank/coverage design using deterministic greedy D-optimal selection."""

    @classmethod
    def plan(
        cls,
        candidates: Mapping[int, Mapping[str, Any]],
        parameter_space: ParameterSpace | Mapping[str, Any],
        *,
        candidate_budget: int,
        startup_trials: int | None = None,
    ) -> InitialDesignPlan:
        if candidate_budget < 1 or not candidates:
            raise ValueError("Initial design requires a positive candidate budget and pool.")
        space = (
            parameter_space
            if isinstance(parameter_space, ParameterSpace)
            else ParameterSpace.from_schema(parameter_space, tuple(candidates.values()))
        )
        ordered = tuple(sorted(candidates))
        matrix = cls._design_matrix(candidates, space, ordered)
        full_rank = int(np.linalg.matrix_rank(matrix))
        obligations = cls._obligations(candidates, space)
        selectable = min(candidate_budget, len(ordered))
        explicit = startup_trials is not None
        target = min(selectable, startup_trials) if startup_trials is not None else selectable
        selected: list[int] = []
        covered: set[str] = set()
        current_rank = 0
        reasons: dict[int, str] = {}
        row_by_trial = {trial: matrix[index] for index, trial in enumerate(ordered)}
        obligation_by_trial = {
            trial: {
                obligation.key for obligation in obligations if trial in obligation.candidate_trials
            }
            for trial in ordered
        }
        while len(selected) < target:
            best: tuple[tuple[float, ...], int, int, int, float] | None = None
            best_trial: int | None = None
            for trial in ordered:
                if trial in selected:
                    continue
                candidate_rows = np.asarray([row_by_trial[value] for value in (*selected, trial)])
                rank = int(np.linalg.matrix_rank(candidate_rows))
                rank_gain = rank - current_rank
                coverage_gain = len(obligation_by_trial[trial] - covered)
                information = cls._log_information(candidate_rows)
                separation = min(
                    (space.distance(candidates[trial], candidates[value]) for value in selected),
                    default=1.0,
                )
                score = (
                    (float(rank_gain), float(coverage_gain), information, separation),
                    -trial,
                    rank,
                    coverage_gain,
                    information,
                )
                if best is None or score > best:
                    best, best_trial = score, trial
            if best_trial is None or best is None:
                break
            _score, _tie, new_rank, coverage_gain, _information = best
            selected.append(best_trial)
            current_rank = new_rank
            covered.update(obligation_by_trial[best_trial])
            reasons[best_trial] = (
                f"rank gain to {current_rank}/{full_rank}; "
                f"{coverage_gain} new authored support obligation(s); "
                "deterministic D-optimal/maximin tie-break"
            )
            if (
                not explicit
                and current_rank >= full_rank
                and all(obligation.key in covered for obligation in obligations)
            ):
                break
        uncovered = tuple(value.key for value in obligations if value.key not in covered)
        anchors = tuple(
            InitialDesignAnchor(
                trial,
                tuple(sorted(obligation_by_trial[trial])),
                reasons[trial],
            )
            for trial in selected
        )
        return InitialDesignPlan(
            "explicit" if explicit else "automatic",
            len(space.descriptors),
            int(matrix.shape[1]),
            full_rank,
            current_rank,
            anchors,
            obligations,
            uncovered,
        )

    @staticmethod
    def opportunistic(
        candidates: Mapping[int, Mapping[str, Any]],
        parameter_space: ParameterSpace,
        *,
        reference: Sequence[int],
        count: int,
        excluded: Sequence[int] = (),
    ) -> tuple[int, ...]:
        """Fill otherwise idle startup capacity with non-protected maximin candidates."""
        selected: list[int] = []
        occupied = set(reference) | set(excluded)
        available = sorted(set(candidates) - occupied)
        while available and len(selected) < max(0, count):
            context = [*reference, *selected]
            choice = max(
                available,
                key=lambda trial: (
                    min(
                        (
                            parameter_space.distance(candidates[trial], candidates[value])
                            for value in context
                        ),
                        default=1.0,
                    ),
                    -trial,
                ),
            )
            selected.append(choice)
            available.remove(choice)
        return tuple(selected)

    @staticmethod
    def _design_matrix(
        candidates: Mapping[int, Mapping[str, Any]],
        space: ParameterSpace,
        ordered: Sequence[int],
    ) -> np.ndarray:
        columns: list[list[float]] = [[1.0 for _ in ordered]]
        for descriptor in space.descriptors:
            active = [
                descriptor.active(candidates[trial]) and descriptor.name in candidates[trial]
                for trial in ordered
            ]
            if descriptor.conditional:
                columns.append([float(value) for value in active])
            if descriptor.kind in {"continuous", "integer"}:
                coordinates = [
                    descriptor.normalize(candidates[trial][descriptor.name]) if is_active else 0.0
                    for trial, is_active in zip(ordered, active, strict=True)
                ]
                columns.append(coordinates)
                distinct = {
                    round(value, 12)
                    for value, is_active in zip(coordinates, active, strict=True)
                    if is_active
                }
                if len(distinct) >= 3:
                    columns.append([value * value for value in coordinates])
            else:
                levels = tuple(
                    value
                    for value in descriptor.values
                    if any(
                        is_active and candidates[trial][descriptor.name] == value
                        for trial, is_active in zip(ordered, active, strict=True)
                    )
                )
                for level in levels[1:]:
                    columns.append(
                        [
                            float(is_active and candidates[trial][descriptor.name] == level)
                            for trial, is_active in zip(ordered, active, strict=True)
                        ]
                    )
        return np.asarray(columns, dtype=float).T

    @classmethod
    def _obligations(
        cls, candidates: Mapping[int, Mapping[str, Any]], space: ParameterSpace
    ) -> tuple[CoverageObligation, ...]:
        output: list[CoverageObligation] = []
        for descriptor in space.descriptors:
            active_trials = tuple(
                trial
                for trial, point in sorted(candidates.items())
                if descriptor.active(point) and descriptor.name in point
            )
            inactive_trials = tuple(
                trial
                for trial, point in sorted(candidates.items())
                if not (descriptor.active(point) and descriptor.name in point)
            )
            if descriptor.conditional:
                if inactive_trials:
                    output.append(
                        CoverageObligation(
                            f"{descriptor.name}:inactive",
                            "conditional_activity",
                            {descriptor.name: INACTIVE},
                            inactive_trials,
                        )
                    )
                if active_trials:
                    output.append(
                        CoverageObligation(
                            f"{descriptor.name}:active",
                            "conditional_activity",
                            {descriptor.name: "<active>"},
                            active_trials,
                        )
                    )
            if descriptor.kind in {"categorical", "boolean"}:
                for level in descriptor.values:
                    members = tuple(
                        trial
                        for trial in active_trials
                        if candidates[trial][descriptor.name] == level
                    )
                    if members:
                        output.append(
                            CoverageObligation(
                                f"{descriptor.name}:value:{level!r}",
                                "parameter_value",
                                {descriptor.name: level},
                                members,
                            )
                        )
            else:
                coordinates = {
                    trial: descriptor.normalize(candidates[trial][descriptor.name])
                    for trial in active_trials
                }
                distinct = sorted(set(coordinates.values()))
                targets = (
                    (("low", distinct[0]), ("high", distinct[-1]))
                    if len(distinct) == 2
                    else (
                        ("low", distinct[0]),
                        ("interior", min(distinct[1:-1], key=lambda value: abs(value - 0.5))),
                        ("high", distinct[-1]),
                    )
                    if len(distinct) >= 3
                    else (("support", distinct[0]),)
                    if distinct
                    else ()
                )
                for region, coordinate in targets:
                    members = tuple(
                        trial for trial, value in coordinates.items() if value == coordinate
                    )
                    output.append(
                        CoverageObligation(
                            f"{descriptor.name}:region:{region}",
                            "numeric_region",
                            {descriptor.name: region},
                            members,
                        )
                    )
        return tuple(output)

    @staticmethod
    def _log_information(matrix: np.ndarray) -> float:
        gram = matrix.T @ matrix
        regularized = gram + np.eye(gram.shape[0]) * 1e-9
        sign, value = np.linalg.slogdet(regularized)
        return float(value) if sign > 0 else float("-inf")


class CoverageState:
    """Bounded two-level search/response coverage derived from authoritative Run states."""

    @classmethod
    def summarize(
        cls,
        plan: InitialDesignPlan,
        candidates: Mapping[int, Mapping[str, Any]],
        states: Mapping[int, str],
        parameter_space: ParameterSpace,
        *,
        relevant_interactions: Sequence[Sequence[str]] = (),
    ) -> dict[str, Any]:
        values: list[dict[str, Any]] = []
        for obligation in plan.obligations:
            state_counts = {
                "never_attempted": 0,
                "pending": 0,
                "active": 0,
                "censored_pruned": 0,
                "completed": 0,
            }
            attempted: list[int] = []
            complete: list[int] = []
            for trial in obligation.candidate_trials:
                state = states.get(trial, "never_attempted")
                bucket = cls._bucket(state)
                state_counts[bucket] += 1
                if bucket != "never_attempted":
                    attempted.append(trial)
                if bucket == "completed":
                    complete.append(trial)
            parameter = next(iter(obligation.target), "")
            contexts = [candidates[trial] for trial in attempted if trial in candidates]
            context_diversity = cls._context_diversity(
                contexts, parameter_space, ignore=(parameter,)
            )
            values.append(
                {
                    "key": obligation.key,
                    "kind": obligation.kind,
                    "target": dict(obligation.target),
                    "authored_support": len(obligation.candidate_trials),
                    **state_counts,
                    "search_coverage": bool(attempted),
                    "response_coverage": bool(complete),
                    "matched_context_diversity": context_diversity,
                    "response_uncertainty": (
                        1.0 / math.sqrt(len(complete) * max(context_diversity, 1e-6))
                        if complete
                        else None
                    ),
                }
            )
        return {
            "coverage_state_version": 1,
            "search_coverage": sum(bool(value["search_coverage"]) for value in values),
            "response_coverage": sum(bool(value["response_coverage"]) for value in values),
            "obligation_count": len(values),
            "obligations": values[:128],
            "interactions": cls._interaction_summary(
                candidates,
                states,
                parameter_space,
                relevant_interactions=relevant_interactions,
            ),
            "bounded": len(values) > 128,
        }

    @classmethod
    def _interaction_summary(
        cls,
        candidates: Mapping[int, Mapping[str, Any]],
        states: Mapping[int, str],
        parameter_space: ParameterSpace,
        *,
        relevant_interactions: Sequence[Sequence[str]],
    ) -> list[dict[str, Any]]:
        """Summarize authored pair cells without persisting a factorial matrix."""
        descriptors = parameter_space.descriptors
        by_name = {descriptor.name: descriptor for descriptor in descriptors}
        requested: list[tuple[Any, Any]] = []
        seen: set[tuple[str, str]] = set()
        for names in relevant_interactions:
            normalized = tuple(str(value) for value in names[:2])
            if len(normalized) != 2 or normalized[0] == normalized[1]:
                continue
            first, second = sorted(normalized)
            key = (first, second)
            if key in seen or any(name not in by_name for name in key):
                continue
            seen.add(key)
            requested.append((by_name[key[0]], by_name[key[1]]))
        pairs = requested or [
            (left, right)
            for left_index, left in enumerate(descriptors)
            for right in descriptors[left_index + 1 :]
        ]
        output: list[dict[str, Any]] = []
        for left, right in pairs[:32]:
            cells: dict[tuple[str, str], list[int]] = {}
            for trial, point in sorted(candidates.items()):
                left_value = point.get(left.name, INACTIVE)
                right_value = point.get(right.name, INACTIVE)
                key = (repr(left_value), repr(right_value))
                cells.setdefault(key, []).append(trial)
            if len(cells) < 2:
                continue
            search_cells = response_cells = censored_cells = 0
            diversities: list[float] = []
            for trials in cells.values():
                buckets = [cls._bucket(states.get(trial, "never_attempted")) for trial in trials]
                attempted = [
                    candidates[trial]
                    for trial, bucket in zip(trials, buckets, strict=True)
                    if bucket != "never_attempted"
                ]
                if attempted:
                    search_cells += 1
                    diversities.append(
                        cls._context_diversity(
                            attempted,
                            parameter_space,
                            ignore=(left.name, right.name),
                        )
                    )
                if "completed" in buckets:
                    response_cells += 1
                elif "censored_pruned" in buckets:
                    censored_cells += 1
            output.append(
                {
                    "parameters": [left.name, right.name],
                    "authored_cells": len(cells),
                    "observed_cells": search_cells,
                    "response_cells": response_cells,
                    "censored_cells": censored_cells,
                    "unexplored_cells": len(cells) - search_cells,
                    "matched_context_quality": (
                        float(np.mean(diversities)) if diversities else 0.0
                    ),
                }
            )
        return output

    @staticmethod
    def _bucket(state: str) -> str:
        normalized = state.lower()
        if normalized in {"succeeded", "completed", "observed"}:
            return "completed"
        if normalized in {"pruned", "censored"}:
            return "censored_pruned"
        if normalized in {"running", "active", "dispatched"}:
            return "active"
        if normalized in {"pending", "queued", "reserved", "waiting_for_resources"}:
            return "pending"
        return "never_attempted"

    @staticmethod
    def _context_diversity(
        contexts: Sequence[Mapping[str, Any]],
        parameter_space: ParameterSpace,
        *,
        ignore: Sequence[str],
    ) -> float:
        if len(contexts) < 2:
            return 0.0
        # Coverage is a bounded controller read model.  A deterministic evenly-spaced sample
        # preserves broad context support without turning a large proposal pool into an O(N²)
        # analysis on every terminal event.
        if len(contexts) > 32:
            indices = np.linspace(0, len(contexts) - 1, num=32, dtype=int)
            contexts = tuple(contexts[int(index)] for index in indices)
        nearest = [
            min(
                parameter_space.distance(left, right, ignore=ignore)
                for other_index, right in enumerate(contexts)
                if other_index != index
            )
            for index, left in enumerate(contexts)
        ]
        return min(1.0, max(0.0, float(np.mean(nearest))))


__all__ = [
    "AnchorState",
    "CoverageObligation",
    "CoverageState",
    "InitialDesignAnchor",
    "InitialDesignPlan",
    "InitialDesignPlanner",
]
