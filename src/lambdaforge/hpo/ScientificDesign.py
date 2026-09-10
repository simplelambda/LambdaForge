"""Scientific-question inference and active experimental design for adaptive HPO.

The controller and post-hoc analysis intentionally share this module.  It provides a bounded,
deterministic empirical-Bayes approximation; it does not claim that its resampling frequencies are
frequentist confidence intervals or that predictive associations are causal effects.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from threading import Lock
from typing import Any, ClassVar

_INACTIVE = "<inactive>"
_PARAMETER_STATES = (
    "PREFERRED",
    "PREFERRED_REGION",
    "PRACTICALLY_EQUIVALENT",
    "WEAK_PREFERENCE",
    "FLAT",
    "CONTEXT_DEPENDENT",
    "NO_CLEAR_PREFERENCE",
    "UNRESOLVED",
)
_INTERACTION_STATES = ("MATERIAL_INTERACTION", "WEAK_INTERACTION", "ADDITIVE", "UNRESOLVED")


@dataclass(frozen=True, slots=True)
class SeedNoiseEstimate:
    """Hierarchical seed-noise estimate that never uses between-candidate spread as noise."""

    variance: float | None
    shared_seed_effects: Mapping[int, float]
    repeated_candidates: int
    residual_degrees_of_freedom: int
    shared_seed_comparisons: int

    @property
    def calibrated(self) -> bool:
        return self.variance is not None and self.residual_degrees_of_freedom > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "calibrated" if self.calibrated else "unresolved",
            "variance": self.variance,
            "standard_deviation": math.sqrt(self.variance) if self.variance is not None else None,
            "shared_seed_effects": {
                str(key): value for key, value in self.shared_seed_effects.items()
            },
            "repeated_candidates": self.repeated_candidates,
            "residual_degrees_of_freedom": self.residual_degrees_of_freedom,
            "shared_seed_comparisons": self.shared_seed_comparisons,
            "method": "pooled-within-candidate-residuals-with-shared-seed-effects",
            "between_candidate_spread_used_as_seed_noise": False,
        }


@dataclass(frozen=True, slots=True)
class CandidateDesignValue:
    """Auditable value of running one currently unobserved candidate."""

    trial: int
    purpose: str
    target_questions: tuple[str, ...]
    optimization_value: float
    information_value: float
    counterfactual_match_quality: float
    expected_cost: float
    cost_ratio: float
    score: float
    predicted_objective: float
    prediction_standard_deviation: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial": self.trial,
            "action": "START_NEW" if self.purpose == "OPTIMIZE" else "DESIGNED_PROBE",
            "purpose": self.purpose,
            "target_questions": list(self.target_questions),
            "optimization_value": self.optimization_value,
            "information_value": self.information_value,
            "counterfactual_match_quality": self.counterfactual_match_quality,
            "expected_cost_seconds": self.expected_cost,
            "cost_ratio": self.cost_ratio,
            "final_score": self.score,
            "predicted_objective": self.predicted_objective,
            "prediction_standard_deviation": self.prediction_standard_deviation,
            "reason": self.reason,
            "value_method": (
                "posterior-practical-improvement-plus-question-entropy-reduction-per-cost"
            ),
        }


class _ScientificDesignBasis:
    """Reusable bounded matching geometry for one immutable proposal pool.

    Counterfactual matching depends on the authored pool, not on sampled objective values.  The
    old implementation nevertheless repeated every nearest-neighbour scan for every Monte Carlo
    realization.  This basis computes those matches once, retains exact matches when the finite
    design contains them and otherwise uses a deterministic, evenly spread support coreset.
    """

    def __init__(
        self,
        pool: Mapping[int, Mapping[str, Any]],
        *,
        reference_budget: int,
    ) -> None:
        self.pool = tuple(
            (int(trial), dict(parameters)) for trial, parameters in sorted(pool.items())
        )
        self.references = self._sample(
            tuple(parameters for _trial, parameters in self.pool),
            max(1, reference_budget),
        )
        # Match against sqrt(R) representatives for R Monte Carlo reference points.  Scientific
        # live cost is therefore bounded by its inference precision rather than by proposal-pool
        # cardinality; terminal analysis naturally uses a denser support.
        # A realization ensemble supplies uncertainty; repeating many near-identical members in
        # every factorial cell only multiplies latency. Live analysis uses at most four matched
        # representatives and two per interaction cell; denser final analysis remains bounded.
        root_budget = math.ceil(math.sqrt(max(1, reference_budget)))
        self.support_budget = max(2, min(16, math.ceil(root_budget / 2)))
        self.cell_budget = max(1, min(8, math.ceil(root_budget / 4)))
        self._counterfactual_cache: dict[
            tuple[str, str], tuple[tuple[str, dict[str, Any], int], ...]
        ] = {}
        self._factorial_cache: dict[
            tuple[str, str, str, str], tuple[dict[str, Any], ...]
        ] = {}
        self._lock = Lock()

    def counterfactual_matches(
        self, name: str, level: Any
    ) -> tuple[tuple[str, dict[str, Any], int], ...]:
        """Return weighted matched authored points for one parameter level."""
        cache_key = (name, _label(level))
        with self._lock:
            cached = self._counterfactual_cache.get(cache_key)
        if cached is not None:
            return cached
        members = tuple(
            parameters
            for _trial, parameters in self.pool
            if parameters.get(name, _INACTIVE) == level
        )
        support = self._sample(members, self.support_budget)
        if not support:
            return ()
        # Prefer a genuine authored counterfactual whenever the finite design contains one.  The
        # smaller support sample is only the bounded fallback for references whose remaining
        # coordinates have no exact counterpart at this level.
        exact = {_projected_key(parameters, ignore=(name,)): parameters for parameters in members}
        counts: dict[str, tuple[dict[str, Any], int]] = {}
        for reference in self.references:
            matched = exact.get(_projected_key(reference, ignore=(name,)))
            if matched is None:
                matched = min(
                    support,
                    key=lambda parameters: _mixed_distance(
                        reference, parameters, ignore=(name,)
                    ),
                )
            encoded = json.dumps(matched, sort_keys=True, separators=(",", ":"), default=str)
            previous = counts.get(encoded)
            counts[encoded] = (matched, 1 if previous is None else previous[1] + 1)
        result = tuple((key, *counts[key]) for key in sorted(counts))
        with self._lock:
            return self._counterfactual_cache.setdefault(cache_key, result)

    def factorial_members(
        self,
        left: str,
        right: str,
        left_value: Any,
        right_value: Any,
        left_levels: Sequence[Any],
        right_levels: Sequence[Any],
    ) -> tuple[dict[str, Any], ...]:
        """Return bounded authored members of one valid endpoint factorial cell."""
        cache_key = (left, right, _label(left_value), _label(right_value))
        with self._lock:
            cached = self._factorial_cache.get(cache_key)
        if cached is not None:
            return cached
        left_numeric = all(_finite(value) for value in left_levels)
        right_numeric = all(_finite(value) for value in right_levels)
        members = tuple(
            parameters
            for _trial, parameters in self.pool
            if _endpoint_level(
                parameters.get(left, _INACTIVE), left_levels, numeric=left_numeric
            )
            == left_value
            and _endpoint_level(
                parameters.get(right, _INACTIVE), right_levels, numeric=right_numeric
            )
            == right_value
        )
        result = self._sample(members, self.cell_budget)
        with self._lock:
            return self._factorial_cache.setdefault(cache_key, result)

    @staticmethod
    def _sample(
        values: Sequence[dict[str, Any]], limit: int
    ) -> tuple[dict[str, Any], ...]:
        """Select deterministic evenly spaced entries without relying on pool size knobs."""
        if len(values) <= limit:
            return tuple(values)
        if limit <= 1:
            return (values[len(values) // 2],)
        indexes = sorted(
            {round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)}
        )
        return tuple(values[index] for index in indexes)


class SeedNoiseModel:
    """Fit ``y[c,s] = f(x_c) + b_s + epsilon[c,s]`` by pooled demeaning."""

    @staticmethod
    def fit(outcomes: Mapping[int, Mapping[int | None, float]]) -> SeedNoiseEstimate:
        repeated = {
            int(trial): {seed: float(value) for seed, value in values.items()}
            for trial, values in outcomes.items()
            if len(values) >= 2
        }
        if not repeated:
            return SeedNoiseEstimate(None, {}, 0, 0, 0)
        candidate_means = {
            trial: statistics.fmean(values.values()) for trial, values in repeated.items()
        }
        residuals_by_seed: dict[int, list[float]] = {}
        for trial, values in repeated.items():
            for seed, value in values.items():
                if seed is not None:
                    residuals_by_seed.setdefault(seed, []).append(value - candidate_means[trial])
        shared_effects = {
            seed: statistics.fmean(residuals)
            for seed, residuals in residuals_by_seed.items()
            if len(residuals) >= 2
        }
        residuals: list[float] = []
        degrees = 0
        for _trial, values in repeated.items():
            adjusted = [
                value - (shared_effects.get(seed, 0.0) if seed is not None else 0.0)
                for seed, value in values.items()
            ]
            center = statistics.fmean(adjusted)
            residuals.extend(value - center for value in adjusted)
            degrees += len(adjusted) - 1
        variance = sum(value * value for value in residuals) / degrees if degrees > 0 else None
        return SeedNoiseEstimate(
            max(variance, 0.0) if variance is not None else None,
            shared_effects,
            len(repeated),
            degrees,
            sum(len(values) for values in residuals_by_seed.values() if len(values) >= 2),
        )


class ScientificQuestionAnalyzer:
    """Infer practical regions and stable parameter/interaction conclusions.

    ``confidence`` always means agreement of the displayed qualitative conclusion across
    plausible evidence realizations.  A flat conclusion can therefore be highly confident.
    """

    # Live controller inference must remain cheaper than one scheduling heartbeat.  Thirty-two
    # deterministic realizations give 1/32 probability resolution; terminal analysis may spend a
    # much larger budget without delaying collection of finished Run processes.
    LIVE_MAX_RESAMPLES = 32
    FINAL_MAX_RESAMPLES = 256
    MIN_RESAMPLES = 64
    RESAMPLE_BATCH = 32
    _CACHE_LIMIT = 8
    _BASIS_CACHE_LIMIT = 4
    _cache: ClassVar[OrderedDict[str, dict[str, Any]]] = OrderedDict()
    _basis_cache: ClassVar[OrderedDict[str, _ScientificDesignBasis]] = OrderedDict()
    _cache_lock: ClassVar[Lock] = Lock()

    @classmethod
    def analyze(
        cls,
        candidates: Sequence[Mapping[str, Any]],
        objective: Mapping[str, Any],
        *,
        practical_margin: float | None,
        fingerprint: str,
        candidate_pool: Mapping[int, Mapping[str, Any]] | None = None,
        final: bool = False,
    ) -> dict[str, Any]:
        """Return one versioned scientific-understanding snapshot."""
        mode = str(objective.get("mode", "max"))
        if mode not in {"min", "max"}:
            raise ValueError("Scientific question analysis requires objective mode min or max.")
        margin = (
            float(practical_margin)
            if practical_margin is not None and math.isfinite(float(practical_margin))
            else None
        )
        if margin is not None and margin < 0:
            raise ValueError("Practical equivalence margin must be non-negative.")
        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "candidates": candidates,
                    "objective": dict(objective),
                    "margin": margin,
                    "fingerprint": fingerprint,
                    "candidate_pool": candidate_pool,
                    "final": final,
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode()
        ).hexdigest()
        with cls._cache_lock:
            cached = cls._cache.get(cache_key)
            if cached is not None:
                cls._cache.move_to_end(cache_key)
                return deepcopy(cached)
        rows, outcomes, pruned = cls._evidence(candidates, mode=mode)
        parameters = {
            int(candidate.get("trial", 0)): dict(candidate.get("parameters", {}))
            for candidate in candidates
            if isinstance(candidate.get("parameters"), Mapping)
        }
        pool = {
            int(trial): dict(values) for trial, values in (candidate_pool or parameters).items()
        }
        for trial, values in parameters.items():
            pool.setdefault(trial, values)
        noise = SeedNoiseModel.fit(outcomes)
        natural_scale = cls._natural_scale(rows, objective)
        resamples = cls._resample_count(
            rows,
            len({name for value in pool.values() for name in value}),
            final=final,
        )
        basis = cls._design_basis(pool, reference_budget=resamples)
        distance_cache: dict[str, dict[int, float]] = {}
        distance_order_cache: dict[str, tuple[int, tuple[int, ...]]] = {}
        model = _MixedKnnModel(
            parameters,
            rows,
            noise=noise,
            natural_scale=natural_scale,
            distance_cache=distance_cache,
            distance_order_cache=distance_order_cache,
        )
        rng = random.Random(cls._seed(fingerprint, "scientific-questions"))
        realizations = cls._realizations(
            parameters,
            outcomes,
            rows,
            noise=noise,
            fallback_sigma=max(model.validation_error, natural_scale / math.sqrt(12.0)),
            count=resamples,
            rng=rng,
        )
        # One predictive model belongs to one evidence realization.  Reusing it across every
        # parameter and interaction is essential: constructing it inside each question repeated
        # leave-one-out validation thousands of times and could starve the scheduler loop.
        realization_models = tuple(
            (
                realization,
                _MixedKnnModel(
                    parameters,
                    realization,
                    noise=None,
                    natural_scale=natural_scale,
                    distance_cache=distance_cache,
                    distance_order_cache=distance_order_cache,
                    validation_error=model.validation_error,
                ),
            )
            for realization in realizations
        )
        names = sorted({str(name) for values in pool.values() for name in values})
        levels_by_name = {
            name: cls._levels([value[name] for value in pool.values() if name in value])
            for name in names
        }
        interactions = cls._interactions(
            names,
            parameters,
            realization_models,
            basis=basis,
            levels_by_name=levels_by_name,
            practical_margin=margin,
            natural_scale=natural_scale,
        )
        interaction_by_parameter: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
        for interaction in interactions:
            for name in interaction["parameters"]:
                interaction_by_parameter[str(name)].append(interaction)
        parameter_questions = [
            cls._parameter(
                name,
                parameters,
                pool,
                rows,
                outcomes,
                realization_models,
                basis=basis,
                levels=levels_by_name[name][0],
                numeric=levels_by_name[name][1],
                interactions=interaction_by_parameter[name],
                practical_margin=margin,
                pruned=pruned,
            )
            for name in names
        ]
        region = cls._practical_region(
            parameters,
            outcomes,
            rows,
            practical_margin=margin,
            mode=mode,
            noise=noise,
            natural_scale=natural_scale,
            realizations=realizations,
        )
        cls._attach_region_flexibility(parameter_questions, region, parameters)
        opportunity = cls._optimization_opportunity(
            basis.references,
            parameters,
            model,
            rows,
            mode=mode,
            practical_margin=margin,
            natural_scale=natural_scale,
        )
        questions = [*parameter_questions, *interactions]
        uncertainty = (
            statistics.fmean(float(value["entropy"]) for value in questions) if questions else 0.0
        )
        denominator = opportunity + uncertainty
        optimization_weight = opportunity / denominator if denominator > 0 else 0.5
        information_weight = uncertainty / denominator if denominator > 0 else 0.5
        phase_resolution = 1.0 / math.sqrt(max(1, len(questions) * resamples))
        phase = (
            "balanced"
            if abs(optimization_weight - information_weight) <= phase_resolution
            else "optimization-dominant"
            if optimization_weight > information_weight
            else "evidence-dominant"
        )
        unresolved = sorted(
            (
                {
                    "question": value["question"],
                    "kind": value["kind"],
                    "confidence": value["confidence"],
                    "entropy": value["entropy"],
                    "missing_evidence": value.get("missing_evidence", []),
                }
                for value in questions
                if value.get("conclusion_kind") in {"UNRESOLVED", "NO_CLEAR_PREFERENCE"}
                or float(value.get("entropy", 0.0)) > 0
            ),
            key=lambda value: (-float(value["entropy"]), str(value["question"])),
        )
        result = {
            "scientific_question_version": 1,
            "status": "available" if rows else "insufficient",
            "objective": dict(objective),
            "practical_margin": margin,
            "practical_margin_source": (
                "authored-equivalence-margin" if margin is not None else "not-authored"
            ),
            "natural_utility_scale": natural_scale,
            "response_scale": "maximized-objective-utility",
            "seed_noise_model": noise.to_dict(),
            "optimization_opportunity": opportunity,
            "optimization_opportunity_method": (
                "expected improvement over a deterministic bounded reference design; "
                "the optimization sampler still considers the complete proposal pool"
            ),
            "scientific_uncertainty": uncertainty,
            "optimization_weight": optimization_weight,
            "information_weight": information_weight,
            "phase": phase,
            "parameter_questions": parameter_questions,
            "interaction_questions": interactions,
            "unresolved_questions": unresolved,
            "practical_optimal_region": region,
            "evidence": {
                "completed_candidates": len(rows),
                "candidate_pool_size": len(pool),
                "resamples": resamples,
                "reference_points": len(basis.references),
                "matching_support_points": basis.support_budget,
                "live_cost_is_pool_bounded": not final,
                "resampling": (
                    "deterministic candidate-level delete-d/shared-seed-aware evidence ensemble"
                ),
                "confidence_semantics": (
                    "agreement of the displayed conclusion across plausible evidence realizations"
                ),
                "confidence_interval_semantics": "not-a-frequentist-confidence-interval",
            },
        }
        with cls._cache_lock:
            cls._cache[cache_key] = deepcopy(result)
            cls._cache.move_to_end(cache_key)
            while len(cls._cache) > cls._CACHE_LIMIT:
                cls._cache.popitem(last=False)
        return result

    @classmethod
    def _design_basis(
        cls,
        pool: Mapping[int, Mapping[str, Any]],
        *,
        reference_budget: int,
    ) -> _ScientificDesignBasis:
        """Reuse immutable proposal-pool geometry across event-driven evidence updates."""
        key = hashlib.sha256(
            json.dumps(
                {"pool": pool, "reference_budget": reference_budget},
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode()
        ).hexdigest()
        with cls._cache_lock:
            cached = cls._basis_cache.get(key)
            if cached is not None:
                cls._basis_cache.move_to_end(key)
                return cached
            basis = _ScientificDesignBasis(pool, reference_budget=reference_budget)
            cls._basis_cache[key] = basis
            cls._basis_cache.move_to_end(key)
            while len(cls._basis_cache) > cls._BASIS_CACHE_LIMIT:
                cls._basis_cache.popitem(last=False)
            return basis

    @staticmethod
    def _evidence(
        candidates: Sequence[Mapping[str, Any]], *, mode: str
    ) -> tuple[dict[int, float], dict[int, dict[int | None, float]], set[int]]:
        outcomes: dict[int, dict[int | None, float]] = {}
        pruned: set[int] = set()
        for candidate in candidates:
            trial = int(candidate.get("trial", 0))
            if candidate.get("state") == "pruned":
                pruned.add(trial)
            runs = [value for value in candidate.get("runs", ()) if isinstance(value, Mapping)]
            for run in runs:
                if run.get("phase") == "confirmation":
                    continue
                if run.get("state") == "pruned" or run.get("censored"):
                    pruned.add(trial)
                    continue
                fidelity = run.get("fidelity", {})
                if isinstance(fidelity, Mapping):
                    target, maximum = fidelity.get("target"), fidelity.get("maximum")
                    if isinstance(target, int | float) and isinstance(maximum, int | float):
                        if float(target) < float(maximum):
                            continue
                value = run.get("final_objective")
                if not _finite(value) and run.get("state") == "succeeded":
                    value = run.get("best_observed_objective", run.get("best_objective"))
                if _finite(value) and run.get("state") in {None, "succeeded", "completed"}:
                    assert isinstance(value, int | float)
                    seed = run.get("seed")
                    seed = (
                        int(seed) if isinstance(seed, int) and not isinstance(seed, bool) else None
                    )
                    direction = 1.0 if mode == "max" else -1.0
                    outcomes.setdefault(trial, {})[seed] = direction * float(value)
            if trial not in outcomes and _finite(candidate.get("mean")):
                direction = 1.0 if mode == "max" else -1.0
                outcomes[trial] = {None: direction * float(candidate["mean"])}
            elif trial not in outcomes and _finite(candidate.get("selection_objective")):
                direction = 1.0 if mode == "max" else -1.0
                outcomes[trial] = {None: direction * float(candidate["selection_objective"])}
        rows = {
            trial: statistics.fmean(values.values()) for trial, values in outcomes.items() if values
        }
        return rows, outcomes, pruned

    @classmethod
    def _realizations(
        cls,
        parameters: Mapping[int, Mapping[str, Any]],
        outcomes: Mapping[int, Mapping[int | None, float]],
        rows: Mapping[int, float],
        *,
        noise: SeedNoiseEstimate,
        fallback_sigma: float,
        count: int,
        rng: random.Random,
    ) -> list[dict[int, float]]:
        seed_labels = sorted(
            {seed for values in outcomes.values() for seed in values if seed is not None}
        )
        realizations: list[dict[int, float]] = []
        trials = tuple(sorted(outcomes))
        for _ in range(count):
            selected_seed = rng.choice(seed_labels) if seed_labels else None
            realization: dict[int, float] = {}
            # Candidate is the experimental unit.  Perturbing only seed values would make a
            # conclusion appear perfectly stable whenever every candidate has the same single
            # seed.  A bounded delete-d ensemble also asks whether the displayed conclusion
            # survives plausible changes in the observed candidate sample.  Keep at least two
            # candidates so sparse live studies still produce a descriptive view.
            included = [trial for trial in trials if len(trials) <= 2 or rng.random() < 0.8]
            if len(included) < min(2, len(trials)):
                included = list(rng.sample(trials, k=min(2, len(trials))))
            for trial in included:
                values = outcomes[trial]
                if selected_seed is not None and selected_seed in values:
                    sampled = values[selected_seed]
                elif len(values) > 1:
                    sampled = rng.choice(tuple(values.values()))
                else:
                    sampled = next(iter(values.values()))
                    sigma = (
                        math.sqrt(noise.variance) if noise.variance is not None else fallback_sigma
                    )
                    sampled += rng.gauss(0.0, sigma)
                realization[trial] = float(sampled)
            realizations.append(realization)
        return realizations

    @classmethod
    def _parameter(
        cls,
        name: str,
        parameters: Mapping[int, Mapping[str, Any]],
        pool: Mapping[int, Mapping[str, Any]],
        rows: Mapping[int, float],
        outcomes: Mapping[int, Mapping[int | None, float]],
        realization_models: Sequence[tuple[Mapping[int, float], _MixedKnnModel]],
        *,
        basis: _ScientificDesignBasis,
        levels: Sequence[Any],
        numeric: bool,
        interactions: Sequence[Mapping[str, Any]],
        practical_margin: float | None,
        pruned: set[int],
    ) -> dict[str, Any]:
        observed = {
            trial: values[name]
            for trial, values in parameters.items()
            if name in values and trial in rows
        }
        pruned_parameter_count = sum(
            trial in pruned for trial, values in parameters.items() if name in values
        )
        observed_inactive = sum(
            trial in rows and name not in values for trial, values in parameters.items()
        )
        authored_inactive = sum(name not in values for values in pool.values())
        if len(levels) < 2 or len(set(map(_label, observed.values()))) < 2:
            probabilities = cls._unresolved_distribution(_PARAMETER_STATES)
            return cls._parameter_result(
                name,
                numeric,
                levels,
                observed,
                probabilities,
                response=[],
                practical_margin=practical_margin,
                interactions=interactions,
                outcomes=outcomes,
                pruned=pruned,
                missing=["at least two observed parameter values in comparable completed Runs"],
                pruned_parameter_count=pruned_parameter_count,
                observed_inactive=observed_inactive,
                authored_inactive=authored_inactive,
            )
        response_realizations: list[list[tuple[Any, float]]] = []
        kinds: list[str] = []
        conclusion_tokens: list[str] = []
        best_labels: list[str] = []
        context_probability = max(
            (
                float(value.get("probabilities", {}).get("MATERIAL_INTERACTION", 0.0))
                for value in interactions
            ),
            default=0.0,
        )
        for realization, model in realization_models:
            response = cls._counterfactual_response(name, levels, basis, model)
            if len(response) < 2:
                kinds.append("UNRESOLVED")
                conclusion_tokens.append("UNRESOLVED")
                continue
            response_realizations.append(response)
            ordered = sorted(response, key=lambda item: item[1], reverse=True)
            best_labels.append(_label(ordered[0][0]))
            span = ordered[0][1] - ordered[-1][1]
            if (
                context_probability > 0
                and random_probability(realization, name) < context_probability
            ):
                kinds.append("CONTEXT_DEPENDENT")
                conclusion_tokens.append("CONTEXT_DEPENDENT")
            elif practical_margin is not None and span <= practical_margin:
                kinds.append("PRACTICALLY_EQUIVALENT")
                conclusion_tokens.append("PRACTICALLY_EQUIVALENT")
            elif numeric:
                close = [
                    value
                    for value, score in ordered
                    if ordered[0][1] - score <= (practical_margin or 0.0)
                ]
                kind = "PREFERRED_REGION" if len(close) > 1 else "PREFERRED"
                kinds.append(kind)
                conclusion_tokens.append(
                    f"{kind}:{'|'.join(sorted(_label(value) for value in close))}"
                )
            else:
                kinds.append("PREFERRED")
                conclusion_tokens.append(f"PREFERRED:{_label(ordered[0][0])}")
        probabilities = cls._probabilities(kinds, _PARAMETER_STATES)
        dominant = max(probabilities, key=lambda key: probabilities[key])
        token_distribution = {
            token: count / max(1, len(conclusion_tokens))
            for token, count in sorted(Counter(conclusion_tokens).items())
        }
        dominant_tokens = {
            token: probability
            for token, probability in token_distribution.items()
            if token.split(":", 1)[0] == dominant
        }
        confidence = max(dominant_tokens.values(), default=float(probabilities[dominant]))
        practical_equivalent = probabilities.get("PRACTICALLY_EQUIVALENT", 0.0)
        best_distribution = Counter(best_labels)
        response_summary = cls._summarize_responses(response_realizations, levels)
        response_means = sorted(
            (float(value["mean"]) for value in response_summary), reverse=True
        )
        nonzero_preference = (
            len(response_means) >= 2
            and not math.isclose(
                response_means[0],
                response_means[1],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )
        if dominant == "PRACTICALLY_EQUIVALENT" and best_distribution and nonzero_preference:
            total = sum(best_distribution.values())
            best_count = max(best_distribution.values())
            chance = 1.0 / max(1, len(levels))
            standard_error = math.sqrt(chance * (1.0 - chance) / total)
            if best_count / total - standard_error > chance:
                probabilities["WEAK_PREFERENCE"] = probabilities.pop("PRACTICALLY_EQUIVALENT")
                dominant = "WEAK_PREFERENCE"
                best_probability = best_count / max(1, total)
                confidence = practical_equivalent * best_probability
                transformed = {
                    token: probability
                    for token, probability in token_distribution.items()
                    if token != "PRACTICALLY_EQUIVALENT"
                }
                for label, count in sorted(best_distribution.items()):
                    token = f"WEAK_PREFERENCE:{label}"
                    transformed[token] = transformed.get(token, 0.0) + (
                        practical_equivalent * count / max(1, total)
                    )
                token_distribution = transformed
        result = cls._parameter_result(
            name,
            numeric,
            levels,
            observed,
            probabilities,
            response=response_summary,
            practical_margin=practical_margin,
            interactions=interactions,
            outcomes=outcomes,
            pruned=pruned,
            missing=[],
            confidence=confidence,
            conclusion_distribution=token_distribution,
            pruned_parameter_count=pruned_parameter_count,
            observed_inactive=observed_inactive,
            authored_inactive=authored_inactive,
        )
        result["practically_equivalent_probability"] = practical_equivalent
        result["best_value_probability"] = {
            _display_label(json.loads(label)): count / max(1, len(best_labels))
            for label, count in sorted(best_distribution.items())
        }
        result["conclusion_kind"] = dominant
        result["summary"] = cls._parameter_summary(result)
        return result

    @classmethod
    def _parameter_result(
        cls,
        name: str,
        numeric: bool,
        levels: Sequence[Any],
        observed: Mapping[int, Any],
        probabilities: Mapping[str, float],
        *,
        response: Sequence[Mapping[str, Any]],
        practical_margin: float | None,
        interactions: Sequence[Mapping[str, Any]],
        outcomes: Mapping[int, Mapping[int | None, float]],
        pruned: set[int],
        missing: Sequence[str],
        confidence: float | None = None,
        conclusion_distribution: Mapping[str, float] | None = None,
        pruned_parameter_count: int = 0,
        observed_inactive: int = 0,
        authored_inactive: int = 0,
    ) -> dict[str, Any]:
        dominant = max(probabilities, key=lambda key: probabilities[key])
        stable_confidence = (
            float(confidence) if confidence is not None else float(probabilities[dominant])
        )
        active_trials = set(observed)
        shared = cls._shared_seed_count({trial: outcomes.get(trial, {}) for trial in active_trials})
        del pruned
        return {
            "question_version": 1,
            "question": f"What can we conclude about {name} in the studied space?",
            "parameter": name,
            "kind": "numeric" if numeric else "categorical",
            "conclusion_kind": dominant,
            "summary": "The question remains unresolved."
            if dominant == "UNRESOLVED"
            else dominant.replace("_", " ").title(),
            "confidence": stable_confidence,
            "confidence_label": _confidence_label(stable_confidence),
            "probabilities": dict(probabilities),
            "conclusion_distribution": dict(conclusion_distribution or probabilities),
            # Question uncertainty concerns the exact displayed answer (including which value or
            # region is preferred), not merely the coarse conclusion family.  Otherwise two
            # realizations that both say PREFERRED but disagree on the value would incorrectly
            # contribute zero scientific uncertainty.
            "entropy": _normalized_entropy(conclusion_distribution or probabilities),
            "practical_margin": practical_margin,
            "authored_values": list(levels),
            "observed_values": sorted({_json_value(value) for value in observed.values()}, key=str),
            "response": list(response),
            "support": {
                "completed_candidates": len(active_trials),
                "values": dict(Counter(_display_label(value) for value in observed.values())),
                "shared_seed_comparisons": shared,
                "pruned_candidates": pruned_parameter_count,
                "inactive_completed_candidates": observed_inactive,
                "inactive_authored_candidates": authored_inactive,
            },
            "main_interactions": [
                {
                    "parameters": list(value.get("parameters", ())),
                    "kind": value.get("conclusion_kind"),
                    "confidence": value.get("confidence"),
                }
                for value in sorted(
                    interactions, key=lambda item: float(item.get("entropy", 0)), reverse=True
                )[:5]
            ],
            "missing_evidence": list(missing),
            "evidence": {
                "agreement": stable_confidence,
                "interpretation": "conclusion stability across evidence realizations",
                "coverage_is_not_confidence": True,
            },
        }

    @staticmethod
    def _parameter_summary(value: Mapping[str, Any]) -> str:
        name = str(value["parameter"])
        kind = str(value["conclusion_kind"])
        response = [item for item in value.get("response", ()) if isinstance(item, Mapping)]
        if kind == "CONTEXT_DEPENDENT":
            interactions = value.get("main_interactions", ())
            other = next(
                (
                    next((item for item in relation.get("parameters", ()) if item != name), None)
                    for relation in interactions
                    if isinstance(relation, Mapping)
                ),
                None,
            )
            return f"The effect of {name} depends on {other or 'other parameters'}."
        if kind in {"PRACTICALLY_EQUIVALENT", "FLAT"}:
            return f"No practically relevant {name} effect is supported in the studied range."
        if kind == "WEAK_PREFERENCE":
            best = max(response, key=lambda item: float(item.get("mean", -math.inf)), default={})
            return (
                f"{best.get('value', 'One value')} is slightly preferred, but alternatives "
                "are practically equivalent."
            )
        if kind in {"PREFERRED", "PREFERRED_REGION"}:
            ordered = sorted(
                response, key=lambda item: float(item.get("mean", -math.inf)), reverse=True
            )
            selected = [str(item.get("value")) for item in ordered if item.get("preferred")]
            region = (
                ", ".join(selected)
                if selected
                else str(ordered[0].get("value", "the supported optimum"))
            )
            return f"{name} is preferred around {region}."
        if kind == "NO_CLEAR_PREFERENCE":
            return f"No clear {name} preference is resolved; no practical margin was authored."
        return f"The current evidence does not resolve {name}."

    @classmethod
    def _interactions(
        cls,
        names: Sequence[str],
        parameters: Mapping[int, Mapping[str, Any]],
        realization_models: Sequence[tuple[Mapping[int, float], _MixedKnnModel]],
        *,
        basis: _ScientificDesignBasis,
        levels_by_name: Mapping[str, tuple[list[Any], bool]],
        practical_margin: float | None,
        natural_scale: float,
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                left_levels, left_numeric = levels_by_name[left]
                right_levels, right_numeric = levels_by_name[right]
                if len(left_levels) < 2 or len(right_levels) < 2:
                    continue
                states: list[str] = []
                magnitudes: list[float] = []
                reversals = 0
                supported = 0
                for realization, model in realization_models:
                    observed_cells = set()
                    for trial in realization:
                        if (
                            trial not in parameters
                            or left not in parameters[trial]
                            or right not in parameters[trial]
                        ):
                            continue
                        left_bin = _endpoint_level(
                            parameters[trial][left], left_levels, numeric=left_numeric
                        )
                        right_bin = _endpoint_level(
                            parameters[trial][right], right_levels, numeric=right_numeric
                        )
                        if left_bin is not None and right_bin is not None:
                            observed_cells.add((_label(left_bin), _label(right_bin)))
                    if len(observed_cells) < 4:
                        states.append("UNRESOLVED")
                        continue
                    cells = cls._factorial_cells(
                        left, right, left_levels, right_levels, basis, model
                    )
                    if len(cells) < 4:
                        states.append("UNRESOLVED")
                        continue
                    supported += 1
                    effects = []
                    for right_value in right_levels:
                        values = [
                            cells.get((_label(left_value), _label(right_value)))
                            for left_value in (left_levels[0], left_levels[-1])
                        ]
                        if all(value is not None for value in values):
                            first, last = values[0], values[-1]
                            assert first is not None and last is not None
                            effects.append(float(last) - float(first))
                    if len(effects) < 2:
                        states.append("UNRESOLVED")
                        continue
                    magnitude = max(effects) - min(effects)
                    magnitudes.append(abs(magnitude))
                    reversal = min(effects) < 0 < max(effects)
                    reversals += reversal
                    scale = max(
                        practical_margin or 0.0,
                        natural_scale,
                        model.validation_error,
                    )
                    if abs(magnitude) > scale:
                        states.append("MATERIAL_INTERACTION")
                    elif abs(magnitude) > 0:
                        states.append("WEAK_INTERACTION")
                    else:
                        states.append("ADDITIVE")
                probabilities = (
                    cls._probabilities(states, _INTERACTION_STATES)
                    if supported
                    else cls._unresolved_distribution(_INTERACTION_STATES)
                )
                dominant = max(probabilities, key=lambda key: probabilities[key])
                confidence = probabilities[dominant]
                output.append(
                    {
                        "question_version": 1,
                        "question": f"Does {left} depend materially on {right}?",
                        "parameters": [left, right],
                        "kind": "interaction",
                        "conclusion_kind": dominant,
                        "summary": (
                            f"The preferred {left} behavior changes with {right}."
                            if dominant == "MATERIAL_INTERACTION" and reversals > supported / 2
                            else f"{left} × {right}: {dominant.replace('_', ' ').lower()}."
                        ),
                        "confidence": confidence,
                        "confidence_label": _confidence_label(confidence),
                        "probabilities": probabilities,
                        "entropy": _normalized_entropy(probabilities),
                        "materiality": statistics.fmean(magnitudes) if magnitudes else None,
                        "context_reversal_probability": reversals / supported
                        if supported
                        else None,
                        "support": {
                            "completed_candidates": (
                                len(realization_models[0][0]) if realization_models else 0
                            ),
                            "factorial_realizations": supported,
                        },
                        "missing_evidence": (
                            [] if supported else ["matched local factorial support"]
                        ),
                        "evidence": {
                            "agreement": confidence,
                            "interpretation": "pairwise predictive association, not causality",
                        },
                    }
                )
        return output

    @classmethod
    def _practical_region(
        cls,
        parameters: Mapping[int, Mapping[str, Any]],
        outcomes: Mapping[int, Mapping[int | None, float]],
        rows: Mapping[int, float],
        *,
        practical_margin: float | None,
        mode: str,
        noise: SeedNoiseEstimate,
        natural_scale: float,
        realizations: Sequence[Mapping[int, float]],
    ) -> dict[str, Any]:
        if not rows:
            return {"status": "insufficient", "members": [], "probability_by_trial": {}}
        del mode  # evidence is already transformed to a maximized utility scale
        counts = Counter[int]()
        regrets: dict[int, list[float]] = {trial: [] for trial in rows}
        for realization in realizations:
            present = {trial: value for trial, value in realization.items() if trial in rows}
            if not present:
                continue
            best = max(present.values())
            for trial, value in present.items():
                regret = best - value
                regrets[trial].append(regret)
                if practical_margin is not None and regret <= practical_margin:
                    counts[trial] += 1
                elif practical_margin is None and regret <= 0:
                    counts[trial] += 1
        denominator = max(1, len(realizations))
        probabilities = {str(trial): counts[trial] / denominator for trial in sorted(rows)}
        members = [trial for trial in sorted(rows) if counts[trial] > 0]
        return {
            "status": "available" if practical_margin is not None else "winner-uncertainty-only",
            "definition": (
                "expected regret within the authored practical margin"
                if practical_margin is not None
                else "probability of being best; no practical equivalence margin was authored"
            ),
            "practical_margin": practical_margin,
            "members": members,
            "probability_by_trial": probabilities,
            "expected_regret_by_trial": {
                str(trial): statistics.fmean(values) if values else None
                for trial, values in regrets.items()
            },
            "seed_noise": noise.to_dict(),
            "flexibility": {},
            "candidate_count": len(rows),
            "parameters_available": sorted(
                {name for trial in rows for name in parameters.get(trial, {})}
            ),
        }

    @staticmethod
    def _attach_region_flexibility(
        questions: Sequence[dict[str, Any]],
        region: dict[str, Any],
        parameters: Mapping[int, Mapping[str, Any]],
    ) -> None:
        probabilities = region.get("probability_by_trial", {})
        flexibility: dict[str, Any] = {}
        for question in questions:
            name = str(question["parameter"])
            weights: dict[str, float] = {}
            raw_values: dict[str, Any] = {}
            for trial, values in parameters.items():
                if name not in values:
                    continue
                weight = float(probabilities.get(str(trial), 0.0))
                label = _label(values[name])
                weights[label] = weights.get(label, 0.0) + weight
                raw_values[label] = values[name]
            total = sum(weights.values())
            distribution = (
                {label: weight / total for label, weight in weights.items()} if total else {}
            )
            effective = (
                math.exp(
                    -sum(value * math.log(value) for value in distribution.values() if value > 0)
                )
                if distribution
                else 0.0
            )
            classification = (
                "strongly-constrained"
                if effective <= 1.0 + 1e-12
                else "flexible"
                if effective >= max(1.0, len(distribution) - 0.5)
                else "moderately-constrained"
            )
            detail = {
                "classification": classification,
                "effective_supported_values": effective,
                "distribution": distribution,
                "values": raw_values,
            }
            flexibility[name] = detail
            question["practical_region"] = detail
        region["flexibility"] = flexibility

    @classmethod
    def _optimization_opportunity(
        cls,
        candidate_points: Sequence[Mapping[str, Any]],
        observed: Mapping[int, Mapping[str, Any]],
        model: _MixedKnnModel,
        rows: Mapping[int, float],
        *,
        mode: str,
        practical_margin: float | None,
        natural_scale: float,
    ) -> float:
        observed_keys = {
            json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
            for value in observed.values()
        }
        unobserved = [
            value
            for value in candidate_points
            if json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
            not in observed_keys
        ]
        if not rows or not unobserved:
            return 0.0
        del mode  # rows and model predictions are maximized objective utilities
        incumbent = max(rows.values())
        best_ei = 0.0
        for parameters in unobserved:
            mean, deviation = model.predict(parameters)
            improvement_threshold = incumbent + (practical_margin or 0.0)
            best_ei = max(
                best_ei,
                _normal_expected_improvement(mean, deviation, improvement_threshold),
            )
        # The margin defines what counts as an improvement; it must not also amplify a tiny EI
        # into near-certain opportunity.  Normalize magnitude by the observed objective scale.
        scale = max(natural_scale, practical_margin or 0.0)
        return min(1.0, max(0.0, 1.0 - math.exp(-best_ei / max(scale, 1e-12))))

    @classmethod
    def _counterfactual_response(
        cls,
        name: str,
        levels: Sequence[Any],
        basis: _ScientificDesignBasis,
        model: _MixedKnnModel,
    ) -> list[tuple[Any, float]]:
        values: list[tuple[Any, float]] = []
        for level in levels:
            # Average matched counterfactuals over one common reference distribution.  A raw
            # group mean would confound this parameter with whichever settings happened to occur
            # beside each level.  Conditional spaces remain valid because only authored pool
            # members are scored; the reusable basis supplies each reference counterpart once.
            matches = basis.counterfactual_matches(name, level)
            total = sum(weight for _key, _parameters, weight in matches)
            if total:
                values.append(
                    (
                        level,
                        sum(
                            model.predict_keyed(key, parameters)[0] * weight
                            for key, parameters, weight in matches
                        )
                        / total,
                    )
                )
        return values

    @classmethod
    def _factorial_cells(
        cls,
        left: str,
        right: str,
        left_levels: Sequence[Any],
        right_levels: Sequence[Any],
        basis: _ScientificDesignBasis,
        model: _MixedKnnModel,
    ) -> dict[tuple[str, str], float]:
        cells: dict[tuple[str, str], float] = {}
        for left_value in (left_levels[0], left_levels[-1]):
            for right_value in (right_levels[0], right_levels[-1]):
                members = basis.factorial_members(
                    left,
                    right,
                    left_value,
                    right_value,
                    left_levels,
                    right_levels,
                )
                if members:
                    cells[(_label(left_value), _label(right_value))] = statistics.fmean(
                        model.predict(parameters)[0] for parameters in members
                    )
        return cells

    @staticmethod
    def _summarize_responses(
        realizations: Sequence[Sequence[tuple[Any, float]]], levels: Sequence[Any]
    ) -> list[dict[str, Any]]:
        by_level: dict[str, list[float]] = {_label(level): [] for level in levels}
        raw = {_label(level): level for level in levels}
        for response in realizations:
            for level, value in response:
                by_level[_label(level)].append(float(value))
        means = {label: statistics.fmean(values) for label, values in by_level.items() if values}
        if not means:
            return []
        best = max(means.values())
        return [
            {
                "value": _json_value(raw[label]),
                "mean": means[label],
                "standard_deviation": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "lower": _quantile(values, 0.1),
                "upper": _quantile(values, 0.9),
                "preferred": math.isclose(means[label], best, rel_tol=1e-12, abs_tol=1e-12),
            }
            for label, values in by_level.items()
            if values
        ]

    @staticmethod
    def _levels(values: Sequence[Any]) -> tuple[list[Any], bool]:
        unique = {_label(value): value for value in values}
        numeric = bool(unique) and all(
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in unique.values()
        )
        ordered = sorted(unique.values(), key=float if numeric else _label)
        if numeric and len(ordered) > 12:
            indexes = sorted({round(index * (len(ordered) - 1) / 11) for index in range(12)})
            ordered = [ordered[index] for index in indexes]
        return ordered, numeric

    @staticmethod
    def _probabilities(values: Sequence[str], states: Sequence[str]) -> dict[str, float]:
        if not values:
            return ScientificQuestionAnalyzer._unresolved_distribution(states)
        counts = Counter(values)
        total = len(values)
        return {state: counts.get(state, 0) / total for state in states if counts.get(state, 0)}

    @staticmethod
    def _unresolved_distribution(states: Sequence[str]) -> dict[str, float]:
        alternatives = [state for state in states if state != "UNRESOLVED"]
        unresolved = 0.5
        remainder = (1.0 - unresolved) / max(1, len(alternatives))
        return {"UNRESOLVED": unresolved, **{state: remainder for state in alternatives}}

    @classmethod
    def _resample_count(cls, rows: Mapping[int, float], questions: int, *, final: bool) -> int:
        if not final:
            return cls.LIVE_MAX_RESAMPLES
        maximum = cls.FINAL_MAX_RESAMPLES
        # Computational precision, not a scientific threshold: more evidence/questions make a
        # close action ordering worth resolving, while the hard bound protects live refresh.
        target = cls.MIN_RESAMPLES + cls.RESAMPLE_BATCH * math.ceil(
            math.log2(max(1, len(rows) * max(1, questions)))
        )
        return min(maximum, max(cls.MIN_RESAMPLES, target))

    @staticmethod
    def _natural_scale(rows: Mapping[int, float], objective: Mapping[str, Any]) -> float:
        if isinstance(objective.get("metrics"), Mapping):
            return 1.0
        values = sorted(rows.values())
        if len(values) >= 2:
            deviations = [
                abs(left - right) for left, right in zip(values, values[1:], strict=False)
            ]
            positive = [value for value in deviations if value > 0]
            if positive:
                return max(statistics.median(positive), (values[-1] - values[0]) / len(values))
        return max(abs(values[0]), 1.0) if values else 1.0

    @staticmethod
    def _shared_seed_count(outcomes: Mapping[int, Mapping[int | None, float]]) -> int:
        counts = Counter(
            seed for values in outcomes.values() for seed in values if seed is not None
        )
        return sum(count * (count - 1) // 2 for count in counts.values() if count >= 2)

    @staticmethod
    def _seed(fingerprint: str, suffix: str) -> int:
        return int(hashlib.sha256(f"{fingerprint}:{suffix}".encode()).hexdigest()[:16], 16)


class ExperimentalDesignPolicy:
    """Rank finite-pool candidates by practical improvement and question entropy reduction."""

    def __init__(
        self,
        candidates: Mapping[int, Mapping[str, Any]],
        *,
        mode: str,
        practical_margin: float | None,
    ) -> None:
        self.candidates = {int(key): dict(value) for key, value in candidates.items()}
        self.mode = mode
        self.practical_margin = practical_margin

    def rank(
        self,
        outcomes: Mapping[int, Mapping[int | None, float]],
        *,
        selected: Sequence[int],
        scientific_state: Mapping[str, Any],
        costs: Mapping[int, float] | None = None,
        required_candidates: Sequence[int] = (),
        decision_key: str = "",
    ) -> tuple[CandidateDesignValue, ...]:
        rows = {
            trial: statistics.fmean(values.values()) for trial, values in outcomes.items() if values
        }
        if not rows:
            return ()
        noise = SeedNoiseModel.fit(outcomes)
        scale = float(scientific_state.get("natural_utility_scale", 1.0) or 1.0)
        model = _MixedKnnModel(self.candidates, rows, noise=noise, natural_scale=scale)
        questions = [
            value
            for key in ("parameter_questions", "interaction_questions")
            for value in scientific_state.get(key, ())
            if isinstance(value, Mapping)
        ]
        opportunity = float(scientific_state.get("optimization_opportunity", 0.0))
        uncertainty = float(scientific_state.get("scientific_uncertainty", 0.0))
        total = opportunity + uncertainty
        w_opt, w_info = (opportunity / total, uncertainty / total) if total > 0 else (0.5, 0.5)
        durations = [
            float(value) for value in (costs or {}).values() if _finite(value) and float(value) > 0
        ]
        median_cost = statistics.median(durations) if durations else 1.0
        incumbent = (max if self.mode == "max" else min)(rows.values())
        sign = 1.0 if self.mode == "max" else -1.0
        selected_set = set(selected)
        candidates_to_score = self._decision_shortlist(
            selected_set,
            scientific_state=scientific_state,
            required_candidates=required_candidates,
            decision_key=decision_key,
        )
        output: list[CandidateDesignValue] = []
        for trial in candidates_to_score:
            parameters = self.candidates[trial]
            prediction, deviation = model.predict(parameters)
            improvement = _normal_expected_improvement(
                sign * prediction,
                deviation,
                sign * incumbent + (self.practical_margin or 0.0),
            )
            opt_scale = max(scale, self.practical_margin or 0.0)
            optimization_value = 1.0 - math.exp(-improvement / max(opt_scale, 1e-12))
            targeted: list[tuple[float, str, str, float, float]] = []
            for question in questions:
                relevance = self._relevance(parameters, question, selected_set)
                eig = _expected_categorical_entropy_reduction(
                    question.get("conclusion_distribution", question.get("probabilities", {})),
                    float(question.get("support", {}).get("completed_candidates", len(rows))),
                )
                target = (
                    str(question.get("parameter"))
                    if question.get("parameter") is not None
                    else " × ".join(map(str, question.get("parameters", ())))
                )
                targeted.append(
                    (
                        relevance * eig,
                        str(question.get("kind", "parameter")),
                        target,
                        relevance,
                        eig,
                    )
                )
            targeted.sort(reverse=True)
            information_value = min(
                1.0,
                sum(value[0] for value in targeted[: max(1, int(math.sqrt(len(targeted) or 1)))]),
            )
            expected_cost = self._cost(trial, costs or {}, model, median_cost)
            cost_ratio = expected_cost / max(median_cost, 1e-12)
            score = (w_opt * optimization_value + w_info * information_value) / max(
                cost_ratio, 1e-12
            )
            top_value, top_kind, top_target, match_quality, _target_eig = (
                targeted[0]
                if targeted
                else (0.0, "coverage", "search space", 0.0, 0.0)
            )
            targets: tuple[str, ...]
            if w_info * information_value > w_opt * optimization_value and top_value > 0:
                purpose = (
                    "RESOLVE_INTERACTION" if top_kind == "interaction" else "RESOLVE_PARAMETER"
                )
                targets = (top_target,)
                reason = (
                    f"{top_target} currently offers the largest expected question-entropy "
                    "reduction per comparable cost"
                )
            elif information_value > optimization_value and top_value > 0:
                purpose = "EXPLORE_COVERAGE"
                targets = (top_target,)
                reason = (
                    "coverage and model clarification add more value than another "
                    "near-duplicate optimization proposal"
                )
            else:
                purpose = "OPTIMIZE"
                targets = ()
                reason = (
                    "highest expected practical improvement under the current predictive evidence"
                )
            output.append(
                CandidateDesignValue(
                    trial,
                    purpose,
                    targets,
                    optimization_value,
                    information_value,
                    min(1.0, max(0.0, match_quality)),
                    expected_cost,
                    cost_ratio,
                    score,
                    prediction,
                    deviation,
                    reason,
                )
            )
        return tuple(sorted(output, key=lambda value: (-value.score, value.trial)))

    def _decision_shortlist(
        self,
        selected: set[int],
        *,
        scientific_state: Mapping[str, Any],
        required_candidates: Sequence[int],
        decision_key: str,
    ) -> tuple[int, ...]:
        """Bound per-event scientific scoring while keeping the full pool eligible over time.

        The analysis resampling budget already expresses the affordable live precision.  Reusing
        it as the shortlist size avoids a new user-facing knob.  A deterministic decision-keyed
        rotation exposes different pool regions after every event, and the sampler's optimization
        proposal is always retained even when it falls outside this event's scientific sample.
        """
        available = tuple(sorted(set(self.candidates) - selected))
        if not available:
            return ()
        evidence = scientific_state.get("evidence", {})
        raw_budget = evidence.get("resamples") if isinstance(evidence, Mapping) else None
        budget = (
            int(raw_budget)
            if isinstance(raw_budget, int)
            and not isinstance(raw_budget, bool)
            and raw_budget > 0
            else max(1, math.ceil(math.sqrt(len(available))))
        )
        required = tuple(
            dict.fromkeys(
                int(trial)
                for trial in required_candidates
                if int(trial) in self.candidates and int(trial) not in selected
            )
        )
        budget = min(len(available), max(len(required), budget))
        if len(available) <= budget:
            return available
        required_set = set(required)
        rotating = sorted(
            (trial for trial in available if trial not in required_set),
            key=lambda trial: hashlib.sha256(
                (
                    f"{decision_key}:{trial}:"
                    + json.dumps(
                        self.candidates[trial],
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    )
                ).encode()
            ).digest(),
        )
        return tuple((*required, *rotating[: budget - len(required)]))

    def _relevance(
        self, parameters: Mapping[str, Any], question: Mapping[str, Any], selected: set[int]
    ) -> float:
        names = (
            [str(question["parameter"])]
            if question.get("parameter") is not None
            else [str(value) for value in question.get("parameters", ())]
        )
        if not names or any(name not in parameters for name in names):
            return 0.0
        references = [self.candidates[trial] for trial in selected if trial in self.candidates]
        if not references:
            return 1.0
        distances = sorted(
            _mixed_distance(parameters, reference, ignore=names) for reference in references
        )
        nearest = distances[0]
        rank = sum(distance <= nearest for distance in distances) / len(distances)
        matched_quality = 1.0 - min(1.0, nearest)
        novelty = (
            1.0
            if any(
                reference.get(names[0], _INACTIVE) != parameters.get(names[0], _INACTIVE)
                for reference in references
            )
            else 0.0
        )
        return min(1.0, (matched_quality + rank + novelty) / 3.0)

    def _cost(
        self,
        trial: int,
        costs: Mapping[int, float],
        model: _MixedKnnModel,
        fallback: float,
    ) -> float:
        if trial in costs and _finite(costs[trial]) and float(costs[trial]) > 0:
            return float(costs[trial])
        neighbours = sorted(
            (
                (_mixed_distance(self.candidates[trial], self.candidates[other]), float(cost))
                for other, cost in costs.items()
                if other in self.candidates and _finite(cost) and float(cost) > 0
            ),
            key=lambda value: value[0],
        )
        if not neighbours:
            return fallback
        selected = neighbours[: max(1, math.ceil(math.sqrt(len(neighbours))))]
        return statistics.median(value for _distance, value in selected)


class _MixedKnnModel:
    def __init__(
        self,
        parameters: Mapping[int, Mapping[str, Any]],
        rows: Mapping[int, float],
        *,
        noise: SeedNoiseEstimate | None,
        natural_scale: float,
        distance_cache: dict[str, dict[int, float]] | None = None,
        distance_order_cache: dict[str, tuple[int, tuple[int, ...]]] | None = None,
        validation_error: float | None = None,
    ) -> None:
        self.parameters = {
            trial: dict(values) for trial, values in parameters.items() if trial in rows
        }
        self.rows = {trial: float(rows[trial]) for trial in self.parameters}
        self.noise = noise
        self.natural_scale = max(float(natural_scale), 1e-12)
        # Objective realizations change their values and included candidates, but the mixed-space
        # geometry does not.  Sharing this cache prevents every resample from recalculating the
        # same query-to-observation distances.
        self._distance_cache = distance_cache if distance_cache is not None else {}
        self._distance_order_cache = (
            distance_order_cache if distance_order_cache is not None else {}
        )
        # Evidence realizations share the same observed design and use the base model's measured
        # leave-one-candidate-out scale. Recomputing it for every resample is both statistically
        # redundant for this approximation and quadratic in accumulated candidate evidence.
        self.validation_error = (
            float(validation_error)
            if validation_error is not None
            else self._validation_error()
        )
        self._prediction_cache: dict[str, tuple[float, float]] = {}

    def predict(self, parameters: Mapping[str, Any]) -> tuple[float, float]:
        key = json.dumps(parameters, sort_keys=True, separators=(",", ":"), default=str)
        return self.predict_keyed(key, parameters)

    def predict_keyed(
        self, key: str, parameters: Mapping[str, Any]
    ) -> tuple[float, float]:
        """Predict using a caller-owned stable parameter key to avoid repeated serialization."""
        cached = self._prediction_cache.get(key)
        if cached is not None:
            return cached
        if not self.rows:
            return 0.0, self.natural_scale
        distances = self._distances(parameters, key=key)
        ordered_trials = self._ordered_trials(key, distances)
        count = max(1, math.ceil(math.sqrt(len(self.rows))))
        selected: list[tuple[float, float]] = []
        for trial in ordered_trials:
            if trial in self.rows:
                selected.append((distances[trial], self.rows[trial]))
                if len(selected) >= count:
                    break
        weights = [1.0 / max(distance, 1e-6) for distance, _value in selected]
        mean = sum(
            weight * value for weight, (_distance, value) in zip(weights, selected, strict=True)
        ) / sum(weights)
        local = (
            math.sqrt(
                sum(
                    weight * (value - mean) ** 2
                    for weight, (_distance, value) in zip(weights, selected, strict=True)
                )
                / sum(weights)
            )
            if len(selected) > 1
            else 0.0
        )
        seed = (
            math.sqrt(self.noise.variance)
            if self.noise is not None and self.noise.variance is not None
            else 0.0
        )
        distance = statistics.fmean(value[0] for value in selected)
        epistemic = self.validation_error / math.sqrt(max(1, len(self.rows)))
        deviation = math.sqrt(
            local * local
            + epistemic**2
            + seed * seed
            + (distance * self.natural_scale) ** 2
        )
        result = (mean, max(deviation, 1e-12))
        self._prediction_cache[key] = result
        return result

    def _validation_error(self) -> float:
        if len(self.rows) < 2:
            return self.natural_scale
        errors: list[float] = []
        for trial, value in self.rows.items():
            parameters = self.parameters[trial]
            key = json.dumps(parameters, sort_keys=True, separators=(",", ":"), default=str)
            distances = self._distances(parameters, key=key)
            count = max(1, math.ceil(math.sqrt(len(self.rows) - 1)))
            selected = []
            for other in self._ordered_trials(key, distances):
                if other in self.rows and other != trial:
                    selected.append((distances[other], self.rows[other]))
                    if len(selected) >= count:
                        break
            weights = [1.0 / max(distance, 1e-6) for distance, _value in selected]
            prediction = sum(
                weight * observed
                for weight, (_distance, observed) in zip(weights, selected, strict=True)
            ) / sum(weights)
            errors.append(value - prediction)
        return (
            math.sqrt(statistics.fmean(value * value for value in errors))
            if errors
            else self.natural_scale
        )

    def _distances(
        self, parameters: Mapping[str, Any], *, key: str | None = None
    ) -> dict[int, float]:
        key = key or json.dumps(
            parameters, sort_keys=True, separators=(",", ":"), default=str
        )
        distances = self._distance_cache.setdefault(key, {})
        for trial, observed in self.parameters.items():
            if trial not in distances:
                distances[trial] = _mixed_distance(parameters, observed)
        return distances

    def _ordered_trials(self, key: str, distances: Mapping[int, float]) -> tuple[int, ...]:
        cached = self._distance_order_cache.get(key)
        if cached is not None and cached[0] == len(distances):
            return cached[1]
        ordered = tuple(sorted(distances, key=distances.__getitem__))
        self._distance_order_cache[key] = (len(distances), ordered)
        return ordered


def _expected_categorical_entropy_reduction(raw: Any, support: float) -> float:
    probabilities = (
        {
            str(key): float(value)
            for key, value in raw.items()
            if _finite(value) and float(value) > 0
        }
        if isinstance(raw, Mapping)
        else {}
    )
    if len(probabilities) < 2:
        return 0.0
    total_probability = sum(probabilities.values())
    probabilities = {key: value / total_probability for key, value in probabilities.items()}
    current = _normalized_entropy(probabilities)
    concentration = max(1.0, float(support))
    expected = 0.0
    for observed, probability in probabilities.items():
        posterior = {
            state: (concentration * value + (1.0 if state == observed else 0.0))
            / (concentration + 1.0)
            for state, value in probabilities.items()
        }
        expected += probability * _normalized_entropy(posterior)
    return max(0.0, current - expected)


def _normal_expected_improvement(mean: float, deviation: float, threshold: float) -> float:
    if deviation <= 0:
        return max(0.0, mean - threshold)
    z = (mean - threshold) / deviation
    normal = statistics.NormalDist()
    density = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return max(0.0, (mean - threshold) * normal.cdf(z) + deviation * density)


def _mixed_distance(
    left: Mapping[str, Any], right: Mapping[str, Any], *, ignore: Sequence[str] = ()
) -> float:
    ignored = set(ignore)
    names = sorted((set(left) | set(right)) - ignored)
    if not names:
        return 0.0
    values: list[float] = []
    for name in names:
        first, second = left.get(name, _INACTIVE), right.get(name, _INACTIVE)
        if (
            isinstance(first, int | float)
            and not isinstance(first, bool)
            and isinstance(second, int | float)
            and not isinstance(second, bool)
        ):
            scale = max(abs(float(first)), abs(float(second)), 1.0)
            values.append(min(1.0, abs(float(first) - float(second)) / scale))
        else:
            values.append(0.0 if first == second else 1.0)
    return math.sqrt(statistics.fmean(value * value for value in values))


def _projected_key(parameters: Mapping[str, Any], *, ignore: Sequence[str]) -> str:
    """Encode one mixed-space point after removing the experimentally varied fields."""
    ignored = set(ignore)
    return json.dumps(
        {key: parameters[key] for key in sorted(parameters) if key not in ignored},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _normalized_entropy(probabilities: Mapping[str, float]) -> float:
    values = [float(value) for value in probabilities.values() if float(value) > 0]
    if len(values) <= 1:
        return 0.0
    total = sum(values)
    entropy = -sum((value / total) * math.log(value / total) for value in values)
    return min(1.0, max(0.0, entropy / math.log(len(values))))


def _confidence_label(value: float) -> str:
    # Presentation labels only; the controller consumes the continuous confidence/entropy.
    return "high" if value >= 0.85 else "medium" if value >= 0.6 else "low"


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (index - lower) * (ordered[upper] - ordered[lower])


def _label(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _display_label(value: Any) -> str:
    if value == _INACTIVE:
        return "inactive"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _endpoint_level(value: Any, levels: Sequence[Any], *, numeric: bool) -> Any | None:
    if len(levels) < 2:
        return None
    if numeric and _finite(value):
        midpoint = (float(levels[0]) + float(levels[-1])) / 2.0
        return levels[0] if float(value) <= midpoint else levels[-1]
    if value == levels[0]:
        return levels[0]
    if value == levels[-1]:
        return levels[-1]
    return None


def _json_value(value: Any) -> Any:
    return None if value == _INACTIVE else value


def _finite(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def random_probability(realization: Mapping[int, float], name: str) -> float:
    """Deterministic pseudo-uniform used to couple interaction and parameter resamples."""
    payload = json.dumps(sorted(realization.items()), separators=(",", ":")) + name
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16) / float(16**16 - 1)


__all__ = [
    "CandidateDesignValue",
    "ExperimentalDesignPolicy",
    "ScientificQuestionAnalyzer",
    "SeedNoiseEstimate",
    "SeedNoiseModel",
]
