"""Small provider-neutral adaptive study policy and scoring helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from lambdaforge.execution.ResourceRequest import ResourceRequest


@dataclass(frozen=True, slots=True)
class FidelityPolicy:
    """Explicit cumulative Work budget; units are defined by the consumer Work."""

    minimum: int
    maximum: int
    reduction_factor: int = 3

    def __post_init__(self) -> None:
        if self.minimum < 1 or self.maximum < self.minimum:
            raise ValueError("search.fidelity requires 1 <= min <= max.")
        if self.reduction_factor < 2:
            raise ValueError("search.fidelity.reduction_factor must be >= 2.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> FidelityPolicy:
        unknown = set(value) - {"min", "max", "reduction_factor"}
        if unknown:
            raise ValueError(f"Unknown search.fidelity field(s): {sorted(unknown)}.")
        if "min" not in value or "max" not in value:
            raise ValueError("search.fidelity requires min and max.")
        return cls(
            int(value["min"]),
            int(value["max"]),
            int(value.get("reduction_factor", 3)),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "min": self.minimum,
            "max": self.maximum,
            "reduction_factor": self.reduction_factor,
        }


@dataclass(frozen=True, slots=True)
class ControllerValuePolicy:
    """Bounded common-scale heuristics for unlike scheduler actions.

    These are controller-value proxies, not Shannon information gain or calibrated probabilities.
    Seed actions use measured standard-error reduction; these constants put coverage and fidelity
    actions on a transparent conservative scale until retrospective calibration can own them.
    """

    new_region_base: float = 0.15
    observation_sparsity_weight: float = 0.35
    periodic_exploration_bonus: float = 0.10
    fidelity_uncertainty_weight: float = 0.35

    def to_dict(self) -> dict[str, float | str]:
        return {
            "meaning": "bounded-controller-value-proxy-not-information-gain",
            "new_region_base": self.new_region_base,
            "observation_sparsity_weight": self.observation_sparsity_weight,
            "periodic_exploration_bonus": self.periodic_exploration_bonus,
            "fidelity_uncertainty_weight": self.fidelity_uncertainty_weight,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveSearchPolicy:
    """Sequential candidate, fidelity and probability-driven seed policy."""

    # ``None`` is the canonical internal representation of automatic operational/scientific
    # limits. Integer values remain hard user caps.
    runs_per_gpu: int | None = None
    max_parallel: int | None = None
    min_seeds: int = 1
    candidate_budget: int | None = None
    proposal_pool_size: int = 320
    goal: str = "balanced"
    automatic_stop: bool = True
    conclusion_stability: float = 0.95
    early_stopping: bool = True
    early_stopping_min_step: int = 3
    early_stopping_confirmations: int = 2
    early_stopping_probability_threshold: float = 0.05
    early_stopping_equivalence_margin: float = 0.0
    # ``None`` means automatic geometry-derived initial design.  Scientific startup size is
    # deliberately independent of GPUs and execution parallelism.
    startup_trials: int | None = None
    failure_retries: int = 1
    seed_probability_threshold: float = 0.1
    equivalence_margin: float = 0.0
    # Internal provenance: ``None`` means no scientific practical-equivalence scale was authored.
    # The legacy fields remain accepted and feed this one authority during parsing.
    practical_margin: float | None = None
    confirmation_top_k: int = 1
    confirmation_seeds: tuple[int, ...] = ()
    confirmation_auto: bool = True
    max_runs: int | None = None
    max_time_seconds: float | None = None
    # Zero is deliberate: ``trials`` is the authored candidate budget.  Merely observing a
    # sequence without a new record is not evidence that the mixed search space is covered.
    # Researchers may opt into record-based convergence by setting this field explicitly.
    convergence_patience: int = 0
    min_improvement: float = 0.0
    sampler: str = "auto"
    fidelity: FidelityPolicy | None = None
    # Normalized authored dimensions carried internally for resource-space distance.
    parameter_space: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "min_seeds",
            "confirmation_top_k",
            "proposal_pool_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"search.{name} must be a positive integer.")
        if self.candidate_budget is not None and (
            isinstance(self.candidate_budget, bool)
            or not isinstance(self.candidate_budget, int)
            or self.candidate_budget < 1
        ):
            raise ValueError("search.candidate_budget must be a positive integer or auto.")
        if self.runs_per_gpu is not None and (
            isinstance(self.runs_per_gpu, bool)
            or not isinstance(self.runs_per_gpu, int)
            or self.runs_per_gpu < 1
        ):
            raise ValueError("search.runs_per_gpu must be a positive integer or auto.")
        if self.startup_trials is not None and (
            isinstance(self.startup_trials, bool)
            or not isinstance(self.startup_trials, int)
            or self.startup_trials < 1
        ):
            raise ValueError("search.startup_trials must be a positive integer or omitted.")
        if self.max_parallel is not None and (
            isinstance(self.max_parallel, bool) or self.max_parallel < 1
        ):
            raise ValueError("search.max_parallel must be a positive integer, auto or null.")
        if self.early_stopping_min_step < 1:
            raise ValueError("search.early_stopping.min_step must be >= 1.")
        if self.early_stopping_confirmations < 1:
            raise ValueError("search.early_stopping.confirmations must be >= 1.")
        if self.candidate_budget is not None and self.proposal_pool_size < self.candidate_budget:
            raise ValueError("search.proposal_pool_size must be >= search.trials.")
        if self.goal not in {"optimize", "balanced", "understand"}:
            raise ValueError("search.goal must be optimize, balanced or understand.")
        if not 0 < self.conclusion_stability < 1:
            raise ValueError("search.stop.stability must be strictly between 0 and 1.")
        if (
            isinstance(self.failure_retries, bool)
            or not isinstance(self.failure_retries, int)
            or not 0 <= self.failure_retries <= 3
        ):
            raise ValueError("search.failure_retries must be an integer from 0 to 3.")
        if not 0 <= self.seed_probability_threshold <= 1:
            raise ValueError("search.seed_probability_threshold must be in [0, 1].")
        if not 0 <= self.early_stopping_probability_threshold <= 1:
            raise ValueError("search.early_stopping.probability_threshold must be in [0, 1].")
        for name in (
            "equivalence_margin",
            "early_stopping_equivalence_margin",
            "min_improvement",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"search.{name} must be finite and non-negative.")
        if self.practical_margin is not None and (
            not math.isfinite(self.practical_margin) or self.practical_margin < 0
        ):
            raise ValueError("The practical equivalence margin must be finite and non-negative.")
        if len(self.confirmation_seeds) != len(set(self.confirmation_seeds)):
            raise ValueError("search.confirmation_seeds cannot contain duplicates.")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) for seed in self.confirmation_seeds
        ):
            raise TypeError("search.confirmation_seeds must contain integers.")
        if self.max_runs is not None and (
            isinstance(self.max_runs, bool)
            or not isinstance(self.max_runs, int)
            or self.max_runs < 1
        ):
            raise ValueError("search.max_runs must be a positive integer or null.")
        if self.max_time_seconds is not None and self.max_time_seconds <= 0:
            raise ValueError("search.max_time must be positive or null.")
        if (
            isinstance(self.convergence_patience, bool)
            or not isinstance(self.convergence_patience, int)
            or self.convergence_patience < 0
        ):
            raise ValueError("search.convergence_patience must be a non-negative integer.")
        if self.sampler not in {"auto", "knn", "botorch"}:
            raise ValueError("search.sampler must be auto, knn or botorch.")

    @classmethod
    def from_search(cls, value: Mapping[str, Any]) -> AdaptiveSearchPolicy:
        removed = set(value).intersection({"reduction_factor", "confidence"})
        if removed:
            raise ValueError(
                "Removed ambiguous HPO field(s) "
                f"{sorted(removed)}. Use search.fidelity.reduction_factor for fidelity spacing; "
                "seed_racing.probability_threshold and early_stopping.probability_threshold "
                "own their separate statistical decisions."
            )
        raw_early = value.get("early_stopping", True)
        raw_seed_racing = value.get("seed_racing", {})
        if not isinstance(raw_seed_racing, Mapping):
            raise TypeError("search.seed_racing must be a mapping.")
        unknown_seed = set(raw_seed_racing) - {"probability_threshold", "equivalence_margin"}
        if unknown_seed:
            raise ValueError(f"Unknown search.seed_racing field(s): {sorted(unknown_seed)}.")
        seed_margin_authored = (
            "equivalence_margin" in raw_seed_racing or "equivalence_margin" in value
        )
        seed_margin = float(
            raw_seed_racing.get("equivalence_margin", value.get("equivalence_margin", 0.0))
        )
        early_margin_authored = isinstance(raw_early, Mapping) and "equivalence_margin" in raw_early
        if isinstance(raw_early, bool):
            early, min_step, early_probability, early_margin = raw_early, 3, 0.05, 0.0
        elif isinstance(raw_early, Mapping):
            unknown = set(raw_early) - {
                "enabled",
                "min_step",
                "confirmations",
                "probability_threshold",
                "equivalence_margin",
            }
            if unknown:
                raise ValueError(f"Unknown search.early_stopping field(s): {sorted(unknown)}.")
            early = bool(raw_early.get("enabled", True))
            min_step = int(raw_early.get("min_step", 3))
            confirmations = int(raw_early.get("confirmations", 2))
            early_probability = float(raw_early.get("probability_threshold", 0.05))
            early_margin = float(raw_early.get("equivalence_margin", seed_margin))
        else:
            raise TypeError("search.early_stopping must be true/false or a mapping.")
        if isinstance(raw_early, bool):
            confirmations = 2
            if seed_margin_authored:
                early_margin = seed_margin
        if (
            seed_margin_authored
            and early_margin_authored
            and not math.isclose(seed_margin, early_margin, rel_tol=0.0, abs_tol=0.0)
        ):
            raise ValueError(
                "search.seed_racing.equivalence_margin and "
                "search.early_stopping.equivalence_margin describe the same scientific practical "
                "difference and cannot disagree."
            )
        practical_margin = (
            seed_margin if seed_margin_authored else early_margin if early_margin_authored else None
        )
        raw_candidate_budget = value.get("trials")
        candidate_budget = (
            int(raw_candidate_budget) if raw_candidate_budget is not None else None
        )
        proposal_pool_size = int(
            value.get(
                "proposal_pool_size",
                max(candidate_budget, min(4096, candidate_budget * 16))
                if candidate_budget is not None
                else 320,
            )
        )
        maximum = value.get("max_parallel")
        if isinstance(maximum, str) and maximum.lower() != "auto":
            raise ValueError("search.max_parallel must be a positive integer, auto or null.")
        raw_runs_per_gpu = value.get("runs_per_gpu", "auto")
        if isinstance(raw_runs_per_gpu, str):
            if raw_runs_per_gpu.lower() != "auto":
                raise ValueError("search.runs_per_gpu must be a positive integer or auto.")
            parsed_runs_per_gpu = None
        else:
            parsed_runs_per_gpu = int(raw_runs_per_gpu)
        raw_confirmation = value.get("confirmation_seeds", ())
        if not isinstance(raw_confirmation, (list, tuple)):
            raise TypeError("search.confirmation_seeds must be a list of integers.")
        raw_time = value.get("max_time")
        max_time = (
            float(raw_time)
            if isinstance(raw_time, (int, float))
            else ResourceRequest.from_mapping({"time": raw_time}).runtime_seconds
            if raw_time is not None
            else None
        )
        max_runs = value.get("max_runs")
        raw_fidelity = value.get("fidelity")
        if raw_fidelity is not None and not isinstance(raw_fidelity, Mapping):
            raise TypeError("search.fidelity must be a mapping.")
        raw_stop = value.get("stop", "auto")
        if isinstance(raw_stop, str):
            if raw_stop.lower() != "auto":
                raise ValueError("search.stop must be auto or a mapping.")
            automatic_stop = True
            stability = 0.95
        elif isinstance(raw_stop, Mapping):
            # ``to_dict`` is also the persisted, replayable input accepted here.
            unknown_stop = set(raw_stop) - {"mode", "stability", "policy_version"}
            if unknown_stop:
                raise ValueError(f"Unknown search.stop field(s): {sorted(unknown_stop)}.")
            stop_mode = str(raw_stop.get("mode", "auto")).lower()
            if stop_mode not in {"auto", "disabled"}:
                raise ValueError("search.stop.mode must be auto or disabled.")
            automatic_stop = stop_mode == "auto"
            stability = float(raw_stop.get("stability", 0.95))
        else:
            raise TypeError("search.stop must be auto or a mapping.")
        return cls(
            runs_per_gpu=parsed_runs_per_gpu,
            max_parallel=(
                int(maximum) if maximum is not None and not isinstance(maximum, str) else None
            ),
            min_seeds=int(value.get("min_seeds", 1)),
            candidate_budget=candidate_budget,
            proposal_pool_size=proposal_pool_size,
            goal=str(value.get("goal", "balanced")).lower(),
            automatic_stop=automatic_stop,
            conclusion_stability=stability,
            early_stopping=early,
            early_stopping_min_step=min_step,
            early_stopping_confirmations=confirmations,
            early_stopping_probability_threshold=early_probability,
            early_stopping_equivalence_margin=early_margin,
            startup_trials=(
                int(value["startup_trials"]) if value.get("startup_trials") is not None else None
            ),
            failure_retries=int(value.get("failure_retries", 1)),
            seed_probability_threshold=float(
                raw_seed_racing.get(
                    "probability_threshold", value.get("seed_probability_threshold", 0.1)
                )
            ),
            equivalence_margin=seed_margin,
            practical_margin=practical_margin,
            confirmation_top_k=int(value.get("confirmation_top_k", 1)),
            confirmation_seeds=tuple(raw_confirmation),
            confirmation_auto=(
                str(value.get("confirmation", "")).lower() == "auto"
                or "confirmation_seeds" not in value
            ),
            max_runs=int(max_runs) if max_runs is not None else None,
            max_time_seconds=max_time,
            convergence_patience=int(value.get("convergence_patience", 0)),
            min_improvement=float(value.get("min_improvement", 0.0)),
            sampler=str(value.get("sampler", "auto")).lower(),
            fidelity=(
                FidelityPolicy.from_mapping(raw_fidelity)
                if isinstance(raw_fidelity, Mapping)
                else None
            ),
            parameter_space={
                str(name): (
                    dict(descriptor)
                    if isinstance(descriptor, Mapping)
                    else {"values": list(descriptor)}
                    if isinstance(descriptor, (list, tuple))
                    else {"values": [descriptor]}
                )
                for name, descriptor in (
                    value.get("parameter_space", value).items()
                    if isinstance(value.get("parameter_space", value), Mapping)
                    else ()
                )
                if str(name) not in SEARCH_POLICY_FIELDS and str(name) != "trials"
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": "adaptive",
            "runs_per_gpu": self.runs_per_gpu if self.runs_per_gpu is not None else "auto",
            "max_parallel": self.max_parallel if self.max_parallel is not None else "auto",
            "min_seeds": self.min_seeds,
            "trials": self.candidate_budget,
            "proposal_pool_size": self.proposal_pool_size,
            "goal": self.goal,
            "stop": {
                "mode": "auto" if self.automatic_stop else "disabled",
                "stability": self.conclusion_stability,
                "policy_version": "scientific-convergence-v1",
            },
            "early_stopping": {
                "enabled": self.early_stopping,
                "min_step": self.early_stopping_min_step,
                "confirmations": self.early_stopping_confirmations,
                "probability_threshold": self.early_stopping_probability_threshold,
                "equivalence_margin": self.early_stopping_equivalence_margin,
            },
            "startup_trials": self.startup_trials,
            "failure_retries": self.failure_retries,
            "seed_racing": {
                "probability_threshold": self.seed_probability_threshold,
                "equivalence_margin": self.equivalence_margin,
            },
            "practical_equivalence_margin": self.scientific_margin,
            "confirmation_top_k": self.confirmation_top_k,
            "confirmation_seeds": list(self.confirmation_seeds),
            "confirmation": "auto" if self.confirmation_auto else "explicit",
            "max_runs": self.max_runs,
            "max_time": self.max_time_seconds,
            "convergence_patience": self.convergence_patience,
            "min_improvement": self.min_improvement,
            "sampler": self.sampler,
            "fidelity": self.fidelity.to_dict() if self.fidelity is not None else None,
            "parameter_space": {
                str(name): dict(descriptor)
                for name, descriptor in self.parameter_space.items()
                if isinstance(descriptor, Mapping)
            },
        }

    @property
    def scientific_margin(self) -> float | None:
        """Return the one authored scientific equivalence scale, if one exists."""
        if self.practical_margin is not None:
            return self.practical_margin
        return self.equivalence_margin if self.equivalence_margin > 0 else None


SEARCH_POLICY_FIELDS = frozenset(
    {
        "strategy",
        "runs_per_gpu",
        "max_parallel",
        "min_seeds",
        "early_stopping",
        "seed_racing",
        "proposal_pool_size",
        "startup_trials",
        "failure_retries",
        "seed_probability_threshold",
        "equivalence_margin",
        "confirmation_top_k",
        "confirmation_seeds",
        "max_runs",
        "max_time",
        "convergence_patience",
        "min_improvement",
        "sampler",
        "fidelity",
        "parameter_space",
        "goal",
        "stop",
    }
)


__all__ = [
    "AdaptiveSearchPolicy",
    "ControllerValuePolicy",
    "FidelityPolicy",
    "SEARCH_POLICY_FIELDS",
]
