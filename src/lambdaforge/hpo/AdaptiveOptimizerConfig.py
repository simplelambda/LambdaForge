"""Validated adaptive optimization configuration."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from lambdaforge.hpo.SearchSpace import SearchSpace


class AdaptiveOptimizerConfig:
    """Normalize the strict optional ``hpo`` block into controller settings."""

    _BYTES = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b)?\s*$", re.IGNORECASE)

    def __init__(self, value: Mapping[str, Any], *, base: Mapping[str, Any] | None = None) -> None:
        self.raw = dict(value)
        self._base = dict(base or {})
        self.enabled = bool(value.get("enabled", False))
        objective = self._mapping(value.get("objective", {}), "hpo.objective")
        self.metric = str(objective.get("metric", "val_loss"))
        self.direction = str(objective.get("direction", "minimize"))
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("hpo.objective.direction must be maximize or minimize.")
        risk = self._mapping(objective.get("risk", {}), "hpo.objective.risk")
        self.risk_type = str(risk.get("type", "mean"))
        self.risk_lambda = float(risk.get("lambda", 0.0))
        if self.risk_type not in {"mean", "mean_minus_std"} or self.risk_lambda < 0:
            raise ValueError("Invalid hpo objective risk policy.")

        raw_space = self._mapping(value.get("space", {}), "hpo.space")
        self.space = SearchSpace.from_mapping(raw_space)

        initialization = self._mapping(value.get("initialization", {}), "hpo.initialization")
        self.initialization_strategy = str(initialization.get("strategy", "sobol"))
        if self.initialization_strategy not in {"sobol", "random"}:
            raise ValueError("hpo.initialization.strategy must be sobol or random.")
        raw_initial_trials = initialization.get("trials", "auto")
        self.initial_trials = (
            max(4, 2 * (self.space.dimension + 1))
            if raw_initial_trials == "auto"
            else self._positive_int(raw_initial_trials, "hpo.initialization.trials")
        )

        search = self._mapping(value.get("search", {}), "hpo.search")
        self.search_strategy = str(search.get("strategy", "bayesian"))
        if self.search_strategy not in {"bayesian", "sobol", "random"}:
            raise ValueError("hpo.search.strategy must be bayesian, sobol or random.")
        self.candidate_pool_size = self._positive_int(
            search.get("candidate_pool_size", 128), "hpo.search.candidate_pool_size"
        )
        self.surrogate_refresh_interval = self._positive_int(
            search.get("refresh_interval", 1), "hpo.search.refresh_interval"
        )
        self.controller_seed = self._non_negative_int(
            value.get("controller_seed", 0), "hpo.controller_seed"
        )

        fidelity = self._mapping(value.get("fidelity", {}), "hpo.fidelity")
        self.fidelity_unit = str(fidelity.get("unit", "epochs"))
        if self.fidelity_unit != "epochs":
            raise ValueError("The current training backend supports hpo fidelity unit 'epochs'.")
        self.min_budget = self._positive_int(fidelity.get("min", 5), "hpo.fidelity.min")
        self.max_budget = self._positive_int(fidelity.get("max", 100), "hpo.fidelity.max")
        self.budget_step = self._positive_int(
            fidelity.get("step", self.min_budget), "hpo.fidelity.step"
        )
        self.fidelity_strategy = str(fidelity.get("strategy", "adaptive_learning_curve"))
        if self.fidelity_strategy not in {
            "adaptive_learning_curve",
            "fixed",
            "successive_halving",
        }:
            raise ValueError("Unsupported hpo fidelity strategy.")
        if self.max_budget < self.min_budget or self.budget_step > self.max_budget:
            raise ValueError("hpo fidelity requires 0 < min <= max and step <= max.")

        seeds = self._mapping(value.get("seeds", {}), "hpo.seeds")
        self.seed_strategy = str(seeds.get("strategy", "adaptive_racing"))
        if self.seed_strategy not in {"adaptive_racing", "fixed"}:
            raise ValueError("hpo.seeds.strategy must be adaptive_racing or fixed.")
        self.search_seeds = self._integer_sequence(seeds.get("values", (0,)), "hpo.seeds.values")
        self.confirmation_seeds = self._integer_sequence(
            seeds.get("confirmation_values", ()), "hpo.seeds.confirmation_values", allow_empty=True
        )
        overlap = set(self.search_seeds) & set(self.confirmation_seeds)
        if overlap:
            raise ValueError(f"Search and confirmation seeds must be disjoint: {sorted(overlap)}.")
        self.max_search_seeds = self._positive_int(
            seeds.get("max_search_seeds", len(self.search_seeds)),
            "hpo.seeds.max_search_seeds",
        )
        self.seed_probability_threshold = self._probability(
            seeds.get("probability_threshold", 0.9), "hpo.seeds.probability_threshold"
        )

        acquisition = self._mapping(value.get("acquisition", {}), "hpo.acquisition")
        self.acquisition_strategy = str(
            acquisition.get("strategy", "cost_aware_knowledge_gradient")
        )
        if self.acquisition_strategy not in {
            "cost_aware_knowledge_gradient",
            "knowledge_gradient",
            "expected_improvement",
        }:
            raise ValueError("Unsupported hpo acquisition strategy.")
        self.exploration_weight = self._positive_float(
            acquisition.get("exploration_weight", 1.0), "hpo.acquisition.exploration_weight"
        )

        pruning = self._mapping(value.get("pruning", {}), "hpo.pruning")
        self.pruning_enabled = bool(pruning.get("enabled", True))
        self.min_budget_before_drop = self._positive_int(
            pruning.get("min_budget_before_drop", self.min_budget),
            "hpo.pruning.min_budget_before_drop",
        )
        self.drop_probability_threshold = self._probability(
            pruning.get("probability_threshold", 0.01),
            "hpo.pruning.probability_threshold",
            include_zero=True,
        )
        self.equivalence_margin = self._non_negative_float(
            pruning.get("equivalence_margin", 0.0), "hpo.pruning.equivalence_margin"
        )

        memory = self._mapping(value.get("memory", {}), "hpo.memory")
        self.memory_per_job_bytes = self.parse_bytes(memory.get("per_job_budget", 0))
        self.memory_headroom_bytes = self.parse_bytes(memory.get("headroom", 0))
        self.memory_safety_quantile = self._probability(
            memory.get("safety_quantile", 0.99), "hpo.memory.safety_quantile"
        )
        self.memory_min_observations = self._positive_int(
            memory.get("min_observations", 3), "hpo.memory.min_observations"
        )
        self.allocator_cap = bool(memory.get("allocator_cap", True))
        self.memory_preflight = bool(memory.get("preflight", False))
        self.memory_probe_spec = memory.get("probe")
        if self.memory_preflight and not isinstance(self.memory_probe_spec, Mapping):
            raise ValueError("hpo.memory.preflight requires an importable memory.probe spec.")
        if self.memory_preflight and self.memory_per_job_bytes <= 0:
            raise ValueError("hpo.memory.preflight requires a positive per_job_budget.")
        raw_capacities = memory.get("device_capacities", ())
        if not isinstance(raw_capacities, Sequence) or isinstance(raw_capacities, (str, bytes)):
            raise TypeError("hpo.memory.device_capacities must be a sequence.")
        self.device_capacities = tuple(self.parse_bytes(item) for item in raw_capacities)
        raw_resource_features = self._mapping(
            memory.get("resource_features", {}), "hpo.memory.resource_features"
        )
        self.resource_feature_paths = {
            str(name): str(path) for name, path in raw_resource_features.items()
        }
        if any(not name or not path for name, path in self.resource_feature_paths.items()):
            raise ValueError("hpo.memory.resource_features requires non-empty names and paths.")
        probe_policy = self._mapping(memory.get("probe_policy", {}), "hpo.memory.probe_policy")
        self.memory_probe_mode = str(
            probe_policy.get("mode", "auto" if self.memory_preflight else "never")
        )
        if self.memory_probe_mode not in {"auto", "always", "never"}:
            raise ValueError("hpo.memory.probe_policy.mode must be auto, always or never.")
        if self.memory_preflight and self.memory_probe_mode == "never":
            raise ValueError("hpo.memory.preflight contradicts probe_policy.mode: never.")
        if self.memory_probe_mode != "never" and not isinstance(self.memory_probe_spec, Mapping):
            raise ValueError("An enabled memory probe policy requires hpo.memory.probe.")
        if self.memory_probe_mode != "never" and self.memory_per_job_bytes <= 0:
            raise ValueError("Candidate memory probes require a positive per_job_budget.")
        self.memory_probe_relative_uncertainty = self._non_negative_float(
            probe_policy.get("relative_uncertainty_threshold", 0.25),
            "hpo.memory.probe_policy.relative_uncertainty_threshold",
        )
        self.memory_probe_near_limit_fraction = self._probability(
            probe_policy.get("near_limit_fraction", 0.85),
            "hpo.memory.probe_policy.near_limit_fraction",
        )
        self.memory_probe_oom_probability = self._probability(
            probe_policy.get("oom_probability_threshold", 0.05),
            "hpo.memory.probe_policy.oom_probability_threshold",
            include_zero=True,
        )
        self.unknown_memory_policy = str(memory.get("unknown_capacity", "declared_budget"))
        if self.unknown_memory_policy not in {"declared_budget", "fail_closed"}:
            raise ValueError("hpo.memory.unknown_capacity must be declared_budget or fail_closed.")
        structural = self._mapping(memory.get("structural", {}), "hpo.memory.structural")
        self.memory_parameter_count_feature = str(
            structural.get("parameter_count_feature", "parameter_count")
        )
        self.memory_bytes_per_parameter = self._positive_int(
            structural.get("bytes_per_parameter", 4),
            "hpo.memory.structural.bytes_per_parameter",
        )
        self.memory_gradient_copies = self._non_negative_float(
            structural.get("gradient_copies", 1.0),
            "hpo.memory.structural.gradient_copies",
        )
        self.memory_optimizer_copies = self._non_negative_float(
            structural.get("optimizer_copies", 2.0),
            "hpo.memory.structural.optimizer_copies",
        )
        self.memory_buffer_bytes = self.parse_bytes(structural.get("buffer_bytes", 0))

        budget = self._mapping(value.get("budget", {}), "hpo.budget")
        self.max_actions = self._positive_int(
            budget.get("max_actions", 50), "hpo.budget.max_actions"
        )
        self.max_total_epochs = self._positive_int(
            budget.get("max_total_epochs", self.max_actions * self.max_budget),
            "hpo.budget.max_total_epochs",
        )
        raw_gpu_seconds = budget.get("max_gpu_seconds")
        self.max_gpu_seconds = (
            self._positive_float(raw_gpu_seconds, "hpo.budget.max_gpu_seconds")
            if raw_gpu_seconds is not None
            else None
        )

        confirmation = self._mapping(value.get("confirmation", {}), "hpo.confirmation")
        self.confirmation_top_k = self._positive_int(
            confirmation.get("top_k", 1), "hpo.confirmation.top_k"
        )
        self.max_concurrency = self._positive_int(
            value.get("max_concurrency", 1), "hpo.max_concurrency"
        )
        self.components = self._mapping(value.get("components", {}), "hpo.components")

    @classmethod
    def from_experiment(cls, config: Mapping[str, Any]) -> AdaptiveOptimizerConfig:
        """Extract the optional top-level HPO block."""
        value = config.get("hpo", {})
        if not isinstance(value, Mapping):
            raise TypeError("hpo must be a mapping.")
        return cls(value, base=config)

    def resolve_resource_features(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        """Resolve generic candidate resource features from sampled or base configuration."""
        output: dict[str, Any] = {}
        for name, path in self.resource_feature_paths.items():
            if path in parameters:
                output[name] = parameters[path]
            else:
                value = self._dotted_value(self._base, path)
                if value is None:
                    raise ValueError(
                        f"Resource feature {name!r} path {path!r} is missing from the candidate."
                    )
                output[name] = value
        return output

    @classmethod
    def parse_bytes(cls, value: Any) -> int:
        """Parse a non-negative byte count or binary/decimal unit string."""
        if isinstance(value, bool):
            raise TypeError("Memory sizes cannot be booleans.")
        if isinstance(value, int):
            if value < 0:
                raise ValueError("Memory sizes cannot be negative.")
            return value
        if not isinstance(value, str) or (match := cls._BYTES.fullmatch(value)) is None:
            raise ValueError(f"Invalid memory size: {value!r}.")
        amount = float(match.group(1))
        unit = (match.group(2) or "b").lower()
        powers = {
            "b": 1,
            "kb": 1000,
            "mb": 1000**2,
            "gb": 1000**3,
            "tb": 1000**4,
            "kib": 1024,
            "mib": 1024**2,
            "gib": 1024**3,
            "tib": 1024**4,
        }
        return int(amount * powers[unit])

    def study_fingerprint(self, base: Mapping[str, Any]) -> str:
        """Hash scientific base/search semantics separately from infrastructure knobs."""
        scientific_hpo = {
            key: value
            for key, value in self.raw.items()
            if key not in {"memory", "max_concurrency", "budget"}
        }
        base_payload = {
            key: value
            for key, value in base.items()
            if key not in {"execution", "aggregation", "retention", "metadata", "hpo"}
        }
        payload = json.dumps(
            {"base": base_payload, "hpo": scientific_hpo},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _mapping(value: Any, label: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be a mapping.")
        return value

    @staticmethod
    def _positive_int(value: Any, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer.")
        return value

    @staticmethod
    def _non_negative_int(value: Any, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a non-negative integer.")
        return value

    @staticmethod
    def _positive_float(value: Any, label: str) -> float:
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{label} must be a positive finite number.")
        return number

    @staticmethod
    def _non_negative_float(value: Any, label: str) -> float:
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{label} must be a non-negative finite number.")
        return number

    @classmethod
    def _probability(cls, value: Any, label: str, *, include_zero: bool = False) -> float:
        number = cls._non_negative_float(value, label)
        if number > 1 or (not include_zero and number == 0):
            qualifier = "[0, 1]" if include_zero else "(0, 1]"
            raise ValueError(f"{label} must be in {qualifier}.")
        return number

    @staticmethod
    def _integer_sequence(value: Any, label: str, *, allow_empty: bool = False) -> tuple[int, ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError(f"{label} must be an integer sequence.")
        output = tuple(value)
        if (not output and not allow_empty) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in output
        ):
            raise ValueError(
                f"{label} must contain integers{'' if allow_empty else ' and be non-empty'}."
            )
        if len(output) != len(set(output)):
            raise ValueError(f"{label} cannot contain duplicates.")
        return output

    @staticmethod
    def _dotted_value(value: Mapping[str, Any], path: str) -> Any:
        current: Any = value
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return None
            current = current[part]
        return current
