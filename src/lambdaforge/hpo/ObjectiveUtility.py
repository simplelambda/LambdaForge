"""One authoritative, reproducible definition of HPO scientific quality."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeGuard, cast

UTILITY_METRIC = "__lambdaforge_utility__"
_AGGREGATIONS = frozenset({"weighted_mean", "geometric", "chebyshev"})


@dataclass(frozen=True, slots=True)
class UtilityMetric:
    """One fixed-range component of a composite scientific utility."""

    name: str
    mode: str
    weight: float
    low: float
    high: float

    def quality(self, value: float) -> float:
        raw = (
            (float(value) - self.low) / (self.high - self.low)
            if self.mode == "max"
            else (self.high - float(value)) / (self.high - self.low)
        )
        return min(1.0, max(0.0, raw))


class ObjectiveUtility:
    """Evaluate legacy scalar objectives and optional fixed-range composite utilities."""

    def __init__(self, objective: Mapping[str, Any]) -> None:
        self.objective = objective
        raw_metrics = objective.get("metrics")
        self.composite = isinstance(raw_metrics, Mapping)
        self.aggregation = str(objective.get("aggregation", "weighted_mean"))
        if self.composite:
            assert isinstance(raw_metrics, Mapping)
            self.metrics = tuple(
                UtilityMetric(
                    str(name),
                    str(rule["mode"]),
                    float(rule["weight"]),
                    float(rule["range"][0]),
                    float(rule["range"][1]),
                )
                for name, rule in raw_metrics.items()
                if isinstance(rule, Mapping)
            )
            self.metric = UTILITY_METRIC
            self.mode = "max"
        else:
            self.metric = str(objective["metric"])
            self.mode = str(objective["mode"])
            self.metrics = ()

    @property
    def required_metrics(self) -> tuple[str, ...]:
        return tuple(value.name for value in self.metrics) if self.composite else (self.metric,)

    def evaluate(self, values: Mapping[str, Any]) -> dict[str, Any] | None:
        """Evaluate one real checkpoint, returning raw and normalized audit evidence."""
        if not self.composite:
            raw = values.get(self.metric)
            if not _finite(raw):
                return None
            value = float(raw)
            return {
                "value": value,
                "raw_metrics": {self.metric: value},
                "normalized_metrics": {},
                "contributions": {},
                "weights": {},
                "aggregation": "legacy",
            }
        raw_values: dict[str, float] = {}
        qualities: dict[str, float] = {}
        for metric in self.metrics:
            raw = values.get(metric.name)
            if not _finite(raw):
                return None
            raw_values[metric.name] = float(raw)
            qualities[metric.name] = metric.quality(float(raw))
        weights = {metric.name: metric.weight for metric in self.metrics}
        if self.aggregation == "weighted_mean":
            contributions = {
                metric.name: metric.weight * qualities[metric.name] for metric in self.metrics
            }
            utility = sum(contributions.values())
        elif self.aggregation == "geometric":
            active_metrics = tuple(metric for metric in self.metrics if metric.weight > 0)
            if any(qualities[metric.name] <= 0 for metric in active_metrics):
                utility = 0.0
            else:
                utility = math.exp(
                    sum(
                        metric.weight * math.log(qualities[metric.name])
                        for metric in active_metrics
                    )
                )
            contributions = {
                metric.name: qualities[metric.name] ** metric.weight for metric in self.metrics
            }
        else:
            maximum_weight = max((metric.weight for metric in self.metrics), default=1.0)
            regrets = {
                metric.name: (metric.weight / maximum_weight) * (1.0 - qualities[metric.name])
                for metric in self.metrics
            }
            utility = 1.0 - max(regrets.values(), default=0.0)
            contributions = {name: -value for name, value in regrets.items()}
        return {
            "value": min(1.0, max(0.0, utility)),
            "raw_metrics": raw_values,
            "normalized_metrics": qualities,
            "contributions": contributions,
            "weights": weights,
            "aggregation": self.aggregation,
        }

    def observation(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        fallback: Mapping[str, int | float],
    ) -> dict[str, Any] | None:
        """Select current and best complete real checkpoints from metric records."""
        wanted = {
            *self.required_metrics,
            *(str(name) for name in self.objective.get("constraints", {})),
        }
        by_step: dict[int | None, dict[str, tuple[int, float]]] = {}
        sequence = 0
        for record in records:
            raw = record.get("value")
            if not _finite(raw):
                continue
            key = (
                f"{record['split']}_{record['name']}"
                if record.get("split")
                else str(record.get("name", ""))
            )
            if key not in wanted:
                continue
            raw_step = record.get("step")
            step = (
                int(raw_step)
                if isinstance(raw_step, int) and not isinstance(raw_step, bool)
                else None
            )
            sequence += 1
            current = by_step.setdefault(step, {}).get(key)
            if current is None or sequence >= current[0]:
                by_step[step][key] = (sequence, float(raw))
        if (
            not self.composite
            and not by_step
            and all(_finite(fallback.get(name)) for name in self.required_metrics)
        ):
            by_step[None] = {
                name: (index, float(fallback[name]))
                for index, name in enumerate(wanted, 1)
                if _finite(fallback.get(name))
            }
        evaluations: list[tuple[int | None, dict[str, Any], Mapping[str, tuple[int, float]]]] = []
        for step, values in by_step.items():
            evaluated = self.evaluate({name: value for name, (_order, value) in values.items()})
            if evaluated is not None:
                evaluations.append((step, evaluated, values))
        if not evaluations:
            return None
        stepped = [value for value in evaluations if value[0] is not None]
        current_step, current_evaluation, _current_values = (
            max(stepped, key=lambda value: cast(int, value[0])) if stepped else evaluations[-1]
        )
        best_step, best_evaluation, best_values = (min if self.mode == "min" else max)(
            evaluations, key=lambda value: float(value[1]["value"])
        )
        constraints: dict[str, Any] = {}
        raw_constraints = self.objective.get("constraints", {})
        if isinstance(raw_constraints, Mapping):
            for name, raw_rule in raw_constraints.items():
                rule = raw_rule if isinstance(raw_rule, Mapping) else {}
                raw_value = best_values.get(str(name))
                selected = raw_value[1] if raw_value is not None else None
                constraints[str(name)] = {
                    "value": selected,
                    "step": best_step,
                    "min": rule.get("min"),
                    "max": rule.get("max"),
                    "satisfied": constraint_satisfied(selected, rule),
                }
        output = {
            "metric": self.metric,
            "mode": self.mode,
            "current": current_evaluation["value"],
            "current_step": current_step,
            "best": best_evaluation["value"],
            "best_step": best_step,
            "selection": (
                "best-complete-checkpoint" if self.composite else "best-observed-checkpoint"
            ),
        }
        if self.composite:
            output.update(
                {
                    "kind": "composite-utility",
                    "aggregation": self.aggregation,
                    "components": {
                        name: {
                            "raw": best_evaluation["raw_metrics"][name],
                            "normalized": best_evaluation["normalized_metrics"][name],
                            "weight": best_evaluation["weights"][name],
                            "contribution": best_evaluation["contributions"][name],
                        }
                        for name in self.required_metrics
                    },
                    "current_components": {
                        name: {
                            "raw": current_evaluation["raw_metrics"][name],
                            "normalized": current_evaluation["normalized_metrics"][name],
                            "weight": current_evaluation["weights"][name],
                            "contribution": current_evaluation["contributions"][name],
                        }
                        for name in self.required_metrics
                    },
                }
            )
        if constraints:
            output["constraints"] = constraints
            output["feasible"] = all(bool(value["satisfied"]) for value in constraints.values())
        return output

    @classmethod
    def normalize(cls, value: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and normalize an authored objective mapping."""
        allowed = {"metric", "mode", "metrics", "aggregation", "constraints"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown objective field(s): {sorted(unknown)}.")
        has_legacy = "metric" in value or "mode" in value
        has_composite = "metrics" in value or "aggregation" in value
        if has_legacy == has_composite:
            raise ValueError(
                "objective requires either metric/mode or metrics/aggregation, never both."
            )
        constraints = cls._constraints(value.get("constraints", {}))
        if has_legacy:
            if not {"metric", "mode"} <= set(value):
                raise ValueError("Legacy objective requires metric and mode.")
            metric = _nonempty(value["metric"], "objective.metric")
            mode = str(value["mode"]).lower()
            if mode not in {"min", "max"}:
                raise ValueError("objective.mode must be min or max.")
            if metric in constraints:
                raise ValueError(
                    f"objective.constraints cannot repeat the primary metric {metric!r}."
                )
            output: dict[str, Any] = {"metric": metric, "mode": mode}
        else:
            raw_metrics = value.get("metrics")
            if not isinstance(raw_metrics, Mapping) or len(raw_metrics) < 2:
                raise ValueError("objective.metrics must define at least two metric components.")
            aggregation = str(value.get("aggregation", "geometric")).lower()
            if aggregation not in _AGGREGATIONS:
                raise ValueError(
                    "objective.aggregation must be weighted_mean, geometric or chebyshev."
                )
            metrics: dict[str, dict[str, Any]] = {}
            total_weight = 0.0
            for raw_name, raw_rule in raw_metrics.items():
                name = _nonempty(raw_name, "objective.metrics metric")
                if not isinstance(raw_rule, Mapping):
                    raise TypeError(f"objective.metrics.{name} must be a mapping.")
                if set(raw_rule) - {"mode", "weight", "range"} or not {
                    "mode",
                    "weight",
                    "range",
                } <= set(raw_rule):
                    raise ValueError(
                        f"objective.metrics.{name} requires only mode, weight and range."
                    )
                mode = str(raw_rule["mode"]).lower()
                if mode not in {"min", "max"}:
                    raise ValueError(f"objective.metrics.{name}.mode must be min or max.")
                weight = raw_rule["weight"]
                if not _finite(weight) or float(weight) < 0:
                    raise ValueError(
                        f"objective.metrics.{name}.weight must be finite and non-negative."
                    )
                bounds = raw_rule["range"]
                if (
                    not isinstance(bounds, Sequence)
                    or isinstance(bounds, str | bytes)
                    or len(bounds) != 2
                    or not all(_finite(bound) for bound in bounds)
                    or float(bounds[0]) >= float(bounds[1])
                ):
                    raise ValueError(
                        f"objective.metrics.{name}.range must be fixed finite [low, high]."
                    )
                total_weight += float(weight)
                metrics[name] = {
                    "mode": mode,
                    "weight": float(weight),
                    "range": [float(bounds[0]), float(bounds[1])],
                }
            if total_weight <= 0:
                raise ValueError("objective.metrics weights must have a positive sum.")
            for rule in metrics.values():
                rule["weight"] = float(rule["weight"]) / total_weight
            repeated = set(metrics).intersection(constraints)
            if repeated:
                raise ValueError(
                    f"objective.constraints cannot repeat utility components: {sorted(repeated)}."
                )
            output = {
                "metric": UTILITY_METRIC,
                "mode": "max",
                "metrics": metrics,
                "aggregation": aggregation,
            }
        if constraints:
            output["constraints"] = constraints
        return output

    @staticmethod
    def _constraints(value: Any) -> dict[str, dict[str, float]]:
        if not isinstance(value, Mapping):
            raise TypeError("objective.constraints must map metric names to min/max bounds.")
        constraints: dict[str, dict[str, float]] = {}
        for raw_name, raw_rule in value.items():
            name = _nonempty(raw_name, "objective.constraints metric")
            if not isinstance(raw_rule, Mapping) or not raw_rule:
                raise TypeError(f"objective.constraints.{name} must contain min and/or max.")
            if set(raw_rule) - {"min", "max"}:
                raise ValueError(f"objective.constraints.{name} accepts only min and max bounds.")
            rule: dict[str, float] = {}
            for bound in ("min", "max"):
                if bound in raw_rule:
                    if not _finite(raw_rule[bound]):
                        raise TypeError(
                            f"objective.constraints.{name}.{bound} must be a finite number."
                        )
                    rule[bound] = float(raw_rule[bound])
            if not rule:
                raise ValueError(f"objective.constraints.{name} requires min and/or max.")
            if rule.get("min", -math.inf) > rule.get("max", math.inf):
                raise ValueError(f"objective.constraints.{name} min cannot exceed max.")
            constraints[name] = rule
        return constraints


def pareto_front(
    candidates: Sequence[Mapping[str, Any]], objective: Mapping[str, Any]
) -> tuple[int, ...]:
    """Return non-dominated candidate IDs using raw composite metrics."""
    evaluator = ObjectiveUtility(objective)
    if not evaluator.composite:
        return ()
    rows: list[tuple[int, Mapping[str, float]]] = []
    for candidate in candidates:
        trial = candidate.get("trial")
        raw = candidate.get("raw_metrics")
        if (
            isinstance(trial, int)
            and isinstance(raw, Mapping)
            and all(_finite(raw.get(metric.name)) for metric in evaluator.metrics)
        ):
            rows.append((trial, {name: float(value) for name, value in raw.items()}))

    def dominates(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
        comparisons = [
            (
                left[metric.name] >= right[metric.name]
                if metric.mode == "max"
                else left[metric.name] <= right[metric.name]
            )
            for metric in evaluator.metrics
        ]
        strict = [
            (
                left[metric.name] > right[metric.name]
                if metric.mode == "max"
                else left[metric.name] < right[metric.name]
            )
            for metric in evaluator.metrics
        ]
        return all(comparisons) and any(strict)

    return tuple(
        trial
        for trial, values in rows
        if not any(
            other != trial and dominates(other_values, values) for other, other_values in rows
        )
    )


def constraint_satisfied(value: Any, rule: Mapping[str, Any]) -> bool:
    """Evaluate one explicit outcome guardrail; missing evidence fails closed."""
    if not _finite(value):
        return False
    numeric = float(value)
    minimum, maximum = rule.get("min"), rule.get("max")
    return not (_finite(minimum) and numeric < float(minimum)) and not (
        _finite(maximum) and numeric > float(maximum)
    )


def _finite(value: Any) -> TypeGuard[int | float]:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


def _nonempty(value: Any, field: str) -> str:
    selected = str(value).strip()
    if not selected:
        raise ValueError(f"{field} cannot be empty.")
    return selected


__all__ = [
    "ObjectiveUtility",
    "UTILITY_METRIC",
    "UtilityMetric",
    "constraint_satisfied",
    "pareto_front",
]
