"""Bounded, dependency-free explanations for a live adaptive study."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

_T = TypeVar("_T")


class StudyInsightAnalyzer:
    """Summarize marginal HPO evidence without claiming causal effects.

    The adaptive sampler remains the decision authority.  These diagnostics intentionally use
    simple candidate-level associations so a researcher can understand coverage and uncertainty
    without loading model checkpoints, seed logs or an optional statistics stack.
    """

    MAX_CANDIDATES = 1_000
    MAX_PARAMETERS = 64
    MAX_RELATIONSHIP_CANDIDATES = 96
    MAX_RELATIONSHIP_PARAMETERS = 24

    @classmethod
    def analyze(
        cls,
        candidates: Sequence[Mapping[str, Any]],
        objective: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return one compact JSON-shaped HPO interpretation snapshot."""
        composite = isinstance(objective.get("metrics"), Mapping)
        metric = "utility" if composite else str(objective.get("metric", ""))
        mode = str(objective.get("mode", "max"))
        if not metric or mode not in {"min", "max"}:
            return cls._empty(metric, mode, "Objective metadata is not available yet.")
        bounded = cls._bounded(candidates, cls.MAX_CANDIDATES)
        observations: list[tuple[Mapping[str, Any], float, bool]] = []
        pruning_observations: list[tuple[Mapping[str, Any], bool]] = []
        provisional = censored = failed = infeasible = 0
        for candidate in bounded:
            value = candidate.get("selection_objective")
            parameters = candidate.get("parameters", {})
            runs = [item for item in candidate.get("runs", ()) if isinstance(item, Mapping)]
            states = {str(item.get("state", "")) for item in runs}
            if isinstance(parameters, Mapping):
                scientific_states = [
                    str(item.get("state", ""))
                    for item in runs
                    if str(item.get("state", "")) in {"succeeded", "pruned"}
                ]
                if scientific_states:
                    # One candidate contributes one regional outcome irrespective of how many
                    # seeds it happened to receive. Any performance-pruned seed censors the
                    # candidate; otherwise completed seeds are positive survival evidence.
                    pruning_observations.append((parameters, "pruned" in scientific_states))
                elif not runs and candidate.get("state") == "pruned":
                    pruning_observations.append((parameters, True))
            if candidate.get("state") == "pruned" or (
                states and states <= {"pruned", "cancelled"} and "pruned" in states
            ):
                censored += 1
            elif candidate.get("state") == "infeasible":
                infeasible += 1
            elif "running" in states or "retrying" in states or "scheduled" in states:
                provisional += 1
            elif "failed" in states and "succeeded" not in states:
                failed += 1
            if (
                not isinstance(parameters, Mapping)
                or not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                continue
            terminal = bool(runs) and all(
                item.get("state") in {"succeeded", "failed", "pruned"} for item in runs
            )
            observations.append((parameters, float(value), terminal))
        names = sorted(
            {
                str(name)
                for parameters in (
                    *(value[0] for value in observations),
                    *(value[0] for value in pruning_observations),
                )
                for name in parameters
            }
        )
        truncated_parameters = max(0, len(names) - cls.MAX_PARAMETERS)
        insights = [
            cls._parameter(name, observations, mode=mode) for name in names[: cls.MAX_PARAMETERS]
        ]
        relationship_names = [
            str(value["parameter"])
            for value in sorted(
                insights,
                key=lambda value: (
                    -float(value.get("priority", 0)),
                    str(value.get("parameter", "")),
                ),
            )[: cls.MAX_RELATIONSHIP_PARAMETERS]
        ]
        relationship_observations = cls._bounded(observations, cls.MAX_RELATIONSHIP_CANDIDATES)
        relationships = cls._relationships(relationship_names, relationship_observations)
        enriched: list[dict[str, Any]] = []
        for value in insights:
            name = str(value["parameter"])
            joint = relationships.get(name, [])
            contextual = [
                relationship
                for relationship in joint
                if relationship.get("status") == "predictive"
                and float(relationship.get("gain", 0.0)) > 0.05
            ]
            contextual.sort(key=lambda item: float(item.get("gain", 0.0)), reverse=True)
            interaction_context = (
                "No global independent preference; the observed response depends on "
                + ", ".join(str(item.get("parameter")) for item in contextual[:3])
                + "."
                if contextual and value.get("status") != "actionable"
                else None
            )
            enriched.append(
                {
                    **value,
                    "response": cls._response(name, observations, mode=mode),
                    "joint_relationships": joint,
                    "interaction_context": interaction_context,
                    "higher_order_caution": (
                        "Pairwise summaries are incomplete; higher-order/context-dependent "
                        "interactions may remain in the joint surrogate."
                    ),
                    "pruning_signal": cls._pruning_signal(name, pruning_observations),
                }
            )
        insights = enriched
        actionable = [value for value in insights if value["status"] == "actionable"]
        selected = max(insights, key=lambda value: float(value["priority"]), default=None)
        complete = sum(done for _parameters, _value, done in observations)
        return {
            "analysis_version": 4,
            "analysis_kind": "marginal-and-pairwise-descriptive-diagnostics",
            "objective": {
                "metric": metric,
                "mode": mode,
                "constraints": dict(objective.get("constraints", {})),
                "kind": "composite-utility" if composite else "scalar",
                "metrics": dict(objective.get("metrics", {})) if composite else {},
                "aggregation": objective.get("aggregation") if composite else None,
            },
            "candidate_observations": len(observations),
            "terminal_candidate_observations": complete,
            "provisional_candidate_observations": provisional,
            "censored_pruned_candidates": censored,
            "failed_candidates": failed,
            "infeasible_candidates": infeasible,
            "sampled_candidate_limit": cls.MAX_CANDIDATES,
            "truncated_candidates": max(0, len(candidates) - len(bounded)),
            "truncated_parameters": truncated_parameters,
            "relationship_parameter_limit": cls.MAX_RELATIONSHIP_PARAMETERS,
            "relationship_candidate_limit": cls.MAX_RELATIONSHIP_CANDIDATES,
            "status": (
                "insufficient"
                if len(observations) < 3
                else "actionable"
                if actionable
                else "learning"
            ),
            "next_question": (
                {
                    "parameter": selected["parameter"],
                    "suggestion": selected["recommendation"],
                    "reason": selected["conclusion"],
                }
                if selected is not None
                else None
            ),
            "parameters": insights,
            "interaction_matrix": cls._interaction_matrix(relationship_names, relationships),
            "decision_model": (
                "Controller proposals use a joint mixed-space GP (or multivariate k-NN "
                "fallback). Per-parameter panels are marginal explanations only."
            ),
            "caveat": (
                "Exploratory candidate-level associations, not causal effects: correlated "
                "hyperparameters, "
                "conditional spaces, unequal seed counts and partial fidelities can confound "
                "them. Pruned partial curves are censored: they are not fabricated into completed "
                "objectives. Their candidate-level survival probability and uncertainty softly "
                "adjust acquisition without treating operational failures as poor science. The "
                "multivariate adaptive sampler remains the "
                "decision authority."
            ),
        }

    @classmethod
    def _parameter(
        cls,
        name: str,
        observations: Sequence[tuple[Mapping[str, Any], float, bool]],
        *,
        mode: str,
    ) -> dict[str, Any]:
        rows = [
            (parameters.get(name, _INACTIVE), value) for parameters, value, _done in observations
        ]
        present = [value for value, _objective in rows if value is not _INACTIVE]
        numeric = bool(present) and all(
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in present
        )
        return (
            cls._numeric(name, rows, mode=mode)
            if numeric
            else cls._categorical(name, rows, mode=mode)
        )

    @classmethod
    def _numeric(
        cls,
        name: str,
        rows: Sequence[tuple[Any, float]],
        *,
        mode: str,
    ) -> dict[str, Any]:
        values = [
            (float(value), objective if mode == "max" else -objective)
            for value, objective in rows
            if value is not _INACTIVE
        ]
        distinct = sorted({value for value, _objective in values})
        inactive = sum(value is _INACTIVE for value, _objective in rows)
        if len(values) < 2 or len(distinct) < 2:
            return cls._insufficient(
                name,
                "numeric",
                len(values),
                len(distinct),
                inactive,
                "Too few distinct completed candidates to estimate a marginal direction.",
            )
        x = [value for value, _objective in values]
        y = [objective for _value, objective in values]
        correlation = cls._spearman(x, y)
        confidence = cls._association_confidence(correlation, len(values))
        outcome_scale = max(statistics.pstdev(y), 1e-12)
        ordered = sorted(values)
        middle = len(ordered) // 2
        low = ordered[:middle]
        high = ordered[middle:]
        contrast = (
            statistics.fmean(item[1] for item in high) - statistics.fmean(item[1] for item in low)
        ) / outcome_scale
        threshold = cls._threshold(ordered, outcome_scale)
        direction = "higher" if correlation > 0 else "lower"
        best_value = max(values, key=lambda item: item[1])[0]
        effect = max(abs(correlation), min(2.0, abs(contrast)) / 2)
        if abs(correlation) >= 0.35 and confidence >= 0.6:
            conclusion = (
                f"{direction.capitalize()} values appear associated with a better objective "
                f"(rank correlation {correlation:+.2f})."
            )
            edge = max(distinct) if direction == "higher" else min(distinct)
            recommendation = (
                f"Resolve the {direction}-value region near {_number(edge)} while varying other "
                "parameters as little as possible."
            )
            status = "actionable"
        elif threshold is not None and float(threshold["standardized_effect"]) >= 0.5:
            side = str(threshold["better_side"])
            conclusion = (
                f"A possible response change appears around {_number(threshold['value'])}; "
                f"the {side} side currently looks better."
            )
            recommendation = (
                f"Sample matched candidates on both sides of {_number(threshold['value'])} to "
                "check whether this apparent threshold is real."
            )
            status = "actionable" if len(values) >= 6 else "learning"
        else:
            conclusion = (
                "No stable marginal low/high tendency is visible yet; the joint sampler may "
                "still detect interactions with other hyperparameters."
            )
            recommendation = (
                "Add candidates across the observed range while holding stronger parameters "
                "similar; an interaction may be hiding the marginal effect."
            )
            status = "learning"
        return {
            "parameter": name,
            "kind": "numeric",
            "status": status,
            "observations": len(values),
            "distinct_values": len(distinct),
            "inactive_observations": inactive,
            "observed_range": [min(distinct), max(distinct)],
            "best_observed_value": best_value,
            "rank_correlation": correlation,
            "confidence": confidence,
            "confidence_label": cls._confidence_label(confidence, len(values)),
            "standardized_low_high_effect": contrast,
            "possible_threshold": threshold,
            "conclusion": conclusion,
            "recommendation": recommendation,
            "priority": cls._priority(effect, confidence, len(values)),
        }

    @classmethod
    def _categorical(
        cls,
        name: str,
        rows: Sequence[tuple[Any, float]],
        *,
        mode: str,
    ) -> dict[str, Any]:
        grouped: dict[str, list[float]] = {}
        labels: dict[str, Any] = {}
        for value, objective in rows:
            label = "<inactive>" if value is _INACTIVE else cls._category(value)
            grouped.setdefault(label, []).append(objective if mode == "max" else -objective)
            labels[label] = None if value is _INACTIVE else value
        summaries = sorted(
            (
                {
                    "value": labels[label],
                    "label": label,
                    "observations": len(values),
                    "objective_mean": (
                        statistics.fmean(values) if mode == "max" else -statistics.fmean(values)
                    ),
                    "normalized_mean": statistics.fmean(values),
                }
                for label, values in grouped.items()
            ),
            key=lambda value: (-float(value["normalized_mean"]), str(value["label"])),
        )
        if len(summaries) < 2 or len(rows) < 2:
            return cls._insufficient(
                name,
                "categorical",
                len(rows),
                len(summaries),
                len(grouped.get("<inactive>", ())),
                "Too few categories or candidate observations to compare levels.",
            )
        all_outcomes = [value for values in grouped.values() for value in values]
        scale = max(statistics.pstdev(all_outcomes), 1e-12)
        difference = float(summaries[0]["normalized_mean"]) - float(summaries[1]["normalized_mean"])
        effect = abs(difference) / scale
        confidence = 1 - math.exp(-max(1, len(rows) - 2) * effect**2 / 4)
        confidence = min(0.99, max(0.0, confidence))
        best = summaries[0]
        weakest = min(summaries, key=lambda value: int(value["observations"]))
        if effect >= 0.35 and confidence >= 0.6 and len(rows) >= 6:
            conclusion = (
                f"{best['label']} currently appears best; its advantage over the next level is "
                f"{effect:.2f} pooled outcome standard deviations."
            )
            recommendation = (
                f"Confirm {best['label']} against {weakest['label']} with matched surrounding "
                "parameters and additional seeds."
            )
            status = "actionable"
        else:
            conclusion = "No category has a convincing marginal advantage yet."
            recommendation = (
                f"Add evidence for the least represented level {weakest['label']} using matched "
                "surrounding parameters."
            )
            status = "learning"
        public_groups = [
            {key: value for key, value in summary.items() if key != "normalized_mean"}
            for summary in summaries[:12]
        ]
        return {
            "parameter": name,
            "kind": "categorical",
            "status": status,
            "observations": len(rows),
            "distinct_values": len(summaries),
            "inactive_observations": len(grouped.get("<inactive>", ())),
            "groups": public_groups,
            "groups_truncated": max(0, len(summaries) - len(public_groups)),
            "best_observed_value": best["value"],
            "confidence": confidence,
            "confidence_label": cls._confidence_label(confidence, len(rows)),
            "standardized_effect": effect,
            "conclusion": conclusion,
            "recommendation": recommendation,
            "priority": cls._priority(effect, confidence, len(rows)),
        }

    @classmethod
    def _response(
        cls,
        name: str,
        observations: Sequence[tuple[Mapping[str, Any], float, bool]],
        *,
        mode: str,
    ) -> dict[str, Any]:
        """Return bounded points for a visual marginal-response panel."""
        rows = [
            (parameters.get(name, _INACTIVE), value) for parameters, value, _done in observations
        ]
        present = [row for row in rows if row[0] is not _INACTIVE]
        numeric = bool(present) and all(
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value, _objective in present
        )
        if numeric:
            ordered = sorted((float(value), objective) for value, objective in present)
            bins = min(12, len(ordered))
            points: list[dict[str, Any]] = []
            for index in range(bins):
                start = round(index * len(ordered) / bins)
                end = round((index + 1) * len(ordered) / bins)
                group = ordered[start:end]
                if not group:
                    continue
                points.append(
                    {
                        "parameter": statistics.fmean(item[0] for item in group),
                        "objective": statistics.fmean(item[1] for item in group),
                        "samples": len(group),
                    }
                )
            return {"kind": "numeric-binned", "mode": mode, "points": points}
        grouped: dict[str, list[float]] = {}
        for value, objective in rows:
            label = "<inactive>" if value is _INACTIVE else cls._category(value)
            grouped.setdefault(label, []).append(objective)
        return {
            "kind": "categorical",
            "mode": mode,
            "points": [
                {
                    "label": label,
                    "objective": statistics.fmean(values),
                    "samples": len(values),
                }
                for label, values in sorted(grouped.items())[:16]
            ],
        }

    @classmethod
    def _relationships(
        cls,
        names: Sequence[str],
        observations: Sequence[tuple[Mapping[str, Any], float, bool]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Estimate bounded joint predictive gain without claiming causal interaction.

        Leave-one-out k-NN compares each pair against its best one-parameter predictor. A positive
        gain means the pair predicts held-out objectives better than either marginal alone. This is
        deliberately a descriptive diagnostic; the actual GP/k-NN acquisition remains authoritative.
        """
        output: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                detail = cls._joint_gain(left, right, observations)
                if detail is None:
                    continue
                output[left].append({"parameter": right, **detail})
                output[right].append({"parameter": left, **detail})
        for name in output:
            output[name].sort(
                key=lambda value: (
                    -float(value.get("gain", 0.0)),
                    -int(value["observations"]),
                    str(value["parameter"]),
                )
            )
            output[name] = output[name][:16]
        return output

    @classmethod
    def _joint_gain(
        cls,
        left: str,
        right: str,
        observations: Sequence[tuple[Mapping[str, Any], float, bool]],
    ) -> dict[str, Any] | None:
        rows = [
            (parameters.get(left, _INACTIVE), parameters.get(right, _INACTIVE), value)
            for parameters, value, _done in observations
        ]
        if len(rows) < 2:
            return None
        left_values, right_values = [row[0] for row in rows], [row[1] for row in rows]
        left_distinct = len({cls._category(value) for value in left_values})
        right_distinct = len({cls._category(value) for value in right_values})
        joint_cells = len(
            {
                (cls._category(left_value), cls._category(right_value))
                for left_value, right_value, _objective in rows
            }
        )
        if len(rows) < 3 or left_distinct < 2 or right_distinct < 2:
            return {
                "status": "coverage-only",
                "gain": 0.0,
                "observations": len(rows),
                "left_values": left_distinct,
                "right_values": right_distinct,
                "joint_cells": joint_cells,
                "confidence": 0.0,
                "confidence_label": "low",
            }
        left_distance = cls._distance_function(left_values)
        right_distance = cls._distance_function(right_values)
        targets = [row[2] for row in rows]

        def error(use_left: bool, use_right: bool) -> float:
            predictions: list[float] = []
            for index, row in enumerate(rows):
                neighbours: list[tuple[float, float]] = []
                for other, candidate in enumerate(rows):
                    if other == index:
                        continue
                    components = []
                    if use_left:
                        components.append(left_distance(row[0], candidate[0]))
                    if use_right:
                        components.append(right_distance(row[1], candidate[1]))
                    distance = math.sqrt(
                        sum(value * value for value in components) / len(components)
                    )
                    neighbours.append((distance, candidate[2]))
                nearest = sorted(neighbours, key=lambda item: item[0])[: min(3, len(neighbours))]
                weights = [1.0 / max(0.05, distance) for distance, _value in nearest]
                predictions.append(
                    sum(weight * item[1] for weight, item in zip(weights, nearest, strict=True))
                    / sum(weights)
                )
            return math.sqrt(
                statistics.fmean(
                    (target - predicted) ** 2
                    for target, predicted in zip(targets, predictions, strict=True)
                )
            )

        left_error, right_error, joint_error = (
            error(True, False),
            error(False, True),
            error(True, True),
        )
        scale = max(statistics.pstdev(targets), 1e-12)
        gain = max(0.0, (min(left_error, right_error) - joint_error) / scale)
        confidence = min(0.99, 1 - math.exp(-max(0, len(rows) - 4) * gain**2 / 2))
        return {
            "status": "predictive",
            "gain": gain,
            "observations": len(rows),
            "left_values": left_distinct,
            "right_values": right_distinct,
            "joint_cells": joint_cells,
            "confidence": confidence,
            "confidence_label": cls._confidence_label(confidence, len(rows)),
            "joint_rmse": joint_error,
            "best_marginal_rmse": min(left_error, right_error),
        }

    @classmethod
    def _pruning_signal(
        cls,
        name: str,
        observations: Sequence[tuple[Mapping[str, Any], bool]],
    ) -> dict[str, Any]:
        """Summarize terminal pruning as censored evidence, never as a fake objective."""
        rows = [(parameters.get(name, _INACTIVE), pruned) for parameters, pruned in observations]
        if not rows:
            return {
                "status": "unavailable",
                "observations": 0,
                "pruned": 0,
                "pruned_rate": None,
                "groups": [],
            }
        present = [value for value, _pruned in rows if value is not _INACTIVE]
        numeric = bool(present) and all(
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in present
        )
        groups: list[dict[str, Any]] = []
        if numeric:
            ordered = sorted(
                (float(value), pruned) for value, pruned in rows if value is not _INACTIVE
            )
            bins = min(8, max(1, len({value for value, _pruned in ordered})))
            for index in range(bins):
                start = round(index * len(ordered) / bins)
                end = round((index + 1) * len(ordered) / bins)
                group = ordered[start:end]
                if group:
                    pruned_count = sum(pruned for _value, pruned in group)
                    groups.append(
                        {
                            "value": statistics.fmean(value for value, _pruned in group),
                            "observations": len(group),
                            "pruned": pruned_count,
                            "pruned_rate": pruned_count / len(group),
                            **cls._rate_interval(pruned_count, len(group)),
                        }
                    )
        else:
            grouped: dict[str, list[bool]] = {}
            for value, pruned in rows:
                label = "<inactive>" if value is _INACTIVE else cls._category(value)
                grouped.setdefault(label, []).append(pruned)
            for label, values in sorted(grouped.items())[:16]:
                pruned_count = sum(values)
                groups.append(
                    {
                        "label": label,
                        "observations": len(values),
                        "pruned": pruned_count,
                        "pruned_rate": pruned_count / len(values),
                        **cls._rate_interval(pruned_count, len(values)),
                    }
                )
        pruned_total = sum(pruned for _value, pruned in rows)
        return {
            "status": "descriptive",
            "kind": "numeric-binned" if numeric else "categorical",
            "observations": len(rows),
            "pruned": pruned_total,
            "pruned_rate": pruned_total / len(rows),
            **cls._rate_interval(pruned_total, len(rows)),
            "groups": groups,
            "caveat": (
                "Pruning is a censored negative signal, not a completed objective value. Rates "
                "may also reflect fidelity and asynchronous scheduling."
            ),
        }

    @staticmethod
    def _rate_interval(events: int, samples: int) -> dict[str, float]:
        """Return a smoothed 90% Beta-posterior interval for sparse pruning rates."""
        alpha = 1.0 + events
        beta = 1.0 + samples - events
        total = alpha + beta
        mean = alpha / total
        deviation = math.sqrt(alpha * beta / (total * total * (total + 1.0)))
        return {
            "smoothed_rate": mean,
            "rate_lower": max(0.0, mean - 1.645 * deviation),
            "rate_upper": min(1.0, mean + 1.645 * deviation),
        }

    @classmethod
    def _distance_function(cls, values: Sequence[Any]) -> Callable[[Any, Any], float]:
        numeric = all(
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in values
            if value is not _INACTIVE
        )
        present = [float(value) for value in values if value is not _INACTIVE and numeric]
        if numeric and present:
            low, high = min(present), max(present)
            width = max(high - low, 1e-12)

            def distance(left: Any, right: Any) -> float:
                if left is _INACTIVE or right is _INACTIVE:
                    return 0.0 if left is right else 1.0
                return min(1.0, abs(float(left) - float(right)) / width)

            return distance

        def categorical(left: Any, right: Any) -> float:
            return 0.0 if cls._category(left) == cls._category(right) else 1.0

        return categorical

    @staticmethod
    def _interaction_matrix(
        names: Sequence[str], relationships: Mapping[str, Sequence[Mapping[str, Any]]]
    ) -> dict[str, Any]:
        bounded = list(names[:12])
        lookup = {
            tuple(sorted((name, str(item.get("parameter", ""))))): item.get("gain")
            for name, values in relationships.items()
            for item in values
        }
        return {
            "parameters": bounded,
            "values": [
                [
                    0.0 if left == right else lookup.get(tuple(sorted((left, right))))
                    for right in bounded
                ]
                for left in bounded
            ],
            "measure": "leave-one-out joint predictive gain over best marginal (objective SD)",
        }

    @staticmethod
    def _threshold(
        ordered: Sequence[tuple[float, float]], outcome_scale: float
    ) -> dict[str, Any] | None:
        best: tuple[float, float, str] | None = None
        for index in range(2, len(ordered) - 1):
            if ordered[index - 1][0] == ordered[index][0]:
                continue
            left = statistics.fmean(value for _parameter, value in ordered[:index])
            right = statistics.fmean(value for _parameter, value in ordered[index:])
            effect = abs(right - left) / outcome_scale
            threshold = (ordered[index - 1][0] + ordered[index][0]) / 2
            candidate = (effect, threshold, "higher" if right > left else "lower")
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None:
            return None
        return {"value": best[1], "standardized_effect": best[0], "better_side": best[2]}

    @staticmethod
    def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
        first, second = _ranks(left), _ranks(right)
        first_mean, second_mean = statistics.fmean(first), statistics.fmean(second)
        numerator = sum(
            (x - first_mean) * (y - second_mean) for x, y in zip(first, second, strict=True)
        )
        denominator = math.sqrt(
            sum((x - first_mean) ** 2 for x in first) * sum((y - second_mean) ** 2 for y in second)
        )
        return numerator / denominator if denominator else 0.0

    @staticmethod
    def _association_confidence(correlation: float, samples: int) -> float:
        # A deliberately conservative monotone evidence score, not a hypothesis-test p-value.
        evidence = max(0, samples - 2) * correlation**2 / 2
        return min(0.99, max(0.0, 1 - math.exp(-evidence)))

    @staticmethod
    def _priority(effect: float, confidence: float, samples: int) -> float:
        return min(10.0, max(0.0, effect) * (1 - confidence) + 1 / math.sqrt(samples))

    @staticmethod
    def _confidence_label(value: float, samples: int) -> str:
        # Evidence labels should not oscillate into high confidence from a tiny early sample.
        return (
            "high"
            if samples >= 12 and value >= 0.85
            else "medium"
            if samples >= 6 and value >= 0.6
            else "low"
        )

    @staticmethod
    def _category(value: Any) -> str:
        if value is _INACTIVE:
            return "<inactive>"
        if isinstance(value, str):
            return value
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)

    @staticmethod
    def _insufficient(
        name: str,
        kind: str,
        observations: int,
        distinct: int,
        inactive: int,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "parameter": name,
            "kind": kind,
            "status": "insufficient",
            "observations": observations,
            "distinct_values": distinct,
            "inactive_observations": inactive,
            "confidence": 0.0,
            "confidence_label": "low",
            "conclusion": reason,
            "recommendation": "Collect more distinct candidate observations for this parameter.",
            "priority": 1 / math.sqrt(max(1, observations)),
        }

    @staticmethod
    def _bounded(candidates: Sequence[_T], limit: int) -> list[_T]:
        if len(candidates) <= limit:
            return list(candidates)
        step = (len(candidates) - 1) / (limit - 1)
        return [candidates[round(index * step)] for index in range(limit)]

    @staticmethod
    def _empty(metric: str, mode: str, reason: str) -> dict[str, Any]:
        return {
            "analysis_version": 4,
            "objective": {"metric": metric, "mode": mode},
            "candidate_observations": 0,
            "terminal_candidate_observations": 0,
            "provisional_candidate_observations": 0,
            "censored_pruned_candidates": 0,
            "failed_candidates": 0,
            "infeasible_candidates": 0,
            "relationship_parameter_limit": StudyInsightAnalyzer.MAX_RELATIONSHIP_PARAMETERS,
            "relationship_candidate_limit": StudyInsightAnalyzer.MAX_RELATIONSHIP_CANDIDATES,
            "status": "insufficient",
            "next_question": None,
            "parameters": [],
            "interaction_matrix": {"parameters": [], "values": [], "measure": ""},
            "decision_model": None,
            "caveat": reason,
        }


_INACTIVE = object()


def _ranks(values: Sequence[float]) -> list[float]:
    """Return average ranks with deterministic tie handling."""
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = (start + 1 + end) / 2
        for index, _value in ordered[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _number(value: Any) -> str:
    return f"{float(value):.6g}" if isinstance(value, int | float) else str(value)


__all__ = ["StudyInsightAnalyzer"]
