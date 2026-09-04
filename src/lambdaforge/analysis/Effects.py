"""Bounded surrogate diagnostics, effects and interactions for Study Analysis."""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from lambdaforge.analysis.Ordering import best, supported_region
from lambdaforge.analysis.SearchSpace import (
    INACTIVE,
    assign_values,
    build_space,
    grid,
    sample_point,
    valid_point,
)


def infer_space(
    candidates: Sequence[Mapping[str, Any]], authored: Mapping[str, Any] | None = None
) -> dict[str, dict[str, Any]]:
    """Infer a conservative mixed space while preserving exact authored ``when`` rules."""
    return build_space(candidates, authored)


class AnalysisSurrogate:
    """Small deterministic mixed-space k-NN surrogate with descriptive uncertainty."""

    def __init__(
        self,
        candidates: Sequence[Mapping[str, Any]],
        space: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.rows = [
            (dict(value.get("parameters", {})), float(value["mean"]))
            for value in candidates
            if _finite(value.get("mean")) and valid_point(value.get("parameters", {}), space)
        ]
        self.space = {str(name): dict(rule) for name, rule in space.items()}

    def predict(self, parameters: Mapping[str, Any]) -> tuple[float, float, float]:
        if not self.rows:
            return float("nan"), float("nan"), float("inf")
        distances = sorted(
            (mixed_distance(parameters, row, self.space), target) for row, target in self.rows
        )
        if distances[0][0] <= 1e-12:
            exact = [target for distance, target in distances if distance <= 1e-12]
            return statistics.fmean(exact), statistics.pstdev(exact) if len(exact) > 1 else 0.0, 0.0
        count = min(len(distances), max(2, int(math.sqrt(len(distances))) + 1))
        neighbours = distances[:count]
        weights = [1.0 / max(distance, 1e-9) ** 2 for distance, _target in neighbours]
        total = sum(weights)
        mean = (
            sum(
                weight * target
                for weight, (_distance, target) in zip(weights, neighbours, strict=True)
            )
            / total
        )
        variance = (
            sum(
                weight * (target - mean) ** 2
                for weight, (_distance, target) in zip(weights, neighbours, strict=True)
            )
            / total
        )
        return mean, math.sqrt(max(0.0, variance)), neighbours[0][0]


def validate_surrogate(
    candidates: Sequence[Mapping[str, Any]], space: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    rows = [value for value in candidates if _finite(value.get("mean"))]
    if len(rows) < 3:
        return {
            "backend": "mixed-knn",
            "observations": len(rows),
            "quality": "insufficient",
            "rmse": None,
            "mae": None,
            "spearman": None,
            "interval_coverage_90": None,
            "validation": "leave-one-candidate-out",
            "validation_method": "leave-one-candidate-out",
            "fold_count": len(rows),
        }
    actual: list[float] = []
    predicted: list[float] = []
    covered = 0
    for index, held in enumerate(rows):
        model = AnalysisSurrogate(rows[:index] + rows[index + 1 :], space)
        mean, uncertainty, _support = model.predict(held.get("parameters", {}))
        actual.append(float(held["mean"]))
        predicted.append(mean)
        covered += (
            abs(float(held["mean"]) - mean) <= 1.645 * uncertainty if uncertainty > 0 else False
        )
    errors = [left - right for left, right in zip(actual, predicted, strict=True)]
    rmse = math.sqrt(statistics.fmean(error * error for error in errors))
    mae = statistics.fmean(abs(error) for error in errors)
    scale = statistics.pstdev(actual) if len(actual) > 1 else 0.0
    relative = rmse / max(scale, 1e-12)
    quality = (
        "good"
        if len(rows) >= 8 and relative <= 0.6
        else "moderate"
        if len(rows) >= 8 and relative <= 1.2
        else "poor"
        if len(rows) >= 8
        else "insufficient"
    )
    return {
        "backend": "mixed-knn-bootstrap",
        "observations": len(rows),
        "quality": quality,
        "rmse": rmse,
        "mae": mae,
        "spearman": _spearman(actual, predicted),
        "interval_coverage_90": covered / len(rows),
        "validation": "leave-one-candidate-out",
        "validation_method": "leave-one-candidate-out",
        "fold_count": len(rows),
    }


def analyze_effects(
    candidates: Sequence[Mapping[str, Any]],
    *,
    space: Mapping[str, Mapping[str, Any]],
    mode: str,
    fingerprint: str,
    provisional: bool,
    surrogate_diagnostics: Mapping[str, Any] | None = None,
    coverage_quality: str | None = None,
) -> dict[str, Any]:
    rows = [value for value in candidates if _finite(value.get("mean"))]
    if len(rows) < 2 or not space:
        return _empty_effects(len(rows))
    model = AnalysisSurrogate(rows, space)
    reference = reference_set(
        space,
        count=min(512 if provisional else 1024, max(128, len(rows) * 32)),
        fingerprint=fingerprint,
    )
    predictions = [model.predict(point)[0] for point in reference]
    total_variance = statistics.pvariance(predictions) if len(predictions) > 1 else 0.0
    main: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[Any]] = {}
    for name, rule in space.items():
        labels = [_bin(point.get(name, INACTIVE), rule) for point in reference]
        groups[name] = labels
        explained = _group_variance(labels, predictions)
        support = sum(name in row.get("parameters", {}) for row in rows)
        reliability = _reliability(
            len(rows),
            model_quality=str((surrogate_diagnostics or {}).get("quality", "insufficient")),
            support_fraction=support / len(rows),
            coverage_quality=coverage_quality,
        )
        main[name] = {
            "importance": explained / total_variance if total_variance > 0 else 0.0,
            "variance": explained,
            "support": support,
            "reliability": reliability,
            "reliability_components": {
                "observations": len(rows),
                "active_support": support,
                "surrogate_quality": (surrogate_diagnostics or {}).get("quality"),
                "coverage_quality": coverage_quality,
            },
            "interpretation": "predictive association, not a causal effect",
        }
    interactions: list[dict[str, Any]] = []
    names = list(space)
    pair_total = 0.0
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            labels = list(zip(groups[left], groups[right], strict=True))
            joint = _group_variance(labels, predictions)
            variance = max(0.0, joint - main[left]["variance"] - main[right]["variance"])
            pair_total += variance
            interactions.append(
                {
                    "left": left,
                    "right": right,
                    "importance": variance / total_variance if total_variance > 0 else 0.0,
                    "variance": variance,
                    "support": len(rows),
                    "reliability": _reliability(
                        len(rows),
                        model_quality=str(
                            (surrogate_diagnostics or {}).get("quality", "insufficient")
                        ),
                        support_fraction=1.0,
                        coverage_quality=coverage_quality,
                    ),
                }
            )
    interactions.sort(key=lambda value: float(value["importance"]), reverse=True)
    top = top_region_importance(rows, space=space, mode=mode)
    responses = {
        name: response_curve(
            name,
            rows,
            model=model,
            space=space,
            fingerprint=fingerprint,
            mode=mode,
        )
        for name in space
    }
    surfaces = {
        f"{item['left']}::{item['right']}": pair_surface(
            str(item["left"]),
            str(item["right"]),
            model=model,
            space=space,
            fingerprint=fingerprint,
        )
        for item in interactions[: min(6, len(interactions))]
    }
    return {
        "status": "provisional" if provisional else "final",
        "global_importance": main,
        "top_region_importance": top,
        "response_curves": responses,
        "interactions": interactions,
        "interaction_matrix": _interaction_matrix(names, interactions),
        "pairwise_surfaces": surfaces,
        "functional_variance": total_variance,
        "residual_higher_order_variance": max(
            0.0,
            total_variance - sum(float(value["variance"]) for value in main.values()) - pair_total,
        ),
        "reference_points": len(reference),
    }


def top_region_importance(
    candidates: Sequence[Mapping[str, Any]],
    *,
    space: Mapping[str, Mapping[str, Any]],
    mode: str,
) -> dict[str, Any]:
    ordered = sorted(candidates, key=lambda value: float(value["mean"]), reverse=mode == "max")
    top_count = max(5, math.ceil(len(ordered) * 0.1))
    if len(ordered) < 5:
        return {name: {"status": "insufficient", "observations": len(ordered)} for name in space}
    top = ordered[: min(top_count, len(ordered))]
    output: dict[str, Any] = {}
    for name, rule in space.items():
        all_labels = [
            _bin(value.get("parameters", {}).get(name, INACTIVE), rule) for value in ordered
        ]
        top_labels = [_bin(value.get("parameters", {}).get(name, INACTIVE), rule) for value in top]
        labels = sorted(set(all_labels) | set(top_labels), key=str)
        smooth = 1e-12
        p = [
            (top_labels.count(label) + smooth) / (len(top_labels) + smooth * len(labels))
            for label in labels
        ]
        q = [
            (all_labels.count(label) + smooth) / (len(all_labels) + smooth * len(labels))
            for label in labels
        ]
        output[name] = {
            "status": "available",
            "region_source": "observed-candidate-ranking",
            "importance": _jensen_shannon(p, q),
            "top_count": len(top),
            "total_count": len(ordered),
            "distribution_top": dict(zip(map(str, labels), p, strict=True)),
            "distribution_all": dict(zip(map(str, labels), q, strict=True)),
        }
    return output


def response_curve(
    name: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    model: AnalysisSurrogate,
    space: Mapping[str, Mapping[str, Any]],
    fingerprint: str,
    mode: str,
) -> dict[str, Any]:
    rule = space[name]
    if rule["kind"] == "categorical":
        levels = list(rule.get("values", ())) + ([INACTIVE] if rule.get("conditional") else [])
        raw: list[tuple[Any, float, float, int]] = []
        for level in levels:
            values = []
            uncertainties = []
            for candidate in candidates:
                parameters = dict(candidate.get("parameters", {}))
                assigned = assign_values(parameters, {name: level}, space)
                if assigned is None:
                    continue
                category_prediction, category_uncertainty, _distance = model.predict(assigned)
                values.append(category_prediction)
                uncertainties.append(category_uncertainty)
            if not values:
                continue
            raw.append(
                (
                    level,
                    statistics.fmean(values),
                    statistics.fmean(uncertainties),
                    sum(
                        candidate.get("parameters", {}).get(name, INACTIVE) == level
                        for candidate in candidates
                    ),
                )
            )
        center = statistics.fmean(value for _level, value, _uncertainty, _support in raw)
        return {
            "kind": "categorical-adjusted",
            "points": [
                {
                    "category": level,
                    "effect": value - center,
                    "predicted_objective": value,
                    "uncertainty": uncertainty,
                    "support_count": support,
                }
                for level, value, uncertainty, support in raw
            ],
        }
    values = sorted(
        float(candidate.get("parameters", {})[name])
        for candidate in candidates
        if name in candidate.get("parameters", {})
    )
    if len(set(values)) < 2:
        return {"kind": "numeric-ale", "status": "insufficient", "points": []}
    edges = _quantiles(values, min(10, len(set(values))))
    effects: list[float] = []
    support: list[int] = []
    estimates: list[float] = []
    spreads: list[float] = []
    for lower, upper in zip(edges, edges[1:], strict=False):
        members = [
            candidate
            for candidate in candidates
            if name in candidate.get("parameters", {})
            and lower <= float(candidate["parameters"][name]) <= upper
        ]
        differences = []
        upper_predictions = []
        upper_uncertainties = []
        for candidate in members:
            low_parameters = assign_values(candidate.get("parameters", {}), {name: lower}, space)
            high_parameters = assign_values(candidate.get("parameters", {}), {name: upper}, space)
            if low_parameters is None or high_parameters is None:
                continue
            low_value, _low_uncertainty, _ = model.predict(low_parameters)
            high_value, high_uncertainty, _ = model.predict(high_parameters)
            differences.append(high_value - low_value)
            upper_predictions.append(high_value)
            upper_uncertainties.append(high_uncertainty)
        effects.append(statistics.fmean(differences) if differences else 0.0)
        support.append(len(members))
        estimates.append(statistics.fmean(upper_predictions) if upper_predictions else float("nan"))
        spreads.append(
            statistics.fmean(upper_uncertainties) if upper_uncertainties else float("inf")
        )
    accumulated: list[float] = []
    total = 0.0
    for effect in effects:
        total += effect
        accumulated.append(total)
    center = statistics.fmean(accumulated) if accumulated else 0.0
    points = [
        {
            "x": upper,
            "effect": effect - center,
            "predicted_objective": estimate,
            "support_count": count,
            "uncertainty": spread,
        }
        for upper, effect, estimate, count, spread in zip(
            edges[1:], accumulated, estimates, support, spreads, strict=True
        )
    ]
    eligible = [point for point in points if math.isfinite(float(point["predicted_objective"]))]
    selected = (
        best(eligible, key=lambda value: float(value["predicted_objective"]), mode=mode)
        if eligible
        else None
    )
    return {
        "kind": "numeric-ale",
        "status": "available",
        "points": points,
        "best_supported_point": dict(selected) if selected is not None else None,
        "best_supported_region": supported_region(points, selected=selected, mode=mode),
    }


def pair_surface(
    left: str,
    right: str,
    *,
    model: AnalysisSurrogate,
    space: Mapping[str, Mapping[str, Any]],
    fingerprint: str,
) -> dict[str, Any]:
    left_values = grid(space[left], 9)
    right_values = grid(space[right], 9)
    contexts = reference_set(space, count=32, fingerprint=f"{fingerprint}:{left}:{right}")
    cells = []
    for left_value in left_values:
        for right_value in right_values:
            valid = next(
                (
                    assigned
                    for context in contexts
                    if (
                        assigned := assign_values(
                            context, {left: left_value, right: right_value}, space
                        )
                    )
                    is not None
                ),
                None,
            )
            if valid is None:
                continue
            predicted, uncertainty, distance = model.predict(valid)
            cells.append(
                {
                    "x": left_value,
                    "y": right_value,
                    "predicted_objective": predicted,
                    "uncertainty": uncertainty,
                    "distance_to_observed": distance,
                    "extrapolation": distance > 0.45,
                }
            )
    return {"left": left, "right": right, "x": left_values, "y": right_values, "cells": cells}


def reference_set(
    space: Mapping[str, Mapping[str, Any]], *, count: int, fingerprint: str
) -> list[dict[str, Any]]:
    seed = int(hashlib.sha256(fingerprint.encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    points: list[dict[str, Any]] = []
    for _ in range(count):
        points.append(sample_point(space, rng))
    return points


def mixed_distance(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    space: Mapping[str, Mapping[str, Any]],
) -> float:
    distances = []
    for name, rule in space.items():
        left_active, right_active = name in left, name in right
        if left_active != right_active:
            distances.append(1.0)
            continue
        if not left_active:
            distances.append(0.0)
        elif rule["kind"] == "numeric":
            low, high = float(rule["low"]), float(rule["high"])
            left_value, right_value = float(left[name]), float(right[name])
            if rule.get("scale") == "log" and low > 0 and left_value > 0 and right_value > 0:
                low, high = math.log(low), math.log(high)
                left_value, right_value = math.log(left_value), math.log(right_value)
            distances.append(abs(left_value - right_value) / max(high - low, 1e-12))
        else:
            distances.append(0.0 if left[name] == right[name] else 1.0)
    return math.sqrt(statistics.fmean(value * value for value in distances)) if distances else 0.0


def _bin(value: Any, rule: Mapping[str, Any]) -> Any:
    if value == INACTIVE:
        return INACTIVE
    if rule["kind"] == "categorical":
        return value
    low, high = float(rule["low"]), float(rule["high"])
    numeric = float(value)
    if rule.get("scale") == "log" and low > 0 and numeric > 0:
        low, high, numeric = math.log(low), math.log(high), math.log(numeric)
    return min(7, max(0, int(8 * (numeric - low) / max(high - low, 1e-12))))


def _group_variance(labels: Sequence[Any], values: Sequence[float]) -> float:
    grouped: dict[Any, list[float]] = {}
    for label, value in zip(labels, values, strict=True):
        grouped.setdefault(label, []).append(value)
    weighted_means = [statistics.fmean(grouped[label]) for label in labels]
    return statistics.pvariance(weighted_means) if len(weighted_means) > 1 else 0.0


def _interaction_matrix(
    names: Sequence[str], interactions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    lookup = {(value["left"], value["right"]): value["importance"] for value in interactions}
    return {
        left: {
            right: 0.0
            if left == right
            else lookup.get((left, right), lookup.get((right, left), 0.0))
            for right in names
        }
        for left in names
    }


def _jensen_shannon(left: Sequence[float], right: Sequence[float]) -> float:
    middle = [(a + b) / 2 for a, b in zip(left, right, strict=True)]

    def kl(first: Sequence[float], second: Sequence[float]) -> float:
        return sum(
            a * math.log2(a / b) for a, b in zip(first, second, strict=True) if a > 0 and b > 0
        )

    return min(1.0, max(0.0, 0.5 * kl(left, middle) + 0.5 * kl(right, middle)))


def _quantiles(values: Sequence[float], bins: int) -> list[float]:
    ordered = sorted(values)
    points = [ordered[round(index * (len(ordered) - 1) / bins)] for index in range(bins + 1)]
    return list(dict.fromkeys(points))


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2:
        return None
    left_ranks = _ranks(left)
    right_ranks = _ranks(right)
    left_mean, right_mean = statistics.fmean(left_ranks), statistics.fmean(right_ranks)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left_ranks, right_ranks, strict=True)
    )
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left_ranks)
        * sum((b - right_mean) ** 2 for b in right_ranks)
    )
    return numerator / denominator if denominator else 0.0


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    for rank, (_value, index) in enumerate(ordered, 1):
        ranks[index] = float(rank)
    return ranks


def _reliability(
    observations: int,
    model_quality: str,
    support_fraction: float,
    coverage_quality: str | None,
) -> str:
    if observations < 8 or model_quality in {"poor", "insufficient"} or support_fraction < 0.25:
        return "low"
    if (
        observations >= 24
        and model_quality == "good"
        and coverage_quality == "good"
        and support_fraction >= 0.5
    ):
        return "high"
    return "moderate"


def _empty_effects(observations: int) -> dict[str, Any]:
    return {
        "status": "insufficient",
        "observations": observations,
        "global_importance": {},
        "top_region_importance": {},
        "response_curves": {},
        "interactions": [],
        "interaction_matrix": {},
        "pairwise_surfaces": {},
        "reference_points": 0,
    }


def _finite(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


__all__ = [
    "AnalysisSurrogate",
    "analyze_effects",
    "infer_space",
    "mixed_distance",
    "reference_set",
    "top_region_importance",
    "validate_surrogate",
]
