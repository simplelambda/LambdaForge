"""Descriptive, equal-weight views of observed candidate means.

This is visualization evidence, not a selector, surrogate or HPO policy.
Seed uncertainty and dispersion across different configurations are distinct.
"""
from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any, TypeGuard


def _number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _key(value: Any) -> str:
    if _number(value):
        return json.dumps({"number": 0.0 if value == 0 else float(value)}, allow_nan=False)
    return json.dumps({"category": value}, sort_keys=True, allow_nan=False, separators=(",", ":"))


class ObservedStudyChart:
    """Group recorded candidate means without fitting or altering scientific evidence."""

    @staticmethod
    def summarize(
        analysis: Mapping[str, Any],
        *,
        metrics: Sequence[str],
        x: str = "trial",
        second: str | None = None,
        center: str = "mean",
        spread: str = "sd",
    ) -> dict[str, Any]:
        if center not in {"mean", "median"}:
            raise ValueError("Observed chart center must be mean or median.")
        if spread not in {"none", "sd", "se", "iqr", "range"}:
            raise ValueError("Unsupported observed chart spread.")
        if center == "median" and spread == "se":
            raise ValueError("Standard error of a mean is not uncertainty of a median.")
        primary = str((analysis.get("objective") or {}).get("metric", "objective"))
        selected = tuple(dict.fromkeys(metrics))
        raw: list[dict[str, Any]] = []
        groups: dict[str, list[dict[str, Any]]] = {}
        for candidate in analysis.get("candidates", ()):
            if not isinstance(candidate, Mapping):
                continue
            parameters = candidate.get("parameters") or {}
            trial = candidate.get("trial")
            axis = trial if x == "trial" else parameters.get(x)
            other = parameters.get(second) if second else None
            if axis is None or second and other is None:
                continue
            for metric in selected:
                value = candidate if metric == primary else (
                    candidate.get("diagnostic_metrics") or {}
                ).get(metric, {})
                score = value.get("mean")
                # Missing/censored final evidence is never replaced by a best observation.
                if not _number(score) or not _number(value.get("n")) or value["n"] <= 0:
                    continue
                group_key = _key([metric, json.loads(_key(axis)), other])
                point = {
                    "id": f"{trial}:{metric}", "trial": trial, "x": axis,
                    "y": float(score), "series": metric, "parameters": dict(parameters),
                    "se": value.get("standard_error"), "sd": value.get("standard_deviation"),
                    "seed_count": value.get("n"), "group": group_key,
                }
                raw.append(point)
                groups.setdefault(group_key, []).append(point)
        records: list[dict[str, Any]] = []
        for group_key, observations in groups.items():
            values = [point["y"] for point in observations]
            estimate = statistics.fmean(values) if center == "mean" else statistics.median(values)
            lower: float | None = None
            upper: float | None = None
            scope = "between-configurations"
            if len(values) > 1:
                if spread in {"sd", "se"}:
                    deviation = statistics.stdev(values)
                    error = deviation if spread == "sd" else deviation / math.sqrt(len(values))
                    lower, upper = estimate - error, estimate + error
                elif spread == "iqr":
                    quartiles = statistics.quantiles(values, n=4, method="inclusive")
                    lower, upper = quartiles[0], quartiles[2]
                elif spread == "range":
                    lower, upper = min(values), max(values)
            elif spread in {"sd", "se"}:
                native = observations[0].get(spread)
                if _number(native) and native >= 0:
                    lower, upper = estimate - native, estimate + native
                    scope = "within-seeds"
            first = observations[0]
            records.append({
                "id": group_key, "group": group_key, "x": first["x"], "y": estimate,
                "series": first["series"], "parameters": dict(first["parameters"]),
                "lower": lower, "upper": upper,
                "error": (
                    None if lower is None or upper is None
                    else max(estimate - lower, upper - estimate)
                ),
                "n": len(observations), "trials": [point["trial"] for point in observations],
                "spread_scope": scope, "center": center, "spread": spread,
            })
        records.sort(key=lambda point: (
            point["series"], 0 if _number(point["x"]) else 1,
            point["x"] if _number(point["x"]) else _key(point["x"]),
        ))
        return {
            "chart_projection_version": 1, "records": records, "observations": raw,
            "center": center, "spread": spread,
            "semantics": "equal-weight candidate means; descriptive, not causal",
        }
