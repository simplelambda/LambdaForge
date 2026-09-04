"""Search-space coverage, pool resolution and conservative boundary diagnostics."""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from lambdaforge.analysis.Effects import mixed_distance, reference_set
from lambdaforge.analysis.SearchSpace import valid_point

BOUNDARY_THRESHOLDS: dict[str, float] = {
    "edge_fraction": 0.10,
    "minimum_top": 5,
    "top_concentration": 0.50,
    "enrichment": 1.50,
    "direction_support": 0.70,
}


def analyze_coverage(
    candidates: Sequence[Mapping[str, Any]],
    *,
    space: Mapping[str, Mapping[str, Any]],
    mode: str,
    fingerprint: str,
    proposal_pool_size: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return coverage, candidate-pool resolution and boundary saturation."""
    raw_observed = [dict(value.get("parameters", {})) for value in candidates]
    observed = [value for value in raw_observed if valid_point(value, space)]
    full = [
        dict(value.get("parameters", {}))
        for value in candidates
        if isinstance(value.get("mean"), int | float)
        and valid_point(value.get("parameters", {}), space)
    ]
    marginal = {
        name: _marginal(name, rule, observed=observed, full=full) for name, rule in space.items()
    }
    reference = reference_set(
        space, count=min(2048, max(256, len(observed) * 32)), fingerprint=fingerprint
    )
    attempted_distances = _nearest(reference, observed, space)
    full_distances = _nearest(reference, full, space)
    baselines = []
    for index in range(min(32, max(8, len(observed)))):
        design = reference_set(
            space, count=max(1, len(observed)), fingerprint=f"{fingerprint}:baseline:{index}"
        )
        values = _nearest(reference, design, space)
        baselines.append(statistics.median(values) if values else 1.0)
    attempted_median = statistics.median(attempted_distances) if attempted_distances else None
    percentile = (
        sum(value >= attempted_median for value in baselines) / len(baselines)
        if attempted_median is not None and baselines
        else None
    )
    joint = {
        "attempted": _distance_summary(attempted_distances),
        "full_fidelity": _distance_summary(full_distances),
        "baseline_designs": len(baselines),
        "coverage_percentile_vs_same_size_baseline": percentile,
        "quality": _coverage_quality(percentile, attempted_median),
        "reference_points": len(reference),
    }
    spacing = _pair_spacing(observed, space)
    pool = {
        "kind": "observed-candidate-resolution",
        "observed_candidates": len(observed),
        "invalid_observed_candidates": len(raw_observed) - len(observed),
        "proposal_pool_size": proposal_pool_size,
        "proposal_pool_points_persisted": False,
        "proposal_pool_coverage": None,
        "median_nearest_neighbour_spacing": statistics.median(spacing) if spacing else None,
        "largest_uncovered_region_approximation": max(attempted_distances, default=None),
        "resolution_limited": bool(
            spacing
            and attempted_distances
            and max(attempted_distances) > max(0.35, statistics.median(spacing) * 2)
        ),
        "interpretation": (
            "Resolution of proposed/observed candidates only; the unpersisted sampler pool "
            "is not characterized."
        ),
    }
    boundaries = {
        name: _boundary(name, rule, candidates, mode=mode, fingerprint=fingerprint)
        for name, rule in space.items()
        if rule.get("kind") == "numeric"
    }
    return {"marginal": marginal, "joint": joint}, pool, boundaries


def _marginal(
    name: str,
    rule: Mapping[str, Any],
    *,
    observed: Sequence[Mapping[str, Any]],
    full: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    attempted = [row[name] for row in observed if name in row]
    completed = [row[name] for row in full if name in row]
    inactive = sum(name not in row for row in observed)
    if rule["kind"] == "categorical":
        levels = list(rule.get("values", ()))
        return {
            "kind": "categorical",
            "authored_levels": levels,
            "observed_levels": sorted(set(attempted), key=str),
            "level_counts": {str(level): attempted.count(level) for level in levels},
            "full_fidelity_level_counts": {str(level): completed.count(level) for level in levels},
            "active_fraction": len(attempted) / len(observed) if observed else 0.0,
            "inactive_fraction": inactive / len(observed) if observed else 0.0,
        }
    low, high = float(rule["low"]), float(rule["high"])
    transformed = [_normalized(float(value), rule) for value in attempted]
    bins = {min(9, max(0, int(value * 10))) for value in transformed}
    return {
        "kind": "numeric",
        "authored_range": [low, high],
        "observed_range": [min(map(float, attempted)), max(map(float, attempted))]
        if attempted
        else None,
        "transformed_range_coverage": max(transformed) - min(transformed) if transformed else 0.0,
        "occupied_bins": len(bins),
        "total_bins": 10,
        "edge_support": {
            "lower": sum(value <= 0.1 for value in transformed),
            "upper": sum(value >= 0.9 for value in transformed),
        },
        "active_fraction": len(attempted) / len(observed) if observed else 0.0,
        "inactive_fraction": inactive / len(observed) if observed else 0.0,
        "full_fidelity_count": len(completed),
    }


def _boundary(
    name: str,
    rule: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    fingerprint: str,
) -> dict[str, Any]:
    rows = [
        value
        for value in candidates
        if isinstance(value.get("mean"), int | float) and name in value.get("parameters", {})
    ]
    minimum_top = int(BOUNDARY_THRESHOLDS["minimum_top"])
    if len(rows) < minimum_top:
        return {"status": "none", "reason": "insufficient_evidence", "support": len(rows)}
    ordered = sorted(rows, key=lambda value: float(value["mean"]), reverse=mode == "max")
    top_count = max(minimum_top, math.ceil(len(rows) * 0.1))
    top = ordered[:top_count]
    normalized = [_normalized(float(row["parameters"][name]), rule) for row in rows]
    top_normalized = [_normalized(float(row["parameters"][name]), rule) for row in top]
    lower_global = max(sum(value <= 0.1 for value in normalized) / len(normalized), 1e-12)
    upper_global = max(sum(value >= 0.9 for value in normalized) / len(normalized), 1e-12)
    lower_top = sum(value <= 0.1 for value in top_normalized) / len(top_normalized)
    upper_top = sum(value >= 0.9 for value in top_normalized) / len(top_normalized)
    side = "upper" if upper_top / upper_global >= lower_top / lower_global else "lower"
    concentration = upper_top if side == "upper" else lower_top
    enrichment = concentration / (upper_global if side == "upper" else lower_global)
    local = sorted(
        (z, float(row["mean"]))
        for z, row in zip(normalized, rows, strict=True)
        if (z >= 0.7 if side == "upper" else z <= 0.3)
    )
    slope = _slope(local)
    signed_slope = slope * (1 if mode == "max" else -1) * (1 if side == "upper" else -1)
    direction = "toward_boundary" if signed_slope > 0 else "away_or_flat"
    support = len(local)
    direction_support = _bootstrap_direction_support(
        local,
        sign=(1 if mode == "max" else -1) * (1 if side == "upper" else -1),
        fingerprint=f"{fingerprint}:{name}:{side}",
    )
    likely = (
        concentration >= BOUNDARY_THRESHOLDS["top_concentration"]
        and enrichment >= BOUNDARY_THRESHOLDS["enrichment"]
        and signed_slope > 0
        and support >= 3
        and direction_support >= BOUNDARY_THRESHOLDS["direction_support"]
    )
    possible = concentration >= 0.4 and enrichment >= 1.2 and signed_slope > 0
    return {
        "status": "likely" if likely else "possible" if possible else "none",
        "boundary": side,
        "authored_range": [rule["low"], rule["high"]],
        "top_boundary_concentration": concentration,
        "global_boundary_concentration": upper_global if side == "upper" else lower_global,
        "boundary_enrichment": enrichment,
        "local_response_direction": direction,
        "local_slope": slope,
        "bootstrap_direction_support": direction_support,
        "boundary_support": support,
        "top_count": len(top),
        "thresholds": dict(BOUNDARY_THRESHOLDS),
    }


def _nearest(
    reference: Sequence[Mapping[str, Any]],
    observed: Sequence[Mapping[str, Any]],
    space: Mapping[str, Mapping[str, Any]],
) -> list[float]:
    if not observed:
        return []
    return [min(mixed_distance(point, row, space) for row in observed) for point in reference]


def _pair_spacing(
    points: Sequence[Mapping[str, Any]], space: Mapping[str, Mapping[str, Any]]
) -> list[float]:
    return [
        min(
            mixed_distance(point, other, space)
            for other_index, other in enumerate(points)
            if index != other_index
        )
        for index, point in enumerate(points)
        if len(points) > 1
    ]


def _distance_summary(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "median_nearest_observed_distance": statistics.median(ordered) if ordered else None,
        "p90_distance": ordered[int(0.9 * (len(ordered) - 1))] if ordered else None,
        "maximum_distance": max(ordered) if ordered else None,
    }


def _coverage_quality(percentile: float | None, median: float | None) -> str:
    if percentile is None or median is None:
        return "insufficient"
    if percentile >= 0.67 and median <= 0.3:
        return "good"
    if percentile >= 0.33 and median <= 0.5:
        return "moderate"
    return "poor"


def _normalized(value: float, rule: Mapping[str, Any]) -> float:
    low, high = float(rule["low"]), float(rule["high"])
    if rule.get("scale") == "log" and low > 0 and value > 0:
        low, high, value = math.log(low), math.log(high), math.log(value)
    return min(1.0, max(0.0, (value - low) / max(high - low, 1e-12)))


def _slope(values: Sequence[tuple[float, float]]) -> float:
    if len(values) < 2:
        return 0.0
    x_mean = statistics.fmean(value[0] for value in values)
    y_mean = statistics.fmean(value[1] for value in values)
    denominator = sum((x - x_mean) ** 2 for x, _y in values)
    return sum((x - x_mean) * (y - y_mean) for x, y in values) / denominator if denominator else 0.0


def _bootstrap_direction_support(
    values: Sequence[tuple[float, float]], *, sign: float, fingerprint: str
) -> float:
    if len(values) < 3:
        return 0.0
    rng = random.Random(int(hashlib.sha256(fingerprint.encode()).hexdigest()[:16], 16))
    replicates = 500
    toward = 0
    for _ in range(replicates):
        sampled = [rng.choice(values) for _item in values]
        toward += sign * _slope(sampled) > 0
    return toward / replicates


__all__ = ["BOUNDARY_THRESHOLDS", "analyze_coverage"]
