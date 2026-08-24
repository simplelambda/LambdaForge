"""Small backend-normalized metrics for clustering evidence."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np

from .backend import feature_array, require_sklearn
from .base import ArrayInput, Clusterer, ClusteringResult


def adjusted_rand_index(first: np.ndarray, second: np.ndarray) -> float:
    """Return the permutation-invariant adjusted Rand index for two labelings."""
    require_sklearn("adjusted_rand_index")
    from sklearn.metrics import adjusted_rand_score

    left, right = np.asarray(first), np.asarray(second)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("Adjusted Rand inputs must be equally sized one-dimensional labels.")
    return float(adjusted_rand_score(left, right))


def silhouette_score(
    features: ArrayInput,
    result: ClusteringResult | np.ndarray,
    *,
    distance: str = "euclidean",
) -> float:
    """Return the mean silhouette score for explicit features and labels."""
    require_sklearn("silhouette_score")
    from sklearn.metrics import silhouette_score as backend_score

    labels = result.labels if isinstance(result, ClusteringResult) else np.asarray(result)
    return float(backend_score(feature_array(features), labels, metric=distance))


@dataclass(frozen=True, slots=True)
class StabilityResult:
    """Generic adjusted-Rand evidence across explicitly chosen clusterers."""

    scores: tuple[float, ...]
    pairwise: tuple[tuple[float, ...], ...]
    median: float
    minimum: float
    results: tuple[ClusteringResult, ...]


def stability(
    features: ArrayInput,
    candidates: list[Clusterer] | tuple[Clusterer, ...],
    *,
    reference: ClusteringResult | None = None,
) -> StabilityResult:
    """Cluster explicit candidates and compare label stability without policy thresholds."""
    if not candidates:
        raise ValueError("stability requires at least one candidate clusterer.")
    results = tuple(candidate.cluster(features) for candidate in candidates)
    canonical = reference or results[0]
    scores = tuple(adjusted_rand_index(canonical.labels, result.labels) for result in results)
    pairwise = tuple(
        tuple(adjusted_rand_index(left.labels, right.labels) for right in results)
        for left in results
    )
    return StabilityResult(
        scores,
        pairwise,
        float(statistics.median(scores)),
        float(min(scores)),
        results,
    )


__all__ = [
    "StabilityResult",
    "adjusted_rand_index",
    "silhouette_score",
    "stability",
]
