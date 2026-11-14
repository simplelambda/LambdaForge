"""Small backend-normalized metrics for clustering evidence."""

from __future__ import annotations

import statistics

import numpy as np

from ._types import ArrayInput
from .backend import clustering_array, require_sklearn
from .Clusterer import Clusterer
from .ClusteringResult import ClusteringResult
from .StabilityResult import StabilityResult


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
    max_pairwise_bytes: int | None = 512 * 1024 * 1024,
) -> float:
    """Mean silhouette for features or validated ``distance='precomputed'`` data.

    Noise (-1) remains a label; filter it explicitly if a noise-excluded score is
    desired. This function never silently removes observations.
    """
    require_sklearn("silhouette_score")
    from sklearn.metrics import silhouette_score as backend_score

    labels = result.labels if isinstance(result, ClusteringResult) else np.asarray(result)
    values = clustering_array(features, distance, max_pairwise_bytes)
    return float(backend_score(values, labels, metric=distance))


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
