"""Input, optional-backend and distance adaptation shared by clusterers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from lambdaforge.nn.distances import (
    Distance,
    EuclideanDistance,
    SquaredEuclideanDistance,
)

from .base import ArrayInput, ClusterCapabilities

ALIASES = {
    "l1": "manhattan",
    "cityblock": "manhattan",
    "l2": "euclidean",
    "sqeuclidean": "squared_euclidean",
    "squared-euclidean": "squared_euclidean",
}


def require_sklearn(algorithm: str) -> None:
    """Raise an actionable optional-dependency error before backend use."""
    try:
        import sklearn  # noqa: F401
    except ImportError as error:
        raise ImportError(
            f"{algorithm} requires LambdaForge clustering support. "
            'Install with: pip install "lambdaforge[clustering]"'
        ) from error


def feature_array(features: ArrayInput) -> np.ndarray:
    """Detach tensors, move to CPU and validate a finite non-empty matrix."""
    if isinstance(features, torch.Tensor):
        array = features.detach().cpu().numpy()
    elif isinstance(features, np.ndarray):
        array = features
    else:
        raise TypeError("Clustering features must be a numpy.ndarray or torch.Tensor.")
    if array.ndim != 2:
        raise ValueError(f"Clustering features must have shape [N, F], got {array.shape}.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("Clustering requires at least one sample and one feature.")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("Clustering features must be numeric.")
    selected = np.asarray(array, dtype=np.float64)
    if not np.isfinite(selected).all():
        raise ValueError("Clustering features cannot contain NaN or infinity.")
    return selected


def canonical_distance(distance: str | Distance) -> str | Distance:
    """Normalize documented string aliases while retaining Distance objects."""
    if isinstance(distance, Distance):
        return distance
    if not isinstance(distance, str):
        raise TypeError("distance must be a canonical string or LambdaForge Distance object.")
    selected = distance.strip().lower().replace(" ", "_")
    selected = ALIASES.get(selected, selected)
    if not selected:
        raise ValueError("distance cannot be empty.")
    return selected


def adapt_distance(
    distance: str | Distance,
    capabilities: ClusterCapabilities,
    features: np.ndarray,
    *,
    algorithm: str,
    max_pairwise_bytes: int | None,
    ward: bool = False,
) -> tuple[str, np.ndarray]:
    """Return a backend metric and raw/precomputed features after compatibility checks."""
    selected = canonical_distance(distance)
    if ward:
        compatible = selected in {"euclidean", "squared_euclidean"} or isinstance(
            selected, EuclideanDistance | SquaredEuclideanDistance
        )
        if not compatible:
            raise ValueError(
                "Agglomerative linkage='ward' minimizes Euclidean variance and therefore "
                "requires Euclidean distance."
            )
        return "euclidean", features
    if isinstance(selected, str):
        if selected not in capabilities.native_metrics:
            supported = ", ".join(sorted(capabilities.native_metrics))
            raise ValueError(
                f"{algorithm} does not support distance {selected!r}; native distances: "
                f"{supported or 'none'}."
            )
        return selected, features
    if isinstance(selected, EuclideanDistance) and "euclidean" in capabilities.native_metrics:
        return "euclidean", features
    if isinstance(selected, SquaredEuclideanDistance) and (
        "squared_euclidean" in capabilities.native_metrics
    ):
        return "squared_euclidean", features
    if not capabilities.supports_precomputed:
        raise ValueError(
            f"{algorithm} cannot consume a custom LambdaForge Distance because its backend "
            "does not support precomputed pairwise matrices."
        )
    return "precomputed", pairwise(selected, features, max_pairwise_bytes=max_pairwise_bytes)


def pairwise(
    distance: Distance,
    features: np.ndarray,
    *,
    max_pairwise_bytes: int | None,
) -> np.ndarray:
    """Materialize an explicit custom NxN distance matrix under a memory guard."""
    required = int(features.shape[0]) ** 2 * 8
    if max_pairwise_bytes is not None:
        if (
            isinstance(max_pairwise_bytes, bool)
            or not isinstance(max_pairwise_bytes, int)
            or max_pairwise_bytes < 1
        ):
            raise ValueError("max_pairwise_bytes must be a positive integer or null.")
        if required > max_pairwise_bytes:
            raise MemoryError(
                f"Custom distance for {features.shape[0]} samples needs approximately "
                f"{required} bytes, above max_pairwise_bytes={max_pairwise_bytes}. "
                "Increase the explicit guard only after accounting for O(N^2) memory."
            )
    tensors = tuple(distance.parameters()) + tuple(distance.buffers())
    template = tensors[0] if tensors else None
    dtype = (
        template.dtype if template is not None and template.is_floating_point() else torch.float64
    )
    device = template.device if template is not None else torch.device("cpu")
    values = torch.as_tensor(features, dtype=dtype, device=device).unsqueeze(0)
    with torch.no_grad():
        matrix = distance(values, values)
    if matrix.shape != (1, features.shape[0], features.shape[0]):
        raise ValueError(
            "LambdaForge Distance returned an invalid pairwise shape; expected "
            f"(1, {features.shape[0]}, {features.shape[0]}), got {tuple(matrix.shape)}."
        )
    selected = np.asarray(matrix.detach().cpu(), dtype=np.float64)[0]
    if not np.isfinite(selected).all() or np.any(selected < -1e-12):
        raise ValueError("Custom Distance produced non-finite or negative pairwise values.")
    if not np.allclose(selected, selected.T, rtol=1e-6, atol=1e-8):
        raise ValueError("Custom Distance must produce a symmetric pairwise matrix for clustering.")
    if not np.allclose(np.diag(selected), 0.0, rtol=0.0, atol=1e-7):
        raise ValueError("Custom Distance must produce a zero diagonal for clustering.")
    return selected


@contextmanager
def thread_limit(threads: int | None) -> Iterator[None]:
    """Bound native numerical thread pools for the duration of one backend call."""
    if threads is not None and (
        isinstance(threads, bool) or not isinstance(threads, int) or threads < 1
    ):
        raise ValueError("threads must be an integer >= 1 or null.")
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=threads):
        yield


def diagnostics(algorithm: str, metric: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Create compact serializable backend evidence without retaining estimators."""
    return {
        "algorithm": algorithm,
        "backend": "scikit-learn",
        "distance_mode": metric,
        "parameters": parameters,
    }
