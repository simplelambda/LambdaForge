"""Deterministic contract tests for the optional clustering family."""

from __future__ import annotations

import builtins

import numpy as np
import pytest
import torch

from lambdaforge.clustering import (
    DBSCAN,
    HDBSCAN,
    Agglomerative,
    Clusterer,
    KMeans,
    MiniBatchKMeans,
    adjusted_rand_index,
    silhouette_score,
    stability,
)
from lambdaforge.nn.distances import ManhattanDistance


@pytest.fixture
def separated() -> np.ndarray:
    return np.asarray([[0.0, 0.0], [0.0, 0.1], [0.1, 0.0], [5.0, 5.0], [5.0, 5.1], [5.1, 5.0]])


@pytest.mark.parametrize(
    "clusterer",
    [KMeans(2, seed=7), MiniBatchKMeans(2, seed=7, batch_size=3)],
)
def test_centroid_clusterers_are_uniform_and_deterministic(
    clusterer: Clusterer, separated: np.ndarray
) -> None:
    first = clusterer.cluster(separated)
    second = clusterer.cluster(torch.from_numpy(separated))
    assert adjusted_rand_index(first.labels, second.labels) == pytest.approx(1.0)
    assert first.n_clusters == 2
    assert first.noise_count == 0
    assert first.centers is not None and first.centers.shape == (2, 2)
    assert first.inertia is not None
    assert not first.labels.flags.writeable
    assert silhouette_score(separated, first) > 0.9


def test_density_clusterers_noise_probabilities_and_custom_distance(
    separated: np.ndarray,
) -> None:
    with_noise = np.vstack([separated, [[20.0, -20.0]]])
    dbscan = DBSCAN(eps=0.25, min_samples=2).cluster(with_noise)
    assert dbscan.n_clusters == 2
    assert dbscan.labels[-1] == -1
    assert dbscan.noise_count == 1

    custom = DBSCAN(
        eps=0.25,
        min_samples=2,
        distance=ManhattanDistance(),
    ).cluster(with_noise)
    assert custom.diagnostics["distance_mode"] == "precomputed"
    assert adjusted_rand_index(dbscan.labels, custom.labels) == pytest.approx(1.0)

    hierarchy = HDBSCAN(min_cluster_size=2, min_samples=2).cluster(with_noise)
    assert hierarchy.n_clusters == 2
    assert hierarchy.probabilities is not None
    assert hierarchy.probabilities.shape == (len(with_noise),)


def test_agglomerative_and_distance_compatibility(separated: np.ndarray) -> None:
    result = Agglomerative(2).cluster(separated)
    assert result.n_clusters == 2
    with pytest.raises(ValueError, match="requires Euclidean"):
        Agglomerative(2, linkage="ward", distance="cosine").cluster(separated)
    with pytest.raises(ValueError, match="does not support distance"):
        KMeans(2, distance="cosine").cluster(separated)
    with pytest.raises(ValueError, match="cannot consume a custom"):
        KMeans(2, distance=ManhattanDistance()).cluster(separated)


def test_validation_memory_guard_and_stability(separated: np.ndarray) -> None:
    for invalid in (np.asarray([]), np.zeros((2, 2, 2)), np.asarray([[np.nan, 1.0]])):
        with pytest.raises((ValueError, TypeError)):
            DBSCAN().cluster(invalid)
    with pytest.raises(MemoryError, match="O\(N\^2\)"):
        DBSCAN(distance=ManhattanDistance(), max_pairwise_bytes=10).cluster(separated)
    report = stability(separated, [KMeans(2, seed=1), Agglomerative(2)])
    assert report.minimum == pytest.approx(1.0)
    assert len(report.pairwise) == 2


def test_optional_backend_error_is_actionable(
    monkeypatch: pytest.MonkeyPatch, separated: np.ndarray
) -> None:
    original_import = builtins.__import__

    def without_sklearn(name: str, *args: object, **kwargs: object) -> object:
        if name == "sklearn":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_sklearn)
    with pytest.raises(ImportError, match=r"lambdaforge\[clustering\]"):
        KMeans(2).cluster(separated)
