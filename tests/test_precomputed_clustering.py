"""Dissimilarities are explicit data, not accidental feature vectors."""

import numpy as np
import pytest
import torch
from sklearn import cluster, metrics

from lambdaforge.clustering import DBSCAN, HDBSCAN, Agglomerative, KMeans, silhouette_score
from lambdaforge.distances import validate_dissimilarity


@pytest.fixture
def matrix():
    points = np.array([[0.0], [0.1], [0.2], [5.0], [5.1], [5.2]])
    return abs(points - points.T)


@pytest.mark.parametrize("name", ["dbscan", "hdbscan", "single", "average", "complete"])
def test_precomputed_matches_backend(matrix, name):
    if name == "dbscan":
        ours = DBSCAN(eps=0.3, min_samples=2, distance="precomputed")
        backend = cluster.DBSCAN(eps=0.3, min_samples=2, metric="precomputed")
    elif name == "hdbscan":
        ours = HDBSCAN(min_cluster_size=2, min_samples=2, distance="precomputed")
        backend = cluster.HDBSCAN(
            min_cluster_size=2, min_samples=2, metric="precomputed", copy=True
        )
    else:
        ours = Agglomerative(2, linkage=name, distance="precomputed")
        backend = cluster.AgglomerativeClustering(2, linkage=name, metric="precomputed")
    original = matrix.copy()
    result = ours.cluster(matrix)
    expected = backend.fit_predict(matrix)
    assert metrics.adjusted_rand_score(result.labels, expected) == 1
    assert result.diagnostics["distance_mode"] == "precomputed"
    assert np.array_equal(matrix, original)
    assert np.array_equal(ours.cluster(torch.from_numpy(matrix)).labels, result.labels)
    assert silhouette_score(matrix, result, distance="precomputed") == pytest.approx(
        metrics.silhouette_score(matrix, expected, metric="precomputed")
    )


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros((2, 3)),
        np.zeros((0, 0)),
        np.ones((2, 2)),
        np.array([[0, -1], [-1, 0]]),
        np.array([[0, np.nan], [np.nan, 0]]),
        np.array([[0, np.inf], [np.inf, 0]]),
        np.array([[0, 1], [2, 0]]),
        np.array([[0, 1j], [1j, 0]]),
        np.array([["0"]]),
    ],
)
def test_invalid_matrices_rejected(invalid):
    for algorithm in (DBSCAN(), HDBSCAN(min_cluster_size=2), Agglomerative(linkage="average")):
        algorithm.distance = "precomputed"
        with pytest.raises((TypeError, ValueError)):
            algorithm.cluster(invalid)


def test_incompatible_algorithms_and_memory(matrix):
    for algorithm in (KMeans(2, distance="precomputed"), Agglomerative(distance="precomputed")):
        with pytest.raises(ValueError):
            algorithm.cluster(matrix)
    for values in (matrix.astype(np.float32), torch.from_numpy(matrix)):
        with pytest.raises(MemoryError, match="O\\(N\\^2\\)"):
            DBSCAN(distance="precomputed", max_pairwise_bytes=8).cluster(values)
    with pytest.raises(MemoryError):
        silhouette_score(
            matrix, np.array([0, 0, 0, 1, 1, 1]), distance="precomputed", max_pairwise_bytes=8
        )


def test_explicit_symmetry_policy():
    matrix = np.array([[0.0, 1.0], [1.1, 0.0]])
    with pytest.raises(ValueError, match="symmetric"):
        validate_dissimilarity(matrix)
    result = validate_dissimilarity(matrix, symmetry="average")
    assert result[0, 1] == pytest.approx(1.05)
    assert matrix[0, 1] == 1.0
    assert validate_dissimilarity(matrix, symmetry_atol=0.11)[0, 1] == 1
    with pytest.raises(ValueError):
        validate_dissimilarity(matrix, symmetry_atol=-1)
