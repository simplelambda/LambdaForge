"""Uniform optional clustering API; backend imports remain lazy until execution."""

from .algorithms import DBSCAN, HDBSCAN, Agglomerative, KMeans, MiniBatchKMeans
from .base import ArrayInput, ClusterCapabilities, Clusterer, ClusteringResult
from .metrics import StabilityResult, adjusted_rand_index, silhouette_score, stability

__all__ = [
    "Agglomerative",
    "ArrayInput",
    "ClusterCapabilities",
    "Clusterer",
    "ClusteringResult",
    "DBSCAN",
    "HDBSCAN",
    "KMeans",
    "MiniBatchKMeans",
    "StabilityResult",
    "adjusted_rand_index",
    "silhouette_score",
    "stability",
]
