"""Uniform optional clustering API; imports stay lazy for unrelated CLI/Work metadata reads."""

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from .algorithms import DBSCAN, HDBSCAN, Agglomerative, KMeans, MiniBatchKMeans
    from .base import ArrayInput, ClusterCapabilities, Clusterer, ClusteringResult
    from .metrics import StabilityResult, adjusted_rand_index, silhouette_score, stability

LazyExports.install(
    __name__,
    {
        name: (f"lambdaforge.clustering.{module}", name)
        for module, names in {
            "algorithms": ("DBSCAN", "HDBSCAN", "Agglomerative", "KMeans", "MiniBatchKMeans"),
            "base": ("ArrayInput", "ClusterCapabilities", "Clusterer", "ClusteringResult"),
            "metrics": ("StabilityResult", "adjusted_rand_index", "silhouette_score", "stability"),
        }.items()
        for name in names
    },
)

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
