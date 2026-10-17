"""Agglomerative adapter with the shared native clustering contract."""

from __future__ import annotations

from lambdaforge.nn.distances import Distance

from .._types import ArrayInput
from ..backend import adapt_distance, diagnostics, feature_array, require_sklearn, thread_limit
from ..ClusterCapabilities import ClusterCapabilities
from ..Clusterer import Clusterer
from ..ClusteringResult import ClusteringResult
from ._metrics import DENSITY_METRICS


class Agglomerative(Clusterer):
    """Agglomerative clustering with explicit linkage/distance compatibility."""

    capabilities = ClusterCapabilities(DENSITY_METRICS, True)

    def __init__(
        self,
        n_clusters: int | None = 2,
        *,
        linkage: str = "ward",
        distance: str | Distance = "euclidean",
        distance_threshold: float | None = None,
        threads: int | None = None,
        max_pairwise_bytes: int | None = 512 * 1024 * 1024,
    ) -> None:
        if n_clusters is not None and (
            isinstance(n_clusters, bool) or not isinstance(n_clusters, int) or n_clusters < 1
        ):
            raise ValueError("n_clusters must be an integer >= 1 or null.")
        if linkage not in {"ward", "complete", "average", "single"}:
            raise ValueError("linkage must be ward, complete, average or single.")
        if distance_threshold is not None and n_clusters is not None:
            raise ValueError("n_clusters must be null when distance_threshold is configured.")
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distance = distance
        self.distance_threshold = distance_threshold
        self.threads = threads
        self.max_pairwise_bytes = max_pairwise_bytes

    def cluster(self, features: ArrayInput) -> ClusteringResult:
        require_sklearn(type(self).__name__)
        from sklearn.cluster import AgglomerativeClustering

        values = feature_array(features)
        if self.n_clusters is not None and len(values) < self.n_clusters:
            raise ValueError("n_clusters cannot exceed the number of samples.")
        metric, backend_values = adapt_distance(
            self.distance,
            self.capabilities,
            values,
            algorithm=type(self).__name__,
            max_pairwise_bytes=self.max_pairwise_bytes,
            ward=self.linkage == "ward",
        )
        parameters = {
            "n_clusters": self.n_clusters,
            "linkage": self.linkage,
            "distance_threshold": self.distance_threshold,
            "threads": self.threads,
        }
        with thread_limit(self.threads):
            labels = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage=self.linkage,
                metric=metric,
                distance_threshold=self.distance_threshold,
            ).fit_predict(backend_values)
        return ClusteringResult.from_labels(
            labels,
            diagnostics=diagnostics(type(self).__name__, metric, parameters),
        )
