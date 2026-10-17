"""KMeans adapter with the shared native clustering contract."""

from __future__ import annotations

from lambdaforge.nn.distances import Distance

from .._types import ArrayInput
from ..backend import adapt_distance, diagnostics, feature_array, require_sklearn, thread_limit
from ..ClusterCapabilities import ClusterCapabilities
from ..Clusterer import Clusterer
from ..ClusteringResult import ClusteringResult
from ._metrics import EUCLIDEAN


class KMeans(Clusterer):
    """Centroid clustering with scientifically explicit Euclidean semantics."""

    capabilities = ClusterCapabilities(EUCLIDEAN, False, supports_centers=True)

    def __init__(
        self,
        n_clusters: int,
        *,
        distance: str | Distance = "euclidean",
        seed: int | None = None,
        n_init: int | str = "auto",
        max_iter: int = 300,
        tolerance: float = 1e-4,
        threads: int | None = None,
    ) -> None:
        if isinstance(n_clusters, bool) or not isinstance(n_clusters, int) or n_clusters < 1:
            raise ValueError("n_clusters must be an integer >= 1.")
        self.n_clusters = n_clusters
        self.distance = distance
        self.seed = seed
        self.n_init = n_init
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.threads = threads

    def cluster(self, features: ArrayInput) -> ClusteringResult:
        require_sklearn(type(self).__name__)
        from sklearn.cluster import KMeans as BackendKMeans

        values = feature_array(features)
        if len(values) < self.n_clusters:
            raise ValueError("n_clusters cannot exceed the number of samples.")
        metric, values = adapt_distance(
            self.distance,
            self.capabilities,
            values,
            algorithm=type(self).__name__,
            max_pairwise_bytes=None,
        )
        parameters = {
            "n_clusters": self.n_clusters,
            "seed": self.seed,
            "n_init": self.n_init,
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
            "threads": self.threads,
        }
        with thread_limit(self.threads):
            estimator = BackendKMeans(
                n_clusters=self.n_clusters,
                random_state=self.seed,
                n_init=self.n_init,
                max_iter=self.max_iter,
                tol=self.tolerance,
            ).fit(values)
        return ClusteringResult.from_labels(
            estimator.labels_,
            diagnostics=diagnostics(type(self).__name__, metric, parameters),
            centers=estimator.cluster_centers_,
            inertia=float(estimator.inertia_),
        )
