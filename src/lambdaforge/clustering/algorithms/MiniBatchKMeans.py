"""MiniBatchKMeans adapter with the shared native clustering contract."""

from __future__ import annotations

from lambdaforge.nn.distances import Distance

from .._types import ArrayInput
from ..backend import adapt_distance, diagnostics, feature_array, require_sklearn, thread_limit
from ..ClusteringResult import ClusteringResult
from .KMeans import KMeans


class MiniBatchKMeans(KMeans):
    """Memory-bounded mini-batch variant with the same Euclidean contract as KMeans."""

    def __init__(
        self,
        n_clusters: int,
        *,
        batch_size: int = 1024,
        distance: str | Distance = "euclidean",
        seed: int | None = None,
        n_init: int | str = "auto",
        max_iter: int = 100,
        tolerance: float = 0.0,
        threads: int | None = None,
    ) -> None:
        super().__init__(
            n_clusters,
            distance=distance,
            seed=seed,
            n_init=n_init,
            max_iter=max_iter,
            tolerance=tolerance,
            threads=threads,
        )
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be an integer >= 1.")
        self.batch_size = batch_size

    def cluster(self, features: ArrayInput) -> ClusteringResult:
        require_sklearn(type(self).__name__)
        from sklearn.cluster import MiniBatchKMeans as BackendMiniBatchKMeans

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
            "batch_size": self.batch_size,
            "seed": self.seed,
            "n_init": self.n_init,
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
            "threads": self.threads,
        }
        with thread_limit(self.threads):
            estimator = BackendMiniBatchKMeans(
                n_clusters=self.n_clusters,
                batch_size=self.batch_size,
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
