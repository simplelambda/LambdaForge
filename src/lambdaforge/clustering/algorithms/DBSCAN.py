"""DBSCAN adapter with the shared native clustering contract."""

from __future__ import annotations

from lambdaforge.nn.distances import Distance

from .._types import ArrayInput
from ..backend import adapt_distance, clustering_array, diagnostics, require_sklearn, thread_limit
from ..ClusterCapabilities import ClusterCapabilities
from ..Clusterer import Clusterer
from ..ClusteringResult import ClusteringResult
from ._metrics import DENSITY_METRICS


class DBSCAN(Clusterer):
    """Density clustering with explicit noise labels and optional custom distances."""

    capabilities = ClusterCapabilities(
        DENSITY_METRICS,
        True,
        supports_noise=True,
    )

    def __init__(
        self,
        *,
        eps: float = 0.5,
        min_samples: int = 5,
        distance: str | Distance = "euclidean",
        threads: int | None = None,
        max_pairwise_bytes: int | None = 512 * 1024 * 1024,
    ) -> None:
        if eps <= 0:
            raise ValueError("eps must be greater than zero.")
        if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 1:
            raise ValueError("min_samples must be an integer >= 1.")
        self.eps = float(eps)
        self.min_samples = min_samples
        self.distance = distance
        self.threads = threads
        self.max_pairwise_bytes = max_pairwise_bytes

    def cluster(self, features: ArrayInput) -> ClusteringResult:
        require_sklearn(type(self).__name__)
        from sklearn.cluster import DBSCAN as BackendDBSCAN

        values = clustering_array(features, self.distance, self.max_pairwise_bytes)
        metric, backend_values = adapt_distance(
            self.distance,
            self.capabilities,
            values,
            algorithm=type(self).__name__,
            max_pairwise_bytes=self.max_pairwise_bytes,
        )
        parameters = {
            "eps": self.eps,
            "min_samples": self.min_samples,
            "threads": self.threads,
        }
        with thread_limit(self.threads):
            labels = BackendDBSCAN(
                eps=self.eps,
                min_samples=self.min_samples,
                metric=metric,
                n_jobs=self.threads,
            ).fit_predict(backend_values)
        return ClusteringResult.from_labels(
            labels,
            diagnostics=diagnostics(type(self).__name__, metric, parameters),
        )
