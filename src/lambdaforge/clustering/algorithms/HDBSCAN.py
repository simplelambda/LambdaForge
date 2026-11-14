"""HDBSCAN adapter with the shared native clustering contract."""

from __future__ import annotations

import numpy as np

from lambdaforge.nn.distances import Distance

from .._types import ArrayInput
from ..backend import adapt_distance, clustering_array, diagnostics, require_sklearn, thread_limit
from ..ClusterCapabilities import ClusterCapabilities
from ..Clusterer import Clusterer
from ..ClusteringResult import ClusteringResult
from ._metrics import DENSITY_METRICS


class HDBSCAN(Clusterer):
    """Hierarchical density clustering backed by scikit-learn's HDBSCAN."""

    capabilities = ClusterCapabilities(
        DENSITY_METRICS,
        True,
        supports_noise=True,
        supports_probabilities=True,
    )

    def __init__(
        self,
        *,
        min_cluster_size: int = 5,
        min_samples: int | None = None,
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_method: str = "eom",
        allow_single_cluster: bool = False,
        distance: str | Distance = "euclidean",
        threads: int | None = None,
        max_pairwise_bytes: int | None = 512 * 1024 * 1024,
    ) -> None:
        if (
            isinstance(min_cluster_size, bool)
            or not isinstance(min_cluster_size, int)
            or min_cluster_size < 2
        ):
            raise ValueError("min_cluster_size must be an integer >= 2.")
        if min_samples is not None and (
            isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 1
        ):
            raise ValueError("min_samples must be an integer >= 1 or null.")
        if cluster_selection_method not in {"eom", "leaf"}:
            raise ValueError("cluster_selection_method must be 'eom' or 'leaf'.")
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = float(cluster_selection_epsilon)
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.distance = distance
        self.threads = threads
        self.max_pairwise_bytes = max_pairwise_bytes

    def cluster(self, features: ArrayInput) -> ClusteringResult:
        require_sklearn(type(self).__name__)
        from sklearn.cluster import HDBSCAN as BackendHDBSCAN

        values = clustering_array(features, self.distance, self.max_pairwise_bytes)
        if len(values) < self.min_cluster_size:
            raise ValueError("min_cluster_size cannot exceed the number of samples.")
        metric, backend_values = adapt_distance(
            self.distance,
            self.capabilities,
            values,
            algorithm=type(self).__name__,
            max_pairwise_bytes=self.max_pairwise_bytes,
        )
        parameters = {
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "cluster_selection_epsilon": self.cluster_selection_epsilon,
            "cluster_selection_method": self.cluster_selection_method,
            "allow_single_cluster": self.allow_single_cluster,
            "threads": self.threads,
        }
        with thread_limit(self.threads):
            estimator = BackendHDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                cluster_selection_method=self.cluster_selection_method,
                allow_single_cluster=self.allow_single_cluster,
                metric=metric,
                n_jobs=self.threads,
                copy=True,
            ).fit(backend_values)
        return ClusteringResult.from_labels(
            estimator.labels_,
            diagnostics=diagnostics(type(self).__name__, metric, parameters),
            probabilities=np.asarray(estimator.probabilities_),
        )
