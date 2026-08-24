"""Focused scikit-learn adapters behind LambdaForge clustering contracts."""

from __future__ import annotations

import numpy as np

from lambdaforge.nn.distances import Distance

from .backend import adapt_distance, diagnostics, feature_array, require_sklearn, thread_limit
from .base import ArrayInput, ClusterCapabilities, Clusterer, ClusteringResult

_EUCLIDEAN = frozenset({"euclidean", "squared_euclidean"})
_DENSITY_METRICS = frozenset({"euclidean", "manhattan", "minkowski", "chebyshev", "cosine"})


class KMeans(Clusterer):
    """Centroid clustering with scientifically explicit Euclidean semantics."""

    capabilities = ClusterCapabilities(_EUCLIDEAN, False, supports_centers=True)

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


class DBSCAN(Clusterer):
    """Density clustering with explicit noise labels and optional custom distances."""

    capabilities = ClusterCapabilities(
        _DENSITY_METRICS,
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

        values = feature_array(features)
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


class HDBSCAN(Clusterer):
    """Hierarchical density clustering backed by scikit-learn's HDBSCAN."""

    capabilities = ClusterCapabilities(
        _DENSITY_METRICS,
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

        values = feature_array(features)
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
            ).fit(backend_values)
        return ClusteringResult.from_labels(
            estimator.labels_,
            diagnostics=diagnostics(type(self).__name__, metric, parameters),
            probabilities=np.asarray(estimator.probabilities_),
        )


class Agglomerative(Clusterer):
    """Agglomerative clustering with explicit linkage/distance compatibility."""

    capabilities = ClusterCapabilities(_DENSITY_METRICS, True)

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


__all__ = ["Agglomerative", "DBSCAN", "HDBSCAN", "KMeans", "MiniBatchKMeans"]
