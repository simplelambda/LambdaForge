"""Individually implemented strategies; existing imports stay valid."""

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from .Agglomerative import Agglomerative
    from .DBSCAN import DBSCAN
    from .HDBSCAN import HDBSCAN
    from .KMeans import KMeans
    from .MiniBatchKMeans import MiniBatchKMeans
LazyExports.install(
    __name__,
    {
        name: (f"lambdaforge.clustering.algorithms.{name}", name)
        for name in ("Agglomerative", "DBSCAN", "HDBSCAN", "KMeans", "MiniBatchKMeans")
    },
)
__all__ = ["Agglomerative", "DBSCAN", "HDBSCAN", "KMeans", "MiniBatchKMeans"]
