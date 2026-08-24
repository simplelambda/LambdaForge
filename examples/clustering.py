"""Explicit clustering inside a normal Work; install ``lambdaforge[clustering]``."""

from pathlib import Path

import numpy as np

import lambdaforge as lf


class ClusterFeatures(lf.Work):
    """Load an explicit feature matrix and persist normalized clustering evidence."""

    def run(
        self,
        features: Path,
        min_cluster_size: int = 20,
        min_samples: int = 5,
    ) -> dict[str, int]:
        values = np.load(features, allow_pickle=False)
        result = lf.clustering.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            distance="euclidean",
            threads=self.resources.cpu,
        ).cluster(values)
        labels = self.outputs.file(
            "labels",
            filename="labels.npy",
            role="cluster-labels",
            media_type="application/x-npy",
        )
        labels.build(lambda target: np.save(target, result.labels, allow_pickle=False))
        self.metrics.log("clusters", result.n_clusters)
        self.metrics.log("noise_fraction", result.noise_fraction)
        return {"clusters": result.n_clusters, "noise": result.noise_count}
