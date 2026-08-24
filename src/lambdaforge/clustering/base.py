"""Backend-independent public clustering contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

ArrayInput = np.ndarray | torch.Tensor


@dataclass(frozen=True, slots=True)
class ClusterCapabilities:
    """Explicit scientific and result capabilities of one clustering algorithm."""

    native_metrics: frozenset[str]
    supports_precomputed: bool
    supports_centers: bool = False
    supports_prediction: bool = False
    supports_noise: bool = False
    supports_probabilities: bool = False


@dataclass(frozen=True, slots=True)
class ClusteringResult:
    """Immutable normalized result returned by every LambdaForge clusterer."""

    labels: np.ndarray
    n_clusters: int
    noise_count: int
    noise_fraction: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    probabilities: np.ndarray | None = None
    centers: np.ndarray | None = None
    inertia: float | None = None

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels, dtype=np.int64).copy()
        if labels.ndim != 1:
            raise ValueError("Clustering labels must have shape [N].")
        expected_noise = int(np.count_nonzero(labels == -1))
        expected_clusters = len({int(value) for value in labels.tolist() if int(value) != -1})
        expected_fraction = expected_noise / len(labels) if len(labels) else 0.0
        if self.n_clusters != expected_clusters:
            raise ValueError("n_clusters does not match the distinct non-noise labels.")
        if self.noise_count != expected_noise or not np.isclose(
            self.noise_fraction, expected_fraction
        ):
            raise ValueError("Noise summary does not match labels.")
        labels.setflags(write=False)
        object.__setattr__(self, "labels", labels)
        for name in ("probabilities", "centers"):
            value = getattr(self, name)
            if value is not None:
                frozen = np.asarray(value).copy()
                if name == "probabilities" and (
                    frozen.shape != labels.shape
                    or not np.isfinite(frozen).all()
                    or np.any((frozen < 0) | (frozen > 1))
                ):
                    raise ValueError(
                        "Clustering probabilities must be finite [N] values in [0, 1]."
                    )
                if name == "centers" and frozen.ndim != 2:
                    raise ValueError("Clustering centers must have shape [K, F].")
                frozen.setflags(write=False)
                object.__setattr__(self, name, frozen)
        if self.inertia is not None and (not np.isfinite(self.inertia) or self.inertia < 0):
            raise ValueError("Clustering inertia must be finite and non-negative.")
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    @classmethod
    def from_labels(
        cls,
        labels: np.ndarray,
        *,
        diagnostics: Mapping[str, Any],
        probabilities: np.ndarray | None = None,
        centers: np.ndarray | None = None,
        inertia: float | None = None,
    ) -> ClusteringResult:
        """Normalize labels and derive universal cluster/noise evidence."""
        selected = np.asarray(labels, dtype=np.int64)
        noise_count = int(np.count_nonzero(selected == -1))
        clusters = {int(label) for label in selected.tolist() if int(label) != -1}
        return cls(
            selected,
            len(clusters),
            noise_count,
            noise_count / len(selected) if len(selected) else 0.0,
            diagnostics,
            probabilities,
            centers,
            inertia,
        )


class Clusterer(ABC):
    """Common contract for non-differentiable clustering of ``[N, F]`` features."""

    capabilities: ClusterCapabilities

    @abstractmethod
    def cluster(self, features: ArrayInput) -> ClusteringResult:
        """Cluster NumPy or detached PyTorch features and return normalized evidence."""
        raise NotImplementedError


__all__ = ["ArrayInput", "ClusterCapabilities", "Clusterer", "ClusteringResult"]
