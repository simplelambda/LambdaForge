"""Native Clusterer contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ._types import ArrayInput
from .ClusterCapabilities import ClusterCapabilities
from .ClusteringResult import ClusteringResult


class Clusterer(ABC):
    """Common contract for non-differentiable clustering of ``[N, F]`` features."""

    capabilities: ClusterCapabilities

    @abstractmethod
    def cluster(self, features: ArrayInput) -> ClusteringResult:
        """Cluster NumPy or detached PyTorch features and return normalized evidence."""
        raise NotImplementedError
