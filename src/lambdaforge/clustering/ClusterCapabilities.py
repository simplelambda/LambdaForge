"""Native ClusterCapabilities contract."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClusterCapabilities:
    """Explicit scientific and result capabilities of one clustering algorithm."""

    native_metrics: frozenset[str]
    supports_precomputed: bool
    supports_centers: bool = False
    supports_prediction: bool = False
    supports_noise: bool = False
    supports_probabilities: bool = False
