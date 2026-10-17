"""Adjusted-Rand evidence across explicitly selected clusterers."""

from __future__ import annotations

from dataclasses import dataclass

from .ClusteringResult import ClusteringResult


@dataclass(frozen=True, slots=True)
class StabilityResult:
    """Generic adjusted-Rand evidence across explicitly chosen clusterers."""

    scores: tuple[float, ...]
    pairwise: tuple[tuple[float, ...], ...]
    median: float
    minimum: float
    results: tuple[ClusteringResult, ...]
