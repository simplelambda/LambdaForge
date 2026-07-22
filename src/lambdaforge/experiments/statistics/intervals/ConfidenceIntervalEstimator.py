"""Object contract for confidence-interval strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from lambdaforge.experiments.statistics.intervals.ConfidenceIntervalResult import (
    ConfidenceIntervalResult,
)


class ConfidenceIntervalEstimator(ABC):
    """Estimate an interval without retaining observations between calls."""

    @abstractmethod
    def compute(
        self,
        values: Sequence[float],
        *,
        identity: Sequence[str] = (),
    ) -> ConfidenceIntervalResult:
        """Estimate an interval for one finite sample."""
