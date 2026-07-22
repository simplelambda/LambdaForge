"""Normal-approximation confidence interval for a sample mean."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from lambdaforge.experiments.statistics.intervals.ConfidenceIntervalEstimator import (
    ConfidenceIntervalEstimator,
)
from lambdaforge.experiments.statistics.intervals.ConfidenceIntervalMethod import (
    ConfidenceIntervalMethod,
)
from lambdaforge.experiments.statistics.intervals.ConfidenceIntervalResult import (
    ConfidenceIntervalResult,
)


class NormalConfidenceInterval(ConfidenceIntervalEstimator):
    """Estimate a two-sided normal interval around the arithmetic mean."""

    def __init__(self, confidence_level: float = 0.95) -> None:
        if not 0.0 < float(confidence_level) < 1.0:
            raise ValueError("confidence_level must be strictly between 0 and 1.")
        self.confidence_level = float(confidence_level)

    def compute(
        self,
        values: Sequence[float],
        *,
        identity: Sequence[str] = (),
    ) -> ConfidenceIntervalResult:
        """Return a normal interval, or an explicit insufficient-sample result."""
        del identity
        numeric = [float(value) for value in values]
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("Confidence-interval values must all be finite.")

        n_samples = len(numeric)
        estimate = statistics.fmean(numeric) if numeric else None
        if n_samples < 2:
            return ConfidenceIntervalResult(
                estimate=estimate,
                lower=None,
                upper=None,
                standard_error=None,
                method=ConfidenceIntervalMethod.NORMAL.value,
                confidence_level=self.confidence_level,
                n_samples=n_samples,
                status="unavailable",
                reason="insufficient_samples",
            )

        mean = statistics.fmean(numeric)
        standard_error = statistics.stdev(numeric) / math.sqrt(n_samples)
        probability = 0.5 + self.confidence_level / 2.0
        critical_value = statistics.NormalDist().inv_cdf(probability)
        half_width = critical_value * standard_error
        return ConfidenceIntervalResult(
            estimate=mean,
            lower=mean - half_width,
            upper=mean + half_width,
            standard_error=standard_error,
            method=ConfidenceIntervalMethod.NORMAL.value,
            confidence_level=self.confidence_level,
            n_samples=n_samples,
            status="ok",
            degenerate=standard_error == 0.0,
        )
