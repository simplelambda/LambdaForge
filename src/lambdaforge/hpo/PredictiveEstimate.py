"""Provider-neutral probabilistic scalar estimate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist


@dataclass(frozen=True, slots=True)
class PredictiveEstimate:
    """Carry mean, uncertainty and a documented conservative upper estimate."""

    mean: float
    standard_deviation: float
    upper: float
    source: str

    def probability_at_most(self, limit: float) -> float:
        """Return Gaussian feasibility probability with deterministic zero-variance handling."""
        if self.standard_deviation <= 0:
            return 1.0 if self.mean <= limit else 0.0
        return NormalDist(self.mean, self.standard_deviation).cdf(limit)

    def probability_at_least(self, threshold: float) -> float:
        """Return the probability of meeting or exceeding a scalar threshold."""
        if self.standard_deviation <= 0:
            return 1.0 if self.mean >= threshold else 0.0
        return 1.0 - NormalDist(self.mean, self.standard_deviation).cdf(threshold)

    def to_dict(self) -> dict[str, float | str]:
        """Return a structured explanation payload."""
        return {
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "upper": self.upper,
            "source": self.source,
        }

    @classmethod
    def normal(
        cls, mean: float, standard_deviation: float, *, quantile: float, source: str
    ) -> PredictiveEstimate:
        """Create an estimate with a finite Gaussian upper quantile."""
        if not math.isfinite(mean) or not math.isfinite(standard_deviation):
            raise ValueError("Predictive estimates must be finite.")
        deviation = max(0.0, standard_deviation)
        upper = mean + (NormalDist().inv_cdf(quantile) * deviation if deviation else 0.0)
        return cls(mean, deviation, upper, source)
