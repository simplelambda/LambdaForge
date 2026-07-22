"""Bounded-memory percentile bootstrap interval for a paired mean."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence

import numpy as np

from lambdaforge.experiments.statistics.intervals.ConfidenceIntervalEstimator import (
    ConfidenceIntervalEstimator,
)
from lambdaforge.experiments.statistics.intervals.ConfidenceIntervalMethod import (
    ConfidenceIntervalMethod,
)
from lambdaforge.experiments.statistics.intervals.ConfidenceIntervalResult import (
    ConfidenceIntervalResult,
)


class BootstrapConfidenceInterval(ConfidenceIntervalEstimator):
    """Bootstrap a mean reproducibly while bounding transient index memory.

    A stream seed is derived from the configured base seed and a canonical
    comparison identity. Existing comparisons therefore keep the same interval
    when metrics are reordered or unrelated comparisons are added.
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        *,
        resamples: int = 10_000,
        seed: int = 0,
        batch_size: int = 1_024,
        max_batch_elements: int = 1_000_000,
    ) -> None:
        if not 0.0 < float(confidence_level) < 1.0:
            raise ValueError("confidence_level must be strictly between 0 and 1.")
        if int(resamples) < 1:
            raise ValueError("resamples must be at least 1.")
        if int(seed) < 0:
            raise ValueError("seed must be non-negative.")
        if int(batch_size) < 1:
            raise ValueError("batch_size must be at least 1.")
        if int(max_batch_elements) < 1:
            raise ValueError("max_batch_elements must be at least 1.")
        self.confidence_level = float(confidence_level)
        self.resamples = int(resamples)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.max_batch_elements = int(max_batch_elements)

    def compute(
        self,
        values: Sequence[float],
        *,
        identity: Sequence[str] = (),
    ) -> ConfidenceIntervalResult:
        """Return a deterministic percentile interval for the sample mean."""
        numeric = np.asarray([float(value) for value in values], dtype=np.float64)
        if numeric.ndim != 1 or not bool(np.isfinite(numeric).all()):
            raise ValueError("Bootstrap values must be a finite one-dimensional sample.")

        n_samples = int(numeric.size)
        estimate = float(numeric.mean()) if n_samples else None
        effective_seed = self._effective_seed(identity)
        if n_samples < 2:
            return ConfidenceIntervalResult(
                estimate=estimate,
                lower=None,
                upper=None,
                standard_error=None,
                method=ConfidenceIntervalMethod.BOOTSTRAP_PERCENTILE.value,
                confidence_level=self.confidence_level,
                n_samples=n_samples,
                status="unavailable",
                reason="insufficient_samples",
                resamples=self.resamples,
                base_seed=self.seed,
                effective_seed=effective_seed,
                batch_size=self.batch_size,
                max_batch_elements=self.max_batch_elements,
            )

        if bool(np.all(numeric == numeric[0])):
            constant = float(numeric[0])
            return ConfidenceIntervalResult(
                estimate=constant,
                lower=constant,
                upper=constant,
                standard_error=0.0,
                method=ConfidenceIntervalMethod.BOOTSTRAP_PERCENTILE.value,
                confidence_level=self.confidence_level,
                n_samples=n_samples,
                status="ok",
                resamples=self.resamples,
                base_seed=self.seed,
                effective_seed=effective_seed,
                batch_size=self.batch_size,
                max_batch_elements=self.max_batch_elements,
                degenerate=True,
            )

        generator = np.random.Generator(np.random.PCG64(effective_seed))
        means = np.empty(self.resamples, dtype=np.float64)
        offset = 0
        bounded_batch_size = min(
            self.batch_size,
            max(1, self.max_batch_elements // n_samples),
        )
        while offset < self.resamples:
            count = min(bounded_batch_size, self.resamples - offset)
            indices = generator.integers(0, n_samples, size=(count, n_samples))
            means[offset : offset + count] = numeric[indices].mean(axis=1)
            offset += count

        tail = (1.0 - self.confidence_level) / 2.0
        lower, upper = np.quantile(means, [tail, 1.0 - tail], method="linear")
        standard_error = float(means.std(ddof=1)) if self.resamples > 1 else 0.0
        return ConfidenceIntervalResult(
            estimate=estimate,
            lower=float(lower),
            upper=float(upper),
            standard_error=standard_error,
            method=ConfidenceIntervalMethod.BOOTSTRAP_PERCENTILE.value,
            confidence_level=self.confidence_level,
            n_samples=n_samples,
            status="ok",
            resamples=self.resamples,
            base_seed=self.seed,
            effective_seed=effective_seed,
            batch_size=self.batch_size,
            max_batch_elements=self.max_batch_elements,
            degenerate=math.isclose(float(lower), float(upper), rel_tol=0.0, abs_tol=0.0),
        )

    def _effective_seed(self, identity: Sequence[str]) -> int:
        payload = json.dumps(
            {"seed": self.seed, "identity": [str(part) for part in identity]},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)
