"""Immutable result returned by confidence-interval estimators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ConfidenceIntervalResult:
    """Describe an interval and enough provenance to reproduce it."""

    estimate: float | None
    lower: float | None
    upper: float | None
    standard_error: float | None
    method: str
    confidence_level: float
    n_samples: int
    status: str
    reason: str | None = None
    resamples: int | None = None
    base_seed: int | None = None
    effective_seed: int | None = None
    batch_size: int | None = None
    max_batch_elements: int | None = None
    degenerate: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)
