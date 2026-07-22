"""Immutable result returned by paired statistical tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PairedTestResult:
    """Expose selected and diagnostic p-values without NaN sentinels."""

    method: str
    alternative: str
    calculation_requested: str
    calculation_used: str | None
    statistic: float | None
    positive_statistic: float | None
    negative_statistic: float | None
    p_value: float | None
    p_value_two_sided: float | None
    p_value_better: float | None
    p_value_worse: float | None
    z_statistic: float | None
    n_pairs: int
    n_effective: int
    n_zero: int
    wins: int
    losses: int
    ties: int
    has_rank_ties: bool
    status: str
    reason: str | None = None
    zero_method: str | None = None
    continuity_correction: bool | None = None
    exact_max_pairs: int | None = None
    zero_tolerance: float = 0.0
    round_decimals: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)
