"""Approximate resampling uncertainty with declared independent sampling units."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Approximate percentile interval with its independent resampling unit."""

    estimate: float
    interval: tuple[float, float]
    confidence_level: float
    method: str
    n_resamples: int
    seed: int | None
    unit: str
    n_units: int
    diagnostics: Mapping[str, Any]
