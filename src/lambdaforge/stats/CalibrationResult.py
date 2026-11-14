"""Evidence returned by independent-control empirical null calibration."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Inclusive empirical rank, decision and finite-reference resolution evidence."""

    score: float
    p_value: float
    reject: bool
    alpha: float
    n_controls: int
    tail: str
    ties: int
    resolution: float
    reference_level: float
    metadata: Mapping[str, Any]
    warnings: tuple[str, ...]
