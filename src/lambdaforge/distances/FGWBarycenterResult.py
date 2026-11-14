"""Synthetic relational FGW prototype and honest backend/postcheck evidence."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .FGWResult import FGWResult
from .StructuredMeasure import StructuredMeasure


@dataclass(frozen=True, slots=True)
class FGWBarycenterResult:
    """Local prototype, weighted postcheck cost and separate parameter-change history.

    comparisons are new FGW solves at the FINAL prototype, not the lagging
    transports used inside POT's last barycenter update. No global optimality or
    physical realization of the relational prototype is claimed.
    """

    prototype: StructuredMeasure
    objective: float
    converged: bool
    iterations: int
    seed: int | None
    start_seed: int
    object_weights: np.ndarray
    feature_errors: tuple[float, ...]
    structure_errors: tuple[float, ...]
    comparisons: tuple[FGWResult, ...]
    restarts: tuple[Mapping[str, Any], ...]
    warnings: tuple[str, ...]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not np.isfinite(self.objective) or self.objective < 0:
            raise ValueError("Barycenter objective must be finite and nonnegative.")
        weights = np.array(self.object_weights, dtype=float, copy=True)
        weights.setflags(write=False)
        object.__setattr__(self, "object_weights", weights)

    @property
    def n_support(self) -> int:
        """Authored prototype cardinality, not an inferred optimal size."""
        return len(self.prototype.weights)

    @property
    def alpha(self) -> float:
        """Feature/structure tradeoff used by the actual optimizer."""
        return float(self.diagnostics["alpha"])
