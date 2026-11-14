"""Dense full or explicitly incomplete structured-object discrepancy matrix."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class FGWMatrixResult:
    """Matrix plus per-block diagnostics; complete means finite, not solver convergence."""

    values: np.ndarray
    identity: str
    symmetric: bool
    blocks: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=float, copy=True)
        if values.ndim != 2 or np.any(np.isinf(values)) or np.any(values < 0):
            raise ValueError("Matrix values must be nonnegative or explicitly NaN.")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    @property
    def complete(self) -> bool:
        return bool(np.isfinite(self.values).all())
