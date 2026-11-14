"""Bounded comparison evidence from one deterministic FGW matrix block."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class FGWMatrixBlock:
    """Rows/columns, values (NaN for explicit failures) and bounded solver diagnostics."""

    pairs: tuple[tuple[int, int], ...]
    values: np.ndarray
    identity: str
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=float, copy=True)
        if values.shape != (len(self.pairs),) or np.any(np.isinf(values)) or np.any(values < 0):
            raise ValueError("Block values must match pairs and be nonnegative or explicitly NaN.")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    @property
    def complete(self) -> bool:
        return bool(np.isfinite(self.values).all())
