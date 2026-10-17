"""Inspectable finite-object transport evidence, not a clustering assignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FGWResult:
    """Unrooted q=2 FGW objective and its feasible coupling.

    ``value = (1-alpha)*feature_cost + alpha*structure_cost``. Costs are unweighted
    squared discrepancies. Convergence refers only to the local conditional-gradient
    stopping criterion, never a global optimality guarantee. The NumPy CPU solve is
    non-differentiable; no PyTorch autograd/device behavior is implied.
    """

    value: float
    transport: np.ndarray
    feature_cost: float
    structure_cost: float
    alpha: float
    converged: bool
    iterations: int
    objective_history: tuple[float, ...]
    marginal_error: float
    backend: str = "POT conditional gradient (q=2, square_loss)"

    def __post_init__(self) -> None:
        transport = np.array(self.transport, dtype=np.float64, copy=True)
        if transport.ndim != 2 or not np.isfinite(transport).all() or np.any(transport < 0):
            raise ValueError("FGW transport must be a finite nonnegative matrix.")
        if not all(
            np.isfinite(value) and value >= 0
            for value in (
                self.value,
                self.feature_cost,
                self.structure_cost,
                self.marginal_error,
            )
        ):
            raise ValueError("FGW costs and marginal error must be finite and nonnegative.")
        transport.setflags(write=False)
        object.__setattr__(self, "transport", transport)
