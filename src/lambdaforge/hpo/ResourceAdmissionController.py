"""Conservative resource admission for adaptive actions."""

from __future__ import annotations

import math

from lambdaforge.hpo.AdaptiveAction import AdaptiveAction
from lambdaforge.hpo.AdaptiveOptimizerState import AdaptiveOptimizerState
from lambdaforge.hpo.FeatureAwareMemoryModel import FeatureAwareMemoryModel
from lambdaforge.hpo.MemoryCapacity import MemoryCapacity
from lambdaforge.hpo.MemoryCapacityKind import MemoryCapacityKind
from lambdaforge.hpo.PredictiveEstimate import PredictiveEstimate


class ResourceAdmissionController:
    """Apply probabilistic feasibility and hard reservation independently of scheduling."""

    def __init__(
        self,
        *,
        minimum_feasibility: float = 0.05,
        logical_limit_bytes: int = 0,
    ) -> None:
        if not 0.0 <= minimum_feasibility <= 1.0:
            raise ValueError("Minimum feasibility must be in [0, 1].")
        self.minimum_feasibility = float(minimum_feasibility)
        self.logical_limit_bytes = int(logical_limit_bytes)
        if self.logical_limit_bytes < 0:
            raise ValueError("Logical memory limit cannot be negative.")

    def assess(
        self,
        action: AdaptiveAction,
        state: AdaptiveOptimizerState,
        model: FeatureAwareMemoryModel,
        *,
        available_bytes: int | MemoryCapacity | None,
    ) -> tuple[bool, float, int, PredictiveEstimate]:
        """Return admission, fit probability and conservative byte reservation.

        ``None`` means UNKNOWN, ``MemoryCapacity.unbounded()`` is the only unbounded state, and an
        integer zero is the exact capacity KNOWN(0).
        """
        capacity = (
            available_bytes
            if isinstance(available_bytes, MemoryCapacity)
            else MemoryCapacity.unknown()
            if available_bytes is None
            else MemoryCapacity.known(available_bytes)
        )
        estimate = model.predict(action, state)
        if not math.isfinite(estimate.upper):
            raise ValueError("Predicted memory reservation must be finite.")
        reservation = max(0, math.ceil(estimate.upper))
        probability = 1.0
        hard_limit: int | None = None
        if capacity.kind is MemoryCapacityKind.UNKNOWN:
            return False, 0.0, reservation, estimate
        if capacity.kind is MemoryCapacityKind.KNOWN:
            assert capacity.bytes is not None
            hard_limit = capacity.bytes
            if estimate.upper <= 0 and estimate.source == "logical_budget_cold_start":
                reservation = capacity.bytes
        if self.logical_limit_bytes > 0:
            hard_limit = (
                self.logical_limit_bytes
                if hard_limit is None
                else min(hard_limit, self.logical_limit_bytes)
            )
        if hard_limit is not None:
            headroom = int(getattr(model, "headroom_bytes", 0))
            probability = estimate.probability_at_most(max(0, hard_limit - headroom))
            admitted = reservation <= hard_limit and probability >= self.minimum_feasibility
        else:
            admitted = True
        return admitted, probability, reservation, estimate
