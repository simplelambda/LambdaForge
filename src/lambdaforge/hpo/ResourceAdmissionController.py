"""Conservative resource admission for adaptive actions."""

from __future__ import annotations

from lambdaforge.hpo.AdaptiveAction import AdaptiveAction
from lambdaforge.hpo.AdaptiveOptimizerState import AdaptiveOptimizerState
from lambdaforge.hpo.EmpiricalMemoryModel import EmpiricalMemoryModel
from lambdaforge.hpo.PredictiveEstimate import PredictiveEstimate


class ResourceAdmissionController:
    """Combine logical memory limits and learned feasibility without CUDA assumptions."""

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
        model: EmpiricalMemoryModel,
        *,
        available_bytes: int,
    ) -> tuple[bool, float, int, PredictiveEstimate]:
        """Return admission, probability and reservation for one resource."""
        estimate = model.predict(action, state)
        if available_bytes <= 0:
            return True, 1.0, 0, estimate
        headroom = model.headroom_bytes
        predicted_peak_upper = max(0.0, estimate.upper - headroom)
        device_peak_limit = max(0, available_bytes - headroom)
        probability = estimate.probability_at_most(device_peak_limit)
        logical_admitted = True
        if self.logical_limit_bytes > 0:
            logical_admitted = predicted_peak_upper <= self.logical_limit_bytes
            probability = min(
                probability,
                estimate.probability_at_most(self.logical_limit_bytes),
            )
        reservation = max(0, int(estimate.upper))
        admitted = (
            logical_admitted
            and probability >= self.minimum_feasibility
            and reservation <= available_bytes
        )
        return admitted, probability, reservation, estimate
