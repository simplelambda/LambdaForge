"""Conservative learned GPU-memory model with OOM censoring."""

from __future__ import annotations

from statistics import fmean, pstdev

from lambdaforge.hpo.AdaptiveAction import AdaptiveAction
from lambdaforge.hpo.AdaptiveOptimizerState import AdaptiveOptimizerState
from lambdaforge.hpo.PredictiveEstimate import PredictiveEstimate


class EmpiricalMemoryModel:
    """Estimate peak reserved VRAM, using a hard logical budget during cold start."""

    def __init__(
        self,
        *,
        cold_start_bytes: int,
        headroom_bytes: int = 0,
        safety_quantile: float = 0.99,
        min_observations: int = 3,
    ) -> None:
        if cold_start_bytes < 0 or headroom_bytes < 0 or min_observations < 1:
            raise ValueError("Invalid adaptive memory-model configuration.")
        if not 0.5 <= safety_quantile < 1.0:
            raise ValueError("Memory safety quantile must be in [0.5, 1).")
        self.cold_start_bytes = int(cold_start_bytes)
        self.headroom_bytes = int(headroom_bytes)
        self.safety_quantile = float(safety_quantile)
        self.min_observations = int(min_observations)

    def predict(self, action: AdaptiveAction, state: AdaptiveOptimizerState) -> PredictiveEstimate:
        """Return a conservative peak estimate from same-config then global evidence."""
        same = [
            observation
            for observation in state.observations_for(action.config_id)
            if observation.peak_reserved_bytes > 0
        ]
        population = same or [
            observation for observation in state.observations if observation.peak_reserved_bytes > 0
        ]
        if len(population) < self.min_observations:
            mean = float(self.cold_start_bytes)
            deviation = 0.0
            source = "logical_budget_cold_start"
        else:
            peaks = [float(observation.peak_reserved_bytes) for observation in population]
            lower_bounds = [
                float(self.cold_start_bytes)
                for observation in state.observations
                if observation.config_id == action.config_id and observation.oom
            ]
            mean = max(fmean(peaks), max(lower_bounds, default=0.0))
            deviation = pstdev(peaks) if len(peaks) > 1 else mean * 0.1
            source = "same_configuration" if same else "global_empirical"
        estimate = PredictiveEstimate.normal(
            mean,
            max(0.0, deviation),
            quantile=self.safety_quantile,
            source=source,
        )
        return PredictiveEstimate(
            estimate.mean,
            estimate.standard_deviation,
            estimate.upper + self.headroom_bytes,
            estimate.source,
        )
