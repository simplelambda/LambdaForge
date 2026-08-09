"""Deterministic candidate memory-probe policy."""

from __future__ import annotations

from lambdaforge.hpo.AdaptiveAction import AdaptiveAction
from lambdaforge.hpo.AdaptiveOptimizerState import AdaptiveOptimizerState
from lambdaforge.hpo.FeatureAwareMemoryModel import FeatureAwareMemoryModel
from lambdaforge.hpo.MemoryCapacity import MemoryCapacity
from lambdaforge.hpo.MemoryCapacityKind import MemoryCapacityKind


class MemoryProbePolicy:
    """Request expensive probes only for cold, uncertain, OOD or boundary cases."""

    def __init__(
        self,
        *,
        mode: str = "auto",
        relative_uncertainty_threshold: float = 0.25,
        near_limit_fraction: float = 0.85,
        oom_probability_threshold: float = 0.05,
    ) -> None:
        if mode not in {"auto", "always", "never"}:
            raise ValueError("Memory probe mode must be auto, always or never.")
        if relative_uncertainty_threshold < 0:
            raise ValueError("Memory probe uncertainty threshold cannot be negative.")
        if not 0 < near_limit_fraction <= 1 or not 0 <= oom_probability_threshold <= 1:
            raise ValueError("Invalid memory probe probability/limit threshold.")
        self.mode = mode
        self.relative_uncertainty_threshold = float(relative_uncertainty_threshold)
        self.near_limit_fraction = float(near_limit_fraction)
        self.oom_probability_threshold = float(oom_probability_threshold)

    def should_probe(
        self,
        action: AdaptiveAction,
        state: AdaptiveOptimizerState,
        model: FeatureAwareMemoryModel,
        capacity: MemoryCapacity,
    ) -> bool:
        """Return a reproducible decision from persisted evidence and candidate features."""
        if self.mode == "always":
            return True
        if self.mode == "never":
            return False
        if model.requires_probe(state) or capacity.kind is MemoryCapacityKind.UNKNOWN:
            return True
        estimate = model.predict(action, state)
        relative = estimate.standard_deviation / max(estimate.mean, 1.0)
        if relative >= self.relative_uncertainty_threshold:
            return True
        if capacity.kind is MemoryCapacityKind.KNOWN:
            assert capacity.bytes is not None
            if estimate.upper >= capacity.bytes * self.near_limit_fraction:
                return True
            if 1.0 - estimate.probability_at_most(capacity.bytes) >= self.oom_probability_threshold:
                return True
        return False
