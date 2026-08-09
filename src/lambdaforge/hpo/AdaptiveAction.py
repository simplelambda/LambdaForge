"""Immutable adaptive-controller action."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping
from lambdaforge.hpo.AdaptiveActionKind import AdaptiveActionKind
from lambdaforge.hpo.AdaptivePhase import AdaptivePhase


@dataclass(frozen=True, slots=True)
class AdaptiveAction:
    """Describe one comparable use of the next compute allocation."""

    action_id: str
    kind: AdaptiveActionKind
    config_id: str
    parameters: Mapping[str, Any]
    seed: int
    current_budget: int
    target_budget: int
    phase: AdaptivePhase = AdaptivePhase.SEARCH
    information_gain: float = 0.0
    predicted_cost: float = 1.0
    feasibility_probability: float = 1.0
    utility: float = 0.0
    memory_reservation_bytes: int = 0
    reasons: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.action_id or not self.config_id:
            raise ValueError("Adaptive action identifiers cannot be empty.")
        if self.current_budget < 0 or self.target_budget <= self.current_budget:
            if self.kind not in {AdaptiveActionKind.DROP, AdaptiveActionKind.PAUSE}:
                raise ValueError("Adaptive action target budget must exceed current budget.")
        if self.predicted_cost <= 0:
            raise ValueError("Adaptive action predicted cost must be positive.")
        if not 0.0 <= self.feasibility_probability <= 1.0:
            raise ValueError("Adaptive action feasibility probability must be in [0, 1].")
        if self.memory_reservation_bytes < 0:
            raise ValueError("Adaptive action memory reservation cannot be negative.")
        object.__setattr__(self, "parameters", FrozenJsonMapping(self.parameters))
        object.__setattr__(self, "reasons", FrozenJsonMapping(self.reasons))

    def with_scores(
        self,
        *,
        information_gain: float,
        predicted_cost: float,
        feasibility_probability: float,
        memory_reservation_bytes: int,
        reasons: Mapping[str, Any],
    ) -> AdaptiveAction:
        """Return a scored copy using cost-aware constrained utility."""
        utility = information_gain / max(predicted_cost, 1e-12) * feasibility_probability
        return AdaptiveAction(
            self.action_id,
            self.kind,
            self.config_id,
            self.parameters,
            self.seed,
            self.current_budget,
            self.target_budget,
            self.phase,
            information_gain,
            predicted_cost,
            feasibility_probability,
            utility,
            memory_reservation_bytes,
            reasons,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a durable JSON-compatible action envelope."""
        return {
            "action_id": self.action_id,
            "kind": self.kind.value,
            "config_id": self.config_id,
            "parameters": dict(self.parameters),
            "seed": self.seed,
            "current_budget": self.current_budget,
            "target_budget": self.target_budget,
            "phase": self.phase.value,
            "information_gain": self.information_gain,
            "predicted_cost": self.predicted_cost,
            "feasibility_probability": self.feasibility_probability,
            "utility": self.utility,
            "memory_reservation_bytes": self.memory_reservation_bytes,
            "reasons": dict(self.reasons),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AdaptiveAction:
        """Restore one persisted action."""
        return cls(
            action_id=str(value["action_id"]),
            kind=AdaptiveActionKind(str(value["kind"])),
            config_id=str(value["config_id"]),
            parameters=value.get("parameters", {}),
            seed=int(value["seed"]),
            current_budget=int(value["current_budget"]),
            target_budget=int(value["target_budget"]),
            phase=AdaptivePhase(str(value.get("phase", "search"))),
            information_gain=float(value.get("information_gain", 0.0)),
            predicted_cost=float(value.get("predicted_cost", 1.0)),
            feasibility_probability=float(value.get("feasibility_probability", 1.0)),
            utility=float(value.get("utility", 0.0)),
            memory_reservation_bytes=int(value.get("memory_reservation_bytes", 0)),
            reasons=value.get("reasons", {}),
        )
