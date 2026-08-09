"""Durable replay state for adaptive experiment optimization."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from lambdaforge.hpo.AdaptiveAction import AdaptiveAction
from lambdaforge.hpo.AdaptiveObservation import AdaptiveObservation
from lambdaforge.hpo.AdaptivePhase import AdaptivePhase


class AdaptiveOptimizerState:
    """Own atomic, versioned controller knowledge and deterministic counters."""

    VERSION = 1

    def __init__(
        self,
        *,
        study_fingerprint: str,
        controller_seed: int,
        phase: AdaptivePhase = AdaptivePhase.SEARCH,
        decision_index: int = 0,
        configurations: Mapping[str, Mapping[str, Any]] | None = None,
        pending_actions: Mapping[str, AdaptiveAction] | None = None,
        completed_actions: tuple[AdaptiveAction, ...] = (),
        observations: tuple[AdaptiveObservation, ...] = (),
        dropped_configurations: tuple[str, ...] = (),
        total_epochs: int = 0,
        total_gpu_seconds: float = 0.0,
        fallback_count: int = 0,
    ) -> None:
        self.study_fingerprint = study_fingerprint
        self.controller_seed = int(controller_seed)
        self.phase = phase
        self.decision_index = int(decision_index)
        self.configurations = {key: dict(value) for key, value in (configurations or {}).items()}
        self.pending_actions = dict(pending_actions or {})
        self.completed_actions = list(completed_actions)
        self.observations = list(observations)
        self.dropped_configurations = set(dropped_configurations)
        self.total_epochs = int(total_epochs)
        self.total_gpu_seconds = float(total_gpu_seconds)
        self.fallback_count = int(fallback_count)

    def next_action_id(self) -> str:
        """Advance and return a deterministic decision identifier."""
        self.decision_index += 1
        return f"decision-{self.decision_index:06d}"

    def register_pending(self, action: AdaptiveAction) -> None:
        """Record one dispatched action and its configuration."""
        if action.action_id in self.pending_actions:
            raise ValueError(f"Adaptive action {action.action_id!r} is already pending.")
        self.configurations.setdefault(action.config_id, dict(action.parameters))
        self.pending_actions[action.action_id] = action

    def complete(self, observation: AdaptiveObservation) -> AdaptiveAction:
        """Move one pending action to history and account only newly executed budget."""
        try:
            action = self.pending_actions.pop(observation.action_id)
        except KeyError as error:
            raise KeyError(f"Unknown pending adaptive action {observation.action_id!r}.") from error
        self.completed_actions.append(action)
        self.observations.append(observation)
        self.total_epochs += max(0, observation.budget - action.current_budget)
        self.total_gpu_seconds += observation.gpu_seconds
        return action

    def observations_for(
        self, config_id: str, *, seed: int | None = None
    ) -> tuple[AdaptiveObservation, ...]:
        """Return ordered observations for one configuration and optional seed."""
        return tuple(
            observation
            for observation in self.observations
            if observation.config_id == config_id and (seed is None or observation.seed == seed)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a complete replay envelope."""
        return {
            "optimizer_state_version": self.VERSION,
            "study_fingerprint": self.study_fingerprint,
            "controller_seed": self.controller_seed,
            "phase": self.phase.value,
            "decision_index": self.decision_index,
            "configurations": self.configurations,
            "pending_actions": {
                key: action.to_dict() for key, action in self.pending_actions.items()
            },
            "completed_actions": [action.to_dict() for action in self.completed_actions],
            "observations": [observation.to_dict() for observation in self.observations],
            "dropped_configurations": sorted(self.dropped_configurations),
            "total_epochs": self.total_epochs,
            "total_gpu_seconds": self.total_gpu_seconds,
            "fallback_count": self.fallback_count,
        }

    def save(self, path: str | Path) -> Path:
        """Atomically persist complete controller state beside the destination."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(self.to_dict(), stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> AdaptiveOptimizerState:
        """Load a compatible state envelope or fail without guessing migrations."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("optimizer_state_version") != cls.VERSION:
            raise ValueError("Unsupported adaptive optimizer state version.")
        return cls(
            study_fingerprint=str(payload["study_fingerprint"]),
            controller_seed=int(payload["controller_seed"]),
            phase=AdaptivePhase(str(payload.get("phase", "search"))),
            decision_index=int(payload.get("decision_index", 0)),
            configurations=payload.get("configurations", {}),
            pending_actions={
                key: AdaptiveAction.from_mapping(value)
                for key, value in payload.get("pending_actions", {}).items()
            },
            completed_actions=tuple(
                AdaptiveAction.from_mapping(value) for value in payload.get("completed_actions", ())
            ),
            observations=tuple(
                AdaptiveObservation.from_mapping(value) for value in payload.get("observations", ())
            ),
            dropped_configurations=tuple(payload.get("dropped_configurations", ())),
            total_epochs=int(payload.get("total_epochs", 0)),
            total_gpu_seconds=float(payload.get("total_gpu_seconds", 0.0)),
            fallback_count=int(payload.get("fallback_count", 0)),
        )
