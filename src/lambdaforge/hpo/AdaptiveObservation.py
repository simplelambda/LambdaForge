"""One completed adaptive action observation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping
from lambdaforge.hpo.AdaptiveTrialStatus import AdaptiveTrialStatus


@dataclass(frozen=True, slots=True)
class AdaptiveObservation:
    """Persist score, complete curve, cost and resource evidence together."""

    action_id: str
    config_id: str
    parameters: Mapping[str, Any]
    seed: int
    budget: int
    score: float | None
    curve: tuple[tuple[int, float], ...]
    status: AdaptiveTrialStatus
    seconds: float = 0.0
    gpu_seconds: float = 0.0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    oom: bool = False
    run_dir: str | None = None
    error: str | None = None
    observed_at_utc: str = ""

    def __post_init__(self) -> None:
        if self.budget < 0 or self.seconds < 0 or self.gpu_seconds < 0:
            raise ValueError("Adaptive observation budgets and costs cannot be negative.")
        if not math.isfinite(self.seconds) or not math.isfinite(self.gpu_seconds):
            raise ValueError("Adaptive observation costs must be finite.")
        if self.score is not None and not math.isfinite(self.score):
            raise ValueError("Adaptive observation score must be finite or null.")
        if any(budget < 0 or not math.isfinite(score) for budget, score in self.curve):
            raise ValueError("Adaptive learning-curve points must be finite and non-negative.")
        if self.peak_allocated_bytes < 0 or self.peak_reserved_bytes < 0:
            raise ValueError("Adaptive memory observations cannot be negative.")
        object.__setattr__(self, "parameters", FrozenJsonMapping(self.parameters))
        if not self.observed_at_utc:
            object.__setattr__(self, "observed_at_utc", datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Return a durable JSON-compatible observation."""
        return {
            "action_id": self.action_id,
            "config_id": self.config_id,
            "parameters": dict(self.parameters),
            "seed": self.seed,
            "budget": self.budget,
            "score": self.score,
            "curve": [[budget, score] for budget, score in self.curve],
            "status": self.status.value,
            "seconds": self.seconds,
            "gpu_seconds": self.gpu_seconds,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "oom": self.oom,
            "run_dir": self.run_dir,
            "error": self.error,
            "observed_at_utc": self.observed_at_utc,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AdaptiveObservation:
        """Restore an observation from persistent controller state."""
        raw_curve = value.get("curve", ())
        if not isinstance(raw_curve, Sequence):
            raise TypeError("Adaptive observation curve must be a sequence.")
        return cls(
            action_id=str(value["action_id"]),
            config_id=str(value["config_id"]),
            parameters=value.get("parameters", {}),
            seed=int(value["seed"]),
            budget=int(value["budget"]),
            score=float(value["score"]) if value.get("score") is not None else None,
            curve=tuple((int(item[0]), float(item[1])) for item in raw_curve),
            status=AdaptiveTrialStatus(str(value["status"])),
            seconds=float(value.get("seconds", 0.0)),
            gpu_seconds=float(value.get("gpu_seconds", 0.0)),
            peak_allocated_bytes=int(value.get("peak_allocated_bytes", 0)),
            peak_reserved_bytes=int(value.get("peak_reserved_bytes", 0)),
            oom=bool(value.get("oom", False)),
            run_dir=str(value["run_dir"]) if value.get("run_dir") is not None else None,
            error=str(value["error"]) if value.get("error") is not None else None,
            observed_at_utc=str(value.get("observed_at_utc", "")),
        )
