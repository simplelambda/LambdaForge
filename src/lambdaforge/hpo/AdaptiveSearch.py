"""Small provider-neutral adaptive study policy and scoring helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AdaptiveSearchPolicy:
    """Successive-halving policy over candidates and progressively allocated seeds."""

    runs_per_gpu: int = 1
    max_parallel: int | None = None
    min_seeds: int = 1
    reduction_factor: int = 2
    confidence: float = 1.0
    early_stopping: bool = True
    early_stopping_min_step: int = 3

    def __post_init__(self) -> None:
        for name in ("runs_per_gpu", "min_seeds", "reduction_factor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"search.{name} must be a positive integer.")
        if self.reduction_factor < 2:
            raise ValueError("search.reduction_factor must be >= 2.")
        if self.max_parallel is not None and (
            isinstance(self.max_parallel, bool) or self.max_parallel < 1
        ):
            raise ValueError("search.max_parallel must be a positive integer or null.")
        if not math.isfinite(self.confidence) or self.confidence < 0:
            raise ValueError("search.confidence must be a finite non-negative number.")
        if self.early_stopping_min_step < 1:
            raise ValueError("search.early_stopping.min_step must be >= 1.")

    @classmethod
    def from_search(cls, value: Mapping[str, Any]) -> AdaptiveSearchPolicy:
        raw_early = value.get("early_stopping", True)
        if isinstance(raw_early, bool):
            early, min_step = raw_early, 3
        elif isinstance(raw_early, Mapping):
            unknown = set(raw_early) - {"enabled", "min_step"}
            if unknown:
                raise ValueError(f"Unknown search.early_stopping field(s): {sorted(unknown)}.")
            early = bool(raw_early.get("enabled", True))
            min_step = int(raw_early.get("min_step", 3))
        else:
            raise TypeError("search.early_stopping must be true/false or a mapping.")
        maximum = value.get("max_parallel")
        return cls(
            runs_per_gpu=int(value.get("runs_per_gpu", 1)),
            max_parallel=int(maximum) if maximum is not None else None,
            min_seeds=int(value.get("min_seeds", 1)),
            reduction_factor=int(value.get("reduction_factor", 2)),
            confidence=float(value.get("confidence", 1.0)),
            early_stopping=early,
            early_stopping_min_step=min_step,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": "adaptive",
            "runs_per_gpu": self.runs_per_gpu,
            "max_parallel": self.max_parallel,
            "min_seeds": self.min_seeds,
            "reduction_factor": self.reduction_factor,
            "confidence": self.confidence,
            "early_stopping": {
                "enabled": self.early_stopping,
                "min_step": self.early_stopping_min_step,
            },
        }


SEARCH_POLICY_FIELDS = frozenset(
    {
        "strategy",
        "runs_per_gpu",
        "max_parallel",
        "min_seeds",
        "reduction_factor",
        "confidence",
        "early_stopping",
    }
)


__all__ = ["AdaptiveSearchPolicy", "SEARCH_POLICY_FIELDS"]
