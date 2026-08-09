"""Durable exact or censored memory evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping


@dataclass(frozen=True, slots=True)
class AdaptiveMemoryObservation:
    """Record a measured peak or a lower bound caused by OOM."""

    config_id: str
    parameters: Mapping[str, Any]
    resource_features: Mapping[str, Any] = field(default_factory=dict)
    peak_bytes: int | None = None
    lower_bound_bytes: int | None = None
    censored: bool = False
    source: str = "training"
    observed_at_utc: str = ""

    def __post_init__(self) -> None:
        if not self.config_id or not self.source:
            raise ValueError("Memory observations require identifiers and a source.")
        if self.peak_bytes is not None and self.peak_bytes < 0:
            raise ValueError("Measured memory cannot be negative.")
        if self.lower_bound_bytes is not None and self.lower_bound_bytes < 0:
            raise ValueError("Memory lower bounds cannot be negative.")
        if self.censored and self.lower_bound_bytes is None:
            raise ValueError("Censored memory evidence requires a lower bound.")
        if self.peak_bytes is None and self.lower_bound_bytes is None:
            raise ValueError("Memory evidence requires a peak or lower bound.")
        object.__setattr__(self, "parameters", FrozenJsonMapping(self.parameters))
        object.__setattr__(self, "resource_features", FrozenJsonMapping(self.resource_features))
        if not self.observed_at_utc:
            object.__setattr__(self, "observed_at_utc", datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible evidence envelope."""
        return {
            "config_id": self.config_id,
            "parameters": dict(self.parameters),
            "resource_features": dict(self.resource_features),
            "peak_bytes": self.peak_bytes,
            "lower_bound_bytes": self.lower_bound_bytes,
            "censored": self.censored,
            "source": self.source,
            "observed_at_utc": self.observed_at_utc,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AdaptiveMemoryObservation:
        """Restore persisted memory evidence."""
        return cls(
            config_id=str(value["config_id"]),
            parameters=value.get("parameters", {}),
            resource_features=value.get("resource_features", {}),
            peak_bytes=(int(value["peak_bytes"]) if value.get("peak_bytes") is not None else None),
            lower_bound_bytes=(
                int(value["lower_bound_bytes"])
                if value.get("lower_bound_bytes") is not None
                else None
            ),
            censored=bool(value.get("censored", False)),
            source=str(value.get("source", "training")),
            observed_at_utc=str(value.get("observed_at_utc", "")),
        )
