"""Read-only adaptive optimization execution plan."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping


@dataclass(frozen=True, slots=True)
class AdaptiveExperimentPlan:
    """Expose resolved search semantics and resources without creating run artifacts."""

    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", FrozenJsonMapping(self.values))

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON/YAML-friendly plan mapping."""
        return dict(self.values)
