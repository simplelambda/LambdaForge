"""Cluster bootstrap result."""

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping


@dataclass(frozen=True, slots=True)
class ClusterBootstrapResult:
    """Describe an idempotent workspace/environment preparation."""

    cluster: str
    environment_id: str
    python: str
    reused: bool
    pytorch: Mapping[str, Any] | None = None
    runtime: Mapping[str, Any] | None = None
    planned: bool = False

    def __post_init__(self) -> None:
        if self.pytorch is not None:
            object.__setattr__(self, "pytorch", FrozenJsonMapping(self.pytorch))
        if self.runtime is not None:
            object.__setattr__(self, "runtime", FrozenJsonMapping(self.runtime))

    def to_dict(self) -> dict[str, Any]:
        """Return a machine-readable result."""
        return {
            "cluster": self.cluster,
            "environment_id": self.environment_id,
            "python": self.python,
            "reused": self.reused,
            "pytorch": copy.deepcopy(self.pytorch),
            "runtime": copy.deepcopy(self.runtime),
            "planned": self.planned,
        }
