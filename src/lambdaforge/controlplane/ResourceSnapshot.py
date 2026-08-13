"""Provider-neutral cluster resource observation."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """Keep capacity, usage and scheduler assertions explicitly distinct."""

    cluster: str
    online: bool
    scheduler: str
    observed: Mapping[str, Any] = field(default_factory=dict)
    available: Mapping[str, Any] = field(default_factory=dict)
    scheduler_view: Mapping[str, Any] = field(default_factory=dict)
    requested: tuple[Mapping[str, Any], ...] = ()
    observed_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed", FrozenJsonMapping(self.observed))
        object.__setattr__(self, "available", FrozenJsonMapping(self.available))
        object.__setattr__(self, "scheduler_view", FrozenJsonMapping(self.scheduler_view))
        object.__setattr__(
            self, "requested", tuple(FrozenJsonMapping(value) for value in self.requested)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster": self.cluster,
            "online": self.online,
            "scheduler": self.scheduler,
            "observed": copy.deepcopy(self.observed),
            "available": copy.deepcopy(self.available),
            "scheduler_view": copy.deepcopy(self.scheduler_view),
            "requested": [copy.deepcopy(value) for value in self.requested],
            "observed_at_utc": self.observed_at_utc,
            "error": self.error,
        }
