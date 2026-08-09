"""Typed adaptive optimization study result."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping


@dataclass(frozen=True, slots=True)
class AdaptiveExperimentResult:
    """Return durable study paths and the factual controller summary."""

    study_dir: Path
    state_path: Path
    event_log_path: Path
    summary: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary", FrozenJsonMapping(self.summary))

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable result envelope."""
        return {
            "study_dir": str(self.study_dir),
            "state_path": str(self.state_path),
            "event_log_path": str(self.event_log_path),
            "summary": dict(self.summary),
        }
