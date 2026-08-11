"""Immutable workflow dry-run plan."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WorkflowPlan:
    """Describe graph order and concurrency without constructing user code."""

    name: str
    run_dir: Path
    levels: tuple[tuple[str, ...], ...]
    max_parallel: int
    placements: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible plan."""
        return {
            "kind": "workflow",
            "name": self.name,
            "run_dir": str(self.run_dir),
            "levels": [list(level) for level in self.levels],
            "max_parallel": self.max_parallel,
            "placements": dict(self.placements),
        }
