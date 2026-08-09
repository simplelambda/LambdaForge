"""Typed terminal workflow result."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    """Summarize node results while retaining independent branch outcomes."""

    name: str
    run_dir: Path
    status: str
    nodes: Mapping[str, Mapping[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return a defensive JSON-compatible result."""
        return {
            "kind": "workflow",
            "name": self.name,
            "run_dir": str(self.run_dir),
            "status": self.status,
            "nodes": copy.deepcopy(dict(self.nodes)),
        }
