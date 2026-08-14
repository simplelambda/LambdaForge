"""Immutable plan, result and validation envelopes for one workflow DAG."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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


@dataclass(frozen=True, slots=True)
class WorkflowValidationReport:
    """Collect graph and ordered node facts without executing the DAG."""

    source: str | None
    name: str | None = None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    imports_checked: bool = True
    node_reports: tuple[dict[str, Any], ...] = ()

    @property
    def is_valid(self) -> bool:
        """Return whether the graph and all of its nodes are valid."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible validation envelope."""
        return {
            "source": self.source,
            "kind": "workflow",
            "name": self.name,
            "valid": self.is_valid,
            "imports_checked": self.imports_checked,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "nodes": [dict(report) for report in self.node_reports],
        }

    def summary(self) -> str:
        """Render a concise human-readable workflow validation result."""
        label = self.source or self.name or "<mapping>"
        if self.is_valid:
            checked = "imports checked" if self.imports_checked else "imports not checked"
            return f"Valid workflow: {label} ({len(self.node_reports)} node(s), {checked})."
        lines = [f"Invalid workflow: {label}"]
        lines.extend(f"  - {error}" for error in self.errors)
        lines.extend(f"  warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)
