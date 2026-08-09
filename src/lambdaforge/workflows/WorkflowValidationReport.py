"""Immutable validation result for a complete workflow DAG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class WorkflowValidationReport:
    """Collect graph and topologically ordered node facts without executing the DAG."""

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
