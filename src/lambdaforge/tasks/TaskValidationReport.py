"""Immutable validation report for a generic task configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TaskValidationReport:
    """Collect task Schema, import and contract errors without side effects."""

    source: str | None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    imports_checked: bool = True
    schema_version: str | None = None

    @property
    def is_valid(self) -> bool:
        """Return whether no validation error was found."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Return a stable machine-readable validation envelope."""
        return {
            "source": self.source,
            "kind": "task",
            "valid": self.is_valid,
            "imports_checked": self.imports_checked,
            "schema_version": self.schema_version,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }

    def summary(self) -> str:
        """Render a concise human-readable task validation result."""
        label = self.source or "<mapping>"
        if self.is_valid:
            checked = "imports checked" if self.imports_checked else "imports not checked"
            return f"Valid task: {label} (Schema {self.schema_version}, {checked})."
        lines = [f"Invalid task: {label}"]
        lines.extend(f"  - {error}" for error in self.errors)
        lines.extend(f"  warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)
