"""Typed result of validating one experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Collect validation errors, warnings and non-mutating inspection facts."""

    source: str | None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    imports_checked: bool = True
    expanded_runs: int | None = None
    source_schema_version: str | None = None
    target_schema_version: str | None = None
    migration_steps: tuple[dict[str, Any], ...] = ()

    @property
    def is_valid(self) -> bool:
        """Return whether no validation error was found."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "source": self.source,
            "valid": self.is_valid,
            "imports_checked": self.imports_checked,
            "expanded_runs": self.expanded_runs,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "migration": {
                "source_version": self.source_schema_version,
                "target_version": self.target_schema_version,
                "changed": bool(self.migration_steps),
                "steps": [dict(step) for step in self.migration_steps],
            },
        }

    def summary(self) -> str:
        """Render a concise human-readable CLI summary."""
        label = self.source or "<mapping>"
        if self.is_valid:
            import_note = "imports checked" if self.imports_checked else "imports not checked"
            migration_note = (
                f", migrated to Schema {self.target_schema_version}" if self.migration_steps else ""
            )
            return (
                f"Valid experiment: {label} "
                f"({self.expanded_runs} run(s), {import_note}{migration_note})."
            )
        lines = [f"Invalid experiment: {label}"]
        lines.extend(f"  - {error}" for error in self.errors)
        lines.extend(f"  warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)
