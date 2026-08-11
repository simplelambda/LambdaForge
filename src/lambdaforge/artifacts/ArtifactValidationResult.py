"""Structured artifact validation result."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ArtifactValidationResult:
    """Report validator success and every actionable issue."""

    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return machine-readable validation evidence."""
        return {"valid": self.valid, "errors": list(self.errors), "warnings": list(self.warnings)}
