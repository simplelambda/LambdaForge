"""Validation report for a dataset recipe and its stage tasks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatasetRecipeValidationReport:
    """Return all safe static recipe validation findings together."""

    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        """Return whether the recipe can be planned."""
        return not self.errors

    def summary(self) -> str:
        """Render a concise human validation result."""
        if self.is_valid:
            return "Dataset recipe validation: OK"
        return "Dataset recipe validation failed:\n- " + "\n- ".join(self.errors)
