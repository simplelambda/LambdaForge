"""Artifact validation extension boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from lambdaforge.artifacts.ArtifactValidationResult import ArtifactValidationResult


class ArtifactValidator(ABC):
    """Validate generic artifact invariants and return structured evidence."""

    @abstractmethod
    def validate(self, path: Path) -> ArtifactValidationResult:
        """Validate without mutating or trusting arbitrary serialized code."""
