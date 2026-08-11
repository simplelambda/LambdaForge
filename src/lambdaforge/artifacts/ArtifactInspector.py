"""Safe artifact-inspection extension boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from lambdaforge.artifacts.ArtifactInspection import ArtifactInspection


class ArtifactInspector(ABC):
    """Convert artifact bytes into bounded structured metadata."""

    @abstractmethod
    def supports(self, path: Path, *, media_type: str | None = None) -> bool:
        """Return whether this inspector owns the explicit format."""

    @abstractmethod
    def inspect(
        self,
        path: Path,
        *,
        item: str | None = None,
        rows: int = 20,
        slice_expression: str | None = None,
    ) -> ArtifactInspection:
        """Inspect safely without unbounded output or unsafe deserialization."""
