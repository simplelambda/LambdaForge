"""Artifact-store provider boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from lambdaforge.storage.ArtifactReference import ArtifactReference


class ArtifactStore(ABC):
    """Publish immutable content and stage verified local copies."""

    @abstractmethod
    def publish(
        self, source: str | Path, *, key: str | None = None, media_type: str | None = None
    ) -> ArtifactReference:
        """Atomically publish content and return its logical reference."""

    @abstractmethod
    def stage(self, reference: ArtifactReference, destination: str | Path) -> Path:
        """Materialize and verify content at a local destination."""

    @abstractmethod
    def exists(self, reference: ArtifactReference) -> bool:
        """Return whether verified referenced content exists."""
