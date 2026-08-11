"""Project-specific artifact semantics boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ArtifactSchema(ABC):
    """Map domain conventions to explicit generic artifact roles."""

    @abstractmethod
    def describe(self, path: Path) -> Mapping[str, Any]:
        """Return roles such as positions/edges/features without loading unsafe objects."""
