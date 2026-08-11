"""Artifact visualization extension boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lambdaforge.visualization.PlotSpec import PlotSpec


class ArtifactVisualizer(ABC):
    """Build a declarative view from explicitly assigned artifact semantics."""

    @abstractmethod
    def specification(
        self, path: Path, *, visualization_type: str, roles: Mapping[str, Any]
    ) -> PlotSpec:
        """Return data required for a renderer without writing a figure."""
