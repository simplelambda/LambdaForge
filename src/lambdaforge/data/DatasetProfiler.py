"""Project-extensible dataset statistics contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lambdaforge.data.DatasetRecord import DatasetRecord


class DatasetProfiler(ABC):
    """Add domain statistics only when a dataset explicitly declares their schema."""

    @abstractmethod
    def profile(
        self, root: Path, record: DatasetRecord, schema: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Return JSON-compatible statistics without changing dataset bytes."""
