"""Stable-key record passed through a preprocessing pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping


@dataclass(frozen=True, slots=True)
class PreprocessingRecord:
    """Pair an arbitrary value with deterministic identity and JSON metadata."""

    key: str
    value: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("Preprocessing record keys must be non-empty strings.")
        object.__setattr__(self, "metadata", FrozenJsonMapping(self.metadata))

    def with_value(self, value: Any) -> PreprocessingRecord:
        """Return a new record preserving identity and metadata."""
        return PreprocessingRecord(key=self.key, value=value, metadata=self.metadata)
