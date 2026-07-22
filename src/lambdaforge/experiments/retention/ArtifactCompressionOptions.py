"""Validated compression options for one artifact-retention rule."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ArtifactCompressionOptions:
    """Configure ZIP compression while preserving incompressible originals."""

    level: int | None = None
    only_if_smaller: bool = True

    def __post_init__(self) -> None:
        """Reject invalid direct construction as strictly as mapping parsing."""
        if isinstance(self.level, bool) or (
            self.level is not None and not isinstance(self.level, int)
        ):
            raise TypeError("retention rule compression.level must be an integer.")
        if self.level is not None and not 0 <= self.level <= 9:
            raise ValueError("retention rule compression.level must be between 0 and 9.")
        if not isinstance(self.only_if_smaller, bool):
            raise TypeError("retention rule compression.only_if_smaller must be a bool.")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
    ) -> ArtifactCompressionOptions:
        """Parse one optional compression mapping without coercing values."""
        if value is None:
            value = {}
        if not isinstance(value, Mapping):
            raise TypeError("retention rule compression must be a mapping.")
        unknown = sorted(set(value) - {"level", "only_if_smaller"})
        if unknown:
            raise ValueError(f"Unknown retention rule compression options: {unknown}.")
        return cls(
            level=value.get("level"),
            only_if_smaller=value.get("only_if_smaller", True),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        payload: dict[str, Any] = {"only_if_smaller": self.only_if_smaller}
        if self.level is not None:
            payload["level"] = self.level
        return payload
