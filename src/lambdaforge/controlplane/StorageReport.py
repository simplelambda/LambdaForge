"""Immutable cluster storage observation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class StorageReport:
    """Report bounded storage categories for one reachable or offline cluster."""

    cluster: str
    online: bool
    categories: dict[str, dict[str, Any]]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster": self.cluster,
            "online": self.online,
            "categories": self.categories,
            "error": self.error,
        }
