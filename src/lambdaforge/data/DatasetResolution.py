"""Exact result of resolving one logical dataset reference."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from lambdaforge.data.DatasetLocation import DatasetLocation
from lambdaforge.data.DatasetRecord import DatasetRecord
from lambdaforge.data.DatasetReference import DatasetReference


@dataclass(frozen=True, slots=True)
class DatasetResolution:
    """Pin a logical version/content identity while recording the placement used."""

    reference: DatasetReference
    exact_reference: str
    location: DatasetLocation
    identity: Mapping[str, Any]
    descriptor: Mapping[str, Any] = field(default_factory=dict)
    record: DatasetRecord | None = None
    managed: bool = False

    def to_binding(self, *, path: str) -> dict[str, Any]:
        """Return reproducibility evidence with physical placement outside identity."""
        return {
            "path": path,
            "reference": str(self.reference),
            "exact_reference": self.exact_reference,
            "identity": copy.deepcopy(dict(self.identity)),
            "resolved_path": self.location.uri,
            "environment": self.location.environment,
            "managed": self.managed,
        }
