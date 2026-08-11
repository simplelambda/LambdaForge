"""Bounded preprocessing debug result."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PreprocessingDebugResult:
    """Describe sampled source/transform behaviour without publishing a dataset."""

    config_fingerprint: str
    sample_identity: str
    requested_records: int
    records: tuple[Mapping[str, Any], ...]

    @property
    def ok(self) -> bool:
        """Return whether every sampled record completed its transform chain."""
        return all(not record.get("exception") for record in self.records)

    def to_dict(self) -> dict[str, Any]:
        """Return a machine-readable debug report."""
        return {
            "debug_result_version": 1,
            "config_fingerprint": self.config_fingerprint,
            "sample_identity": self.sample_identity,
            "requested_records": self.requested_records,
            "processed_records": len(self.records),
            "ok": self.ok,
            "records": [copy.deepcopy(dict(record)) for record in self.records],
        }
