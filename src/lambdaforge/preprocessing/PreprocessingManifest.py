"""Crash-recoverable per-record preprocessing progress manifest."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping
from lambdaforge.experiments.JsonResult import JsonResult


class PreprocessingManifest(JsonResult):
    """Persist deterministic shard progress for safe resume after interruption."""

    def __init__(
        self,
        *,
        config_fingerprint: str,
        shard_count: int,
        shard_index: int,
        started_at_utc: str,
        updated_at_utc: str,
        complete: bool,
        records: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.config_fingerprint = str(config_fingerprint)
        self.shard_count = int(shard_count)
        self.shard_index = int(shard_index)
        self.started_at_utc = str(started_at_utc)
        self.updated_at_utc = str(updated_at_utc)
        self.complete = bool(complete)
        self.records = FrozenJsonMapping(records or {})
        self._freeze_mapping(self.to_dict())

    @property
    def successful_keys(self) -> frozenset[str]:
        """Return keys whose latest persisted record status is successful."""
        return frozenset(key for key, value in self.records.items() if value.get("status") == "ok")

    @property
    def failed_keys(self) -> frozenset[str]:
        """Return keys whose latest persisted record status is failed."""
        return frozenset(
            key for key, value in self.records.items() if value.get("status") == "failed"
        )

    @classmethod
    def read_json(cls, path: str | Path) -> PreprocessingManifest:
        """Read and validate a previously persisted preprocessing manifest."""
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, Mapping):
            raise TypeError("Preprocessing manifest JSON must contain an object.")
        records = value.get("records", {})
        if not isinstance(records, Mapping):
            raise TypeError("Preprocessing manifest records must be a mapping.")
        return cls(
            config_fingerprint=str(value["config_fingerprint"]),
            shard_count=int(value["shard_count"]),
            shard_index=int(value["shard_index"]),
            started_at_utc=str(value["started_at_utc"]),
            updated_at_utc=str(value["updated_at_utc"]),
            complete=bool(value.get("complete", False)),
            records=records,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned JSON-compatible progress envelope."""
        return {
            "manifest_version": 1,
            "config_fingerprint": self.config_fingerprint,
            "shard_count": self.shard_count,
            "shard_index": self.shard_index,
            "started_at_utc": self.started_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "complete": self.complete,
            "counts": {
                "ok": len(self.successful_keys),
                "failed": len(self.failed_keys),
                "total": len(self.records),
            },
            "records": copy.deepcopy(self.records),
        }
