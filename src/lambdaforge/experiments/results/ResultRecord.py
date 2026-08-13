"""Immutable catalog entry for a canonical or archived run attempt."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lambdaforge.experiments.JsonResult import JsonResult
from lambdaforge.experiments.RunResult import RunResult


class ResultRecord(JsonResult):
    """Describe one persisted attempt without hiding its source artifact."""

    def __init__(
        self,
        *,
        result: RunResult,
        result_path: str | Path,
        run_dir: str | Path,
        archived: bool,
        config_fingerprint: str | None,
        attempt_id: str,
    ) -> None:
        self.result = result
        self.result_path = str(result_path)
        self.run_dir = str(run_dir)
        self.archived = bool(archived)
        self.config_fingerprint = config_fingerprint
        self.attempt_id = str(attempt_id)
        self._freeze_mapping(self.to_dict())

    @property
    def status(self) -> str:
        """Return the serialized terminal state of the attempt."""
        return str(self.result.get("status", "unknown"))

    @property
    def metrics(self) -> Mapping[str, Any]:
        """Return the preferred coherent metrics available for this attempt."""
        best_epoch = self.result.get("best_epoch_metrics")
        if isinstance(best_epoch, Mapping) and isinstance(best_epoch.get("metrics"), Mapping):
            return copy.deepcopy(dict(best_epoch["metrics"]))
        final = self.result.get("final_metrics")
        return copy.deepcopy(dict(final)) if isinstance(final, Mapping) else {}

    def to_dict(self) -> dict[str, Any]:
        """Return a defensive machine-readable catalog entry."""
        return {
            "attempt_id": self.attempt_id,
            "archived": self.archived,
            "config_fingerprint": self.config_fingerprint,
            "result_path": self.result_path,
            "run_dir": self.run_dir,
            "name": self.result.name,
            "variant": self.result.variant,
            "seed": copy.deepcopy(self.result.seed),
            "status": self.status,
            "started_at_utc": self.result.started_at_utc,
            "finished_at_utc": self.result.finished_at_utc,
            "metrics": dict(self.metrics),
            "artifacts": [copy.deepcopy(value) for value in self.result.artifacts],
            "post_run_actions": [copy.deepcopy(value) for value in self.result.post_run_actions],
        }
