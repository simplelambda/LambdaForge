"""Typed terminal result for one materialized experiment run."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lambdaforge.experiments.JsonResult import JsonResult
from lambdaforge.experiments.RunStatus import RunStatus


class RunResult(JsonResult):
    """Represent a run outcome while preserving legacy mapping/JSON behavior."""

    _KNOWN_FIELDS = frozenset(
        {
            "name",
            "result_version",
            "run_dir",
            "variant",
            "seed",
            "status",
            "seconds",
            "best_model_path",
            "last_model_path",
            "best_metric",
            "best_epoch_metrics",
            "final_metrics",
            "best_metrics",
            "error",
            "exit_code",
            "skipped_existing",
            "attempt_id",
            "config_fingerprint",
            "started_at_utc",
            "finished_at_utc",
            "training_stage_identity",
            "post_run_fingerprint",
            "post_run_actions",
            "post_run_warnings",
            "artifacts",
        }
    )

    def __init__(
        self,
        *,
        result_version: int = 1,
        name: str,
        run_dir: str | Path,
        variant: str | None = None,
        seed: Any = None,
        status: RunStatus | str = RunStatus.UNKNOWN,
        seconds: float | None = None,
        best_model_path: str | Path | None = None,
        last_model_path: str | Path | None = None,
        best_metric: Mapping[str, Any] | None = None,
        best_epoch_metrics: Mapping[str, Any] | None = None,
        final_metrics: Mapping[str, Any] | None = None,
        best_metrics: Mapping[str, Any] | None = None,
        error: str | None = None,
        exit_code: int | None = None,
        skipped_existing: bool | None = None,
        attempt_id: str | None = None,
        config_fingerprint: str | None = None,
        started_at_utc: str | None = None,
        finished_at_utc: str | None = None,
        training_stage_identity: str | None = None,
        post_run_fingerprint: str | None = None,
        post_run_actions: Sequence[Mapping[str, Any]] = (),
        post_run_warnings: Sequence[str] = (),
        artifacts: Sequence[Mapping[str, Any]] = (),
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        if result_version < 1:
            raise ValueError("result_version must be at least 1.")
        self.result_version = int(result_version)
        self.name = str(name)
        self.run_dir = str(run_dir)
        self.variant = str(variant) if variant is not None else None
        self.seed = copy.deepcopy(seed)
        raw_status = status.value if isinstance(status, RunStatus) else str(status)
        try:
            self.status = RunStatus(raw_status)
        except ValueError:
            self.status = RunStatus.UNKNOWN
        self._serialized_status = raw_status
        self.seconds = float(seconds) if seconds is not None else None
        self.best_model_path = str(best_model_path) if best_model_path else None
        self.last_model_path = str(last_model_path) if last_model_path else None
        self.best_metric = self._optional_mapping(best_metric)
        self.best_epoch_metrics = self._optional_mapping(best_epoch_metrics)
        self.final_metrics = self._optional_mapping(final_metrics)
        self.best_metrics = self._optional_mapping(best_metrics)
        self.error = str(error) if error is not None else None
        self.exit_code = int(exit_code) if exit_code is not None else None
        self.skipped_existing = bool(skipped_existing) if skipped_existing is not None else None
        self.attempt_id = str(attempt_id) if attempt_id is not None else None
        self.config_fingerprint = (
            str(config_fingerprint) if config_fingerprint is not None else None
        )
        self.started_at_utc = str(started_at_utc) if started_at_utc is not None else None
        self.finished_at_utc = str(finished_at_utc) if finished_at_utc is not None else None
        self.training_stage_identity = (
            str(training_stage_identity) if training_stage_identity is not None else None
        )
        self.post_run_fingerprint = (
            str(post_run_fingerprint) if post_run_fingerprint is not None else None
        )
        from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping

        self.post_run_actions = tuple(FrozenJsonMapping(value) for value in post_run_actions)
        self.post_run_warnings = tuple(str(value) for value in post_run_warnings)
        self.artifacts = tuple(FrozenJsonMapping(value) for value in artifacts)
        self._extra = copy.deepcopy(dict(extra or {}))
        self._freeze_mapping(self.to_dict())

    @property
    def is_terminal(self) -> bool:
        """Return whether the result should not be resumed automatically."""
        return self.status in {RunStatus.OK, RunStatus.FAILED}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RunResult:
        """Parse persisted or legacy result mappings without losing extra fields."""
        payload = dict(value)
        return cls(
            result_version=int(payload.get("result_version", 1)),
            name=str(payload.get("name", "")),
            run_dir=str(payload.get("run_dir", "")),
            variant=payload.get("variant"),
            seed=payload.get("seed"),
            status=str(payload.get("status", RunStatus.UNKNOWN.value)),
            seconds=payload.get("seconds"),
            best_model_path=payload.get("best_model_path"),
            last_model_path=payload.get("last_model_path"),
            best_metric=cls._mapping_or_none(payload.get("best_metric")),
            best_epoch_metrics=cls._mapping_or_none(payload.get("best_epoch_metrics")),
            final_metrics=cls._mapping_or_none(payload.get("final_metrics")),
            best_metrics=cls._mapping_or_none(payload.get("best_metrics")),
            error=payload.get("error"),
            exit_code=payload.get("exit_code"),
            skipped_existing=payload.get("skipped_existing"),
            attempt_id=payload.get("attempt_id"),
            config_fingerprint=payload.get("config_fingerprint"),
            started_at_utc=payload.get("started_at_utc"),
            finished_at_utc=payload.get("finished_at_utc"),
            training_stage_identity=payload.get("training_stage_identity"),
            post_run_fingerprint=payload.get("post_run_fingerprint"),
            post_run_actions=cls._mapping_sequence(
                payload.get("post_run_actions"), "post_run_actions"
            ),
            post_run_warnings=cls._string_sequence(
                payload.get("post_run_warnings"), "post_run_warnings"
            ),
            artifacts=cls._mapping_sequence(payload.get("artifacts"), "artifacts"),
            extra={key: item for key, item in payload.items() if key not in cls._KNOWN_FIELDS},
        )

    @classmethod
    def read_json(cls, path: str | Path) -> RunResult:
        """Load one result file and validate its top-level mapping."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TypeError(f"Run result must contain a JSON object: {path}")
        return cls.from_mapping(payload)

    def with_updates(self, **changes: Any) -> RunResult:
        """Return a new typed result with selected serialized fields replaced."""
        payload = self.to_dict()
        payload.update(changes)
        return self.from_mapping(payload)

    def to_dict(self) -> dict[str, Any]:
        """Return a defensive JSON-compatible mapping."""
        payload = copy.deepcopy(self._extra)
        payload.update(
            {
                "result_version": self.result_version,
                "name": self.name,
                "run_dir": self.run_dir,
                "variant": self.variant,
                "seed": copy.deepcopy(self.seed),
                "status": self._serialized_status,
                "best_model_path": self.best_model_path,
                "last_model_path": self.last_model_path,
            }
        )
        optional = {
            "seconds": self.seconds,
            "best_metric": self.best_metric,
            "best_epoch_metrics": self.best_epoch_metrics,
            "final_metrics": self.final_metrics,
            "best_metrics": self.best_metrics,
            "error": self.error,
            "exit_code": self.exit_code,
            "skipped_existing": self.skipped_existing,
            "attempt_id": self.attempt_id,
            "config_fingerprint": self.config_fingerprint,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "training_stage_identity": self.training_stage_identity,
            "post_run_fingerprint": self.post_run_fingerprint,
        }
        for key, value in optional.items():
            if value is not None:
                payload[key] = copy.deepcopy(value)
        if self.post_run_actions:
            payload["post_run_actions"] = [copy.deepcopy(value) for value in self.post_run_actions]
        if self.post_run_warnings:
            payload["post_run_warnings"] = list(self.post_run_warnings)
        if self.artifacts:
            payload["artifacts"] = [copy.deepcopy(value) for value in self.artifacts]
        return payload

    @staticmethod
    def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise TypeError("Run metric fields must be mappings when provided.")
        return value

    @staticmethod
    def _optional_mapping(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
        return copy.deepcopy(dict(value)) if value is not None else None

    @staticmethod
    def _mapping_sequence(value: Any, label: str) -> tuple[Mapping[str, Any], ...]:
        if value is None:
            return ()
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise TypeError(f"Run result {label} must be a sequence.")
        if any(not isinstance(item, Mapping) for item in value):
            raise TypeError(f"Run result {label} entries must be mappings.")
        return tuple(value)

    @staticmethod
    def _string_sequence(value: Any, label: str) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise TypeError(f"Run result {label} must be a sequence.")
        if any(not isinstance(item, str) for item in value):
            raise TypeError(f"Run result {label} entries must be strings.")
        return tuple(value)
