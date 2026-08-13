"""Durable boundary between successful training and post-run actions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
from lambdaforge.experiments.results.RunFingerprint import RunFingerprint
from lambdaforge.experiments.RunResult import RunResult
from lambdaforge.experiments.RunStatus import RunStatus


class TrainingCompletionStore:
    """Persist reusable training success before potentially long post-processing."""

    RELATIVE_PATH = Path(".lambdaforge/post-run/training-result.json")

    @classmethod
    def stage_identity(cls, config: Mapping[str, Any]) -> str:
        """Identify the completed training stage while excluding post-run configuration."""
        adaptive = ExperimentConfig.get_value(config, "metadata.adaptive")
        payload: dict[str, Any] = {"training_fingerprint": RunFingerprint.digest(config)}
        if isinstance(adaptive, Mapping):
            payload["adaptive_target_budget"] = adaptive.get("target_budget")
            payload["trainer_max_epochs"] = ExperimentConfig.get_value(config, "trainer.max_epochs")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def write(self, config: Mapping[str, Any], result: RunResult) -> Path:
        """Commit a rank-zero successful training result atomically."""
        if result.status is not RunStatus.OK:
            raise ValueError("Only successful training can be committed for post-run reuse.")
        path = Path(result.run_dir) / self.RELATIVE_PATH
        return result.with_updates(
            training_stage_identity=self.stage_identity(config),
            post_run_fingerprint=None,
            post_run_actions=[],
            artifacts=[],
        ).write_json(path)

    def load(self, config: Mapping[str, Any], run_dir: str | Path) -> RunResult | None:
        """Return matching reusable training evidence, never a stale checkpoint stage."""
        path = Path(run_dir) / self.RELATIVE_PATH
        try:
            result = RunResult.read_json(path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if result.status is not RunStatus.OK:
            return None
        if result.get("training_stage_identity") != self.stage_identity(config):
            return None
        if result.config_fingerprint != RunFingerprint.digest(config):
            return None
        checkpoint_policy = str(
            ExperimentConfig.get_value(config, "trainer.checkpoint_policy", "last_and_best")
        )
        if checkpoint_policy != "none":
            from lambdaforge.experiments.retention.CheckpointResolver import CheckpointResolver

            if not CheckpointResolver(run_dir).candidates():
                return None
        return result
