"""Implementation of the ExperimentRunner object."""

from __future__ import annotations

import copy
import csv
import json
import random
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import yaml

from lambdaforge.EnvironmentManifest import EnvironmentManifest
from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
from lambdaforge.experiments.ObjectFactory import ObjectFactory
from lambdaforge.experiments.results.RunFingerprint import RunFingerprint
from lambdaforge.experiments.RunResult import RunResult
from lambdaforge.experiments.RunStatus import RunStatus
from lambdaforge.plugins.PluginRegistry import PluginRegistry


class ExperimentRunner:
    """Build, train, resume and summarize task-agnostic YAML experiments.

    The runner owns one materialized run at a time and delegates process
    scheduling and aggregate generation to dedicated collaborators.
    """

    def _as_list(self, value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return list(value)
        return [value]

    def _set_seed(self, seed: int | None) -> None:
        if seed is None:
            return

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def seed_segment(self, config: Mapping[str, Any]) -> str:
        """Directory segment for the seed level.

        Seeds always get their own subfolder (even a single one), so that all
        seeds of one parameter combination share a parent for aggregation. A run
        without any seed uses ``seed=none``.
        """
        seed = ExperimentConfig.get_value(config, "experiment.seed")
        return f"seed={seed}" if seed is not None else "seed=none"

    def experiment_run_dir(self, config: Mapping[str, Any]) -> Path:
        r"""Resolve the run directory ``<root>/<base>/<variant>/seed=<seed>/``.

        ``variant`` is the grid+ablation slug (the parameter combination) written
        by :meth:`ExperimentConfig.expand`; the seed is
        nested one level deeper. Public so the execution layer can locate
        ``train.log`` / ``result.json`` for a materialized config.
        """
        variant = str(ExperimentConfig.get_value(config, "experiment.variant", "base"))
        return ExperimentConfig.suite_dir_for(config) / variant / self.seed_segment(config)

    def _write_materialized_config(self, config: Mapping[str, Any], run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(dict(config), f, sort_keys=False)

    def _write_environment_manifest(
        self,
        run_dir: Path,
        manifest: EnvironmentManifest | None = None,
    ) -> EnvironmentManifest:
        """Atomically persist one environment snapshot and return it."""
        captured = manifest or EnvironmentManifest.capture()
        captured.write(run_dir / "environment.json")
        return captured

    def _build_datamodule(
        self,
        config: Mapping[str, Any],
        plugins: PluginRegistry,
    ):
        from lambdaforge.training.data.LightningDataModule import LightningDataModule

        data_cfg = config.get("data", {})
        if not isinstance(data_cfg, Mapping):
            raise TypeError("'data' must be a mapping.")

        train = (
            ObjectFactory.build(data_cfg.get("train"), plugins=plugins)
            if "train" in data_cfg
            else None
        )
        val = (
            ObjectFactory.build(data_cfg.get("val"), plugins=plugins) if "val" in data_cfg else None
        )
        test = (
            ObjectFactory.build(data_cfg.get("test"), plugins=plugins)
            if "test" in data_cfg
            else None
        )

        datamodule_cfg = data_cfg.get("datamodule", {})
        if datamodule_cfg is None:
            datamodule_cfg = {}
        if not isinstance(datamodule_cfg, Mapping):
            raise TypeError("data.datamodule must be a mapping.")

        target = datamodule_cfg.get("target")
        params = ObjectFactory.build(datamodule_cfg.get("params", {}), plugins=plugins)
        if not isinstance(params, Mapping):
            raise TypeError("data.datamodule.params must be a mapping.")

        params = dict(params)
        params.setdefault("train", train)
        params.setdefault("val", val)
        params.setdefault("test", test)

        cls = ObjectFactory.import_object(str(target)) if target else LightningDataModule
        return cls(**params)

    def _build_task(
        self,
        config: Mapping[str, Any],
        model: torch.nn.Module,
        plugins: PluginRegistry,
    ):
        from lambdaforge.training.LightningTask import LightningTask

        losses = self._as_list(ObjectFactory.build(config.get("losses"), plugins=plugins))
        if not losses:
            raise RuntimeError("Experiment config must define at least one loss.")

        shared_metrics = self._as_list(
            ObjectFactory.build(config.get("metrics", []), plugins=plugins)
        )
        train_metrics_spec = config.get("train_metrics")
        val_metrics_spec = config.get("val_metrics")
        test_metrics_spec = config.get("test_metrics")
        train_metrics = (
            self._as_list(ObjectFactory.build(train_metrics_spec, plugins=plugins))
            if train_metrics_spec is not None
            else shared_metrics
        )
        val_metrics = (
            self._as_list(ObjectFactory.build(val_metrics_spec, plugins=plugins))
            if val_metrics_spec is not None
            else None
        )
        test_metrics = (
            self._as_list(ObjectFactory.build(test_metrics_spec, plugins=plugins))
            if test_metrics_spec is not None
            else None
        )

        optimizer_cfg = config.get("optimizer", {})
        if optimizer_cfg is None:
            optimizer_cfg = {}
        if not isinstance(optimizer_cfg, Mapping):
            raise TypeError("'optimizer' must be a mapping.")

        optimizer_ref = optimizer_cfg.get("ref", "torch.optim.AdamW")
        optimizer_cls = ObjectFactory.import_object(str(optimizer_ref))
        optimizer_kwargs = ObjectFactory.build(
            optimizer_cfg.get("params", {}),
            plugins=plugins,
        )

        scheduler_cfg = config.get("scheduler")
        scheduler_cls = None
        scheduler_kwargs = {}
        scheduler_config = None
        if scheduler_cfg is not None:
            if not isinstance(scheduler_cfg, Mapping):
                raise TypeError("'scheduler' must be a mapping.")
            scheduler_cls = ObjectFactory.import_object(str(scheduler_cfg["ref"]))
            scheduler_kwargs = ObjectFactory.build(
                scheduler_cfg.get("params", {}),
                plugins=plugins,
            )
            scheduler_config = ObjectFactory.build(
                scheduler_cfg.get("lightning", None),
                plugins=plugins,
            )

        task_cfg = config.get("task", {})
        if task_cfg is None:
            task_cfg = {}
        if not isinstance(task_cfg, Mapping):
            raise TypeError("'task' must be a mapping.")

        target = task_cfg.get("target")
        params = ObjectFactory.build(task_cfg.get("params", {}), plugins=plugins)
        if not isinstance(params, Mapping):
            raise TypeError("task.params must be a mapping.")

        params = dict(params)
        params.setdefault("model", model)
        params.setdefault("losses", losses)
        params.setdefault("metrics", train_metrics)
        params.setdefault("val_metrics", val_metrics)
        params.setdefault("test_metrics", test_metrics)
        params.setdefault("optimizer_cls", optimizer_cls)
        params.setdefault("optimizer_kwargs", optimizer_kwargs)
        params.setdefault("scheduler_cls", scheduler_cls)
        params.setdefault("scheduler_kwargs", scheduler_kwargs)
        params.setdefault("scheduler_config", scheduler_config)

        cls = ObjectFactory.import_object(str(target)) if target else LightningTask
        task = cls(**params)
        selection_metrics = self._as_list(
            getattr(task, "val_metrics", val_metrics or shared_metrics or train_metrics)
        )
        return task, selection_metrics

    def _build_runner(
        self,
        config: Mapping[str, Any],
        metrics: list[Any],
        plugins: PluginRegistry,
    ):
        from lambdaforge.training.LightningRunner import LightningRunner
        from lambdaforge.training.LightningTrainConfig import LightningTrainConfig

        trainer_cfg = dict(config.get("trainer", {}) or {})
        run_dir = self.experiment_run_dir(config)
        trainer_cfg.setdefault("default_root_dir", str(run_dir))

        train_config = LightningTrainConfig(**ObjectFactory.build(trainer_cfg, plugins=plugins))
        early_metric = (
            metrics[0] if metrics and train_config.early_stopping_patience is not None else None
        )
        checkpoint_metric = metrics[0] if metrics else None

        runner_cfg = config.get("runner", {})
        if runner_cfg is None:
            runner_cfg = {}
        if not isinstance(runner_cfg, Mapping):
            raise TypeError("'runner' must be a mapping.")

        target = runner_cfg.get("target")
        params = ObjectFactory.build(runner_cfg.get("params", {}), plugins=plugins)
        if not isinstance(params, Mapping):
            raise TypeError("runner.params must be a mapping.")

        params = dict(params)
        if "callbacks" in config:
            params.setdefault(
                "callbacks",
                self._as_list(ObjectFactory.build(config["callbacks"], plugins=plugins)),
            )
        params.setdefault("config", train_config)
        params.setdefault("early_stopping_metric", early_metric)
        params.setdefault("checkpoint_metric", checkpoint_metric)

        cls = ObjectFactory.import_object(str(target)) if target else LightningRunner
        return cls(**params)

    def _write_hparams(self, config: Mapping[str, Any], run_dir: Path) -> None:
        """Write a flat ``hparams.json`` summary for quick inspection/reload.

        ``config.yaml`` remains the source of truth used by
        :mod:`lambdaforge.experiments.loading`; this file is a convenience flattening of
        the pieces that identify the trained artifact.
        """
        hparams = {
            "name": ExperimentConfig.get_value(config, "experiment.name"),
            "base_name": ExperimentConfig.get_value(config, "experiment.base_name"),
            "variant": ExperimentConfig.get_value(config, "experiment.variant"),
            "seed": ExperimentConfig.get_value(config, "experiment.seed"),
            "model": config.get("model"),
            "data": {
                key: ExperimentConfig.get_value(config, f"data.{key}.params")
                for key in ("train", "val", "test")
                if ExperimentConfig.get_value(config, f"data.{key}") is not None
            },
            "optimizer": config.get("optimizer"),
            "trainer": config.get("trainer"),
        }
        with open(run_dir / "hparams.json", "w", encoding="utf-8") as f:
            json.dump(hparams, f, indent=2, default=str)

    def _best_metric(self, trainer: Any) -> dict[str, Any]:
        """Extract the monitored metric and its best value from the checkpoint."""
        callback = getattr(trainer, "checkpoint_callback", None)
        if callback is None:
            return {}

        best_score = getattr(callback, "best_model_score", None)
        return {
            "monitor": getattr(callback, "monitor", None),
            "best_score": float(best_score) if best_score is not None else None,
        }

    def _scalar_metrics(self, raw_metrics: Mapping[str, Any]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for key, value in raw_metrics.items():
            if str(key) in {"epoch", "step", "hp_metric"} or str(key).endswith("_step"):
                continue
            if torch.is_tensor(value):
                if value.numel() != 1:
                    continue
                metrics[str(key)] = float(value.detach().cpu().item())
            elif isinstance(value, (int, float)):
                metrics[str(key)] = float(value)
        return metrics

    def _metric_directions(self, task: Any) -> dict[str, str]:
        directions: dict[str, str] = {}
        for split, attr in (
            ("train", "train_metrics"),
            ("val", "val_metrics"),
            ("test", "test_metrics"),
        ):
            for metric in getattr(task, attr, []) or []:
                directions[f"{split}_{metric.name}"] = "max" if metric.higher_is_better else "min"

        for split in ("train", "val", "test"):
            directions[f"{split}_loss"] = "min"

        directions["epoch_time_s"] = "min"
        directions["gpu_mem_mb"] = "min"
        directions["cpu_rss_mb"] = "min"
        return directions

    def _best_metrics_from_csv(
        self, run_dir: Path, directions: Mapping[str, str]
    ) -> dict[str, dict[str, float | str]]:
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            return {}

        best: dict[str, dict[str, float | str]] = {}
        with open(metrics_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                epoch = self._maybe_float(row.get("epoch"))
                for name, mode in directions.items():
                    value = self._maybe_float(row.get(name))
                    if value is None:
                        continue

                    current = best.get(name)
                    is_better = (
                        current is None
                        or (mode == "max" and value > float(current["value"]))
                        or (mode == "min" and value < float(current["value"]))
                    )
                    if is_better:
                        best[name] = {
                            "value": value,
                            "epoch": epoch if epoch is not None else -1,
                            "mode": mode,
                        }
        return best

    def _maybe_float(self, value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _best_epoch_metrics(
        self,
        run_dir: Path,
        monitor: str | None,
        mode: str,
    ) -> dict[str, Any]:
        """Return every metric captured at the epoch that optimized ``monitor``.

        Unlike ``_best_metrics_from_csv`` (which takes each metric's best value
        independently, possibly from different epochs), this returns a *coherent*
        snapshot: it locates the single epoch where the monitored/checkpointed
        metric was best and reports all metrics from that same row. This is the
        "best epoch" a checkpoint corresponds to, so downstream summaries do not
        have to rely on the last epoch, which may already be overfit under a
        patient early stopping.
        """
        metrics_path = run_dir / "metrics.csv"
        if monitor is None or not metrics_path.exists():
            return {}

        best_row: dict[str, str] | None = None
        best_score: float | None = None
        with open(metrics_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                value = self._maybe_float(row.get(monitor))
                if value is None:
                    continue
                is_better = (
                    best_score is None
                    or (mode == "max" and value > best_score)
                    or (mode == "min" and value < best_score)
                )
                if is_better:
                    best_score = value
                    best_row = row

        if best_row is None:
            return {}

        metrics = {
            key: self._maybe_float(cell)
            for key, cell in best_row.items()
            if key != "epoch" and self._maybe_float(cell) is not None
        }
        epoch = self._maybe_float(best_row.get("epoch"))
        return {
            "monitor": monitor,
            "mode": mode,
            "epoch": int(epoch) if epoch is not None else -1,
            "best_score": best_score,
            "metrics": metrics,
        }

    def _read_result(self, path: Path) -> RunResult | None:
        if not path.exists():
            return None
        try:
            return RunResult.read_json(path)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None

    def _archive_previous_result(self, run_dir: Path) -> Path | None:
        """Atomically retire a prior result before a new attempt mutates the run."""
        result_path = run_dir / "result.json"
        if not result_path.exists():
            return None

        attempts_dir = self._safe_attempts_dir(run_dir)
        previous = self._read_result(result_path)
        attempt_id = previous.attempt_id if previous is not None else None
        destination = attempts_dir / (
            f"result-{attempt_id}.json"
            if attempt_id
            else f"result-{time.time_ns()}-{uuid4().hex}.json"
        )
        if previous is None or previous.config_fingerprint is not None:
            result_path.replace(destination)
            return destination

        fingerprint = self._materialized_fingerprint(run_dir)
        enriched = previous.with_updates(
            result_version=max(previous.result_version, 2),
            attempt_id=f"legacy-{time.time_ns()}-{uuid4().hex[:8]}",
            config_fingerprint=fingerprint,
            finished_at_utc=previous.finished_at_utc
            or datetime.fromtimestamp(result_path.stat().st_mtime, timezone.utc).isoformat(),
        )
        destination = attempts_dir / f"result-{enriched.attempt_id}.json"
        enriched.write_json(destination)
        result_path.unlink()
        return destination

    def _safe_attempts_dir(self, run_dir: Path) -> Path:
        """Create the attempt archive without traversing links or reparse points."""
        metadata_dir = run_dir / ".lambdaforge"
        attempts_dir = metadata_dir / "attempts"
        for path in (metadata_dir, attempts_dir):
            if self._is_link_or_reparse_point(path):
                raise ValueError(f"Unsafe run-attempt archive directory: {path}")

        metadata_dir.mkdir(exist_ok=True)
        if self._is_link_or_reparse_point(metadata_dir):
            raise ValueError(f"Unsafe run-attempt archive directory: {metadata_dir}")
        attempts_dir.mkdir(exist_ok=True)
        if self._is_link_or_reparse_point(attempts_dir):
            raise ValueError(f"Unsafe run-attempt archive directory: {attempts_dir}")

        resolved_run_dir = run_dir.resolve()
        resolved_metadata_dir = metadata_dir.resolve()
        resolved_attempts_dir = attempts_dir.resolve()
        if not resolved_metadata_dir.is_relative_to(
            resolved_run_dir
        ) or not resolved_attempts_dir.is_relative_to(resolved_run_dir):
            raise ValueError(f"Unsafe run-attempt archive directory: {attempts_dir}")
        return attempts_dir

    @staticmethod
    def _is_link_or_reparse_point(path: Path) -> bool:
        """Detect symbolic links and Windows junction/reparse entries via lstat."""
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return False
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        attributes = getattr(metadata, "st_file_attributes", 0)
        return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)

    def _existing_terminal_results(
        self, run_configs: list[dict[str, Any]]
    ) -> list[RunResult] | None:
        """Return existing results only when every expected run already finished.

        A terminal run can be successful or failed: failed/OOM runs are excluded by
        aggregation but should not force retraining when the user only relaunches an
        already executed suite to regenerate summaries and plots.
        """
        results: list[RunResult] = []
        for run_config in run_configs:
            run_dir = self.experiment_run_dir(run_config)
            result = self._read_result(run_dir / "result.json")
            if result is None:
                return None

            if not self._result_matches_config(result, run_config, run_dir):
                return None

            if result.status is RunStatus.OK:
                completed = self._completed_result(run_config, run_dir)
                if completed is None:
                    return None
                result = completed
            elif not result.is_terminal:
                return None
            results.append(
                result.with_updates(
                    name=result.name
                    or ExperimentConfig.get_value(run_config, "experiment.name", "experiment"),
                    run_dir=result.run_dir or str(run_dir),
                    variant=(
                        result.variant
                        if result.variant is not None
                        else ExperimentConfig.get_value(run_config, "experiment.variant")
                    ),
                    seed=(
                        result.seed
                        if result.seed is not None
                        else ExperimentConfig.get_value(run_config, "experiment.seed")
                    ),
                )
            )

        return results

    def _variant_terminal(self, run_configs: list[dict[str, Any]], variant: str) -> bool:
        """Return true when every seed of ``variant`` has a terminal result."""
        for run_config in run_configs:
            if str(ExperimentConfig.get_value(run_config, "experiment.variant", "base")) != variant:
                continue

            result = self._read_result(self.experiment_run_dir(run_config) / "result.json")
            if result is None or result.get("status") not in {
                RunStatus.OK.value,
                RunStatus.FAILED.value,
            }:
                return False
        return True

    def _completed_result(self, config: Mapping[str, Any], run_dir: Path) -> RunResult | None:
        result = self._read_result(run_dir / "result.json")
        if not result or result.get("status") != RunStatus.OK.value:
            return None
        if not self._result_matches_config(result, config, run_dir):
            return None

        checkpoint_policy = str(
            ExperimentConfig.get_value(config, "trainer.checkpoint_policy", "last_and_best")
        )
        if checkpoint_policy != "none":
            from lambdaforge.experiments.retention.CheckpointResolver import CheckpointResolver

            if not CheckpointResolver(run_dir).candidates():
                return None

        required_artifacts = ExperimentConfig.get_value(
            config,
            "experiment.required_artifacts",
            [],
        )
        if not isinstance(required_artifacts, list):
            raise TypeError("experiment.required_artifacts must be a list of relative paths.")
        resolved_run_dir = run_dir.resolve()
        for artifact in required_artifacts:
            artifact_path = (run_dir / str(artifact)).resolve()
            if not artifact_path.is_relative_to(resolved_run_dir):
                raise ValueError("Required artifact paths must remain inside the run directory.")
            if not artifact_path.exists():
                return None

        if result.config_fingerprint is None:
            return result.with_updates(
                result_version=max(result.result_version, 2),
                config_fingerprint=RunFingerprint.digest(config),
            )
        return result

    def _result_matches_config(
        self,
        result: RunResult,
        config: Mapping[str, Any],
        run_dir: Path,
    ) -> bool:
        """Reject terminal markers created by a different scientific config."""
        expected = RunFingerprint.digest(config)
        observed = result.config_fingerprint or self._materialized_fingerprint(run_dir)
        return observed == expected

    @staticmethod
    def _materialized_fingerprint(run_dir: Path) -> str | None:
        """Recover a legacy result identity from its materialized YAML."""
        path = run_dir / "config.yaml"
        if not path.exists():
            return None
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            return None
        return RunFingerprint.digest(payload) if isinstance(payload, Mapping) else None

    def _latest_checkpoint(self, run_dir: Path) -> Path | None:
        from lambdaforge.experiments.retention.CheckpointResolver import CheckpointResolver

        result = self._read_result(run_dir / "result.json")
        return CheckpointResolver(run_dir).latest(result)

    def _prepare_rerun_config(self, config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
        prepared = copy.deepcopy(dict(config))
        if ExperimentConfig.get_value(prepared, "experiment.ckpt_path") is not None:
            return prepared

        if not bool(ExperimentConfig.get_value(prepared, "experiment.resume", True)):
            return prepared

        previous = self._read_result(run_dir / "result.json")
        if previous is not None and not self._result_matches_config(previous, prepared, run_dir):
            print(
                "Ignoring incompatible checkpoint: the scientific configuration changed.",
                flush=True,
            )
            return prepared
        materialized = self._materialized_fingerprint(run_dir)
        if materialized is not None and materialized != RunFingerprint.digest(prepared):
            print(
                "Ignoring incompatible checkpoint: the materialized configuration changed.",
                flush=True,
            )
            return prepared

        checkpoint = self._latest_checkpoint(run_dir)
        if checkpoint is None:
            return prepared

        ExperimentConfig.set_value(prepared, "experiment.ckpt_path", str(checkpoint))
        print(f"Resuming incomplete run from checkpoint: {checkpoint}", flush=True)
        return prepared

    def _prepare_scheduled_attempts(
        self,
        run_configs: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Retire stale results and resolve resume checkpoints before worker spawn.

        Valid completed runs remain untouched so their worker can take the normal
        skip path. Every run that will execute loses its previous terminal marker
        while the parent still owns the suite activity lease, preventing a worker
        that dies before entering the runner from exposing stale success on disk.
        """
        prepared: list[dict[str, Any]] = []
        for config in run_configs:
            run_dir = self.experiment_run_dir(config)
            rerun_completed = bool(
                ExperimentConfig.get_value(config, "experiment.rerun_completed", False)
            )
            if not rerun_completed and self._completed_result(config, run_dir) is not None:
                prepared.append(copy.deepcopy(config))
                continue

            attempt = self._prepare_rerun_config(config, run_dir)
            self._archive_previous_result(run_dir)
            prepared.append(attempt)
        return prepared

    def run_single_experiment(
        self,
        config: Mapping[str, Any],
        dry_run: bool = False,
        stop_event: Any | None = None,
    ) -> RunResult:
        """Run one config under the suite lifecycle lock.

        The executor uses the unlocked implementation because
        :meth:`run_experiment_config` already owns the same exclusive lease.
        """
        from lambdaforge.experiments.retention.ArtifactRetentionManager import (
            ArtifactRetentionManager,
        )
        from lambdaforge.experiments.retention.ArtifactRetentionPolicy import (
            ArtifactRetentionPolicy,
        )

        normalized = config if isinstance(config, ExperimentConfig) else ExperimentConfig(config)
        manager = ArtifactRetentionManager()
        policy = ArtifactRetentionPolicy.from_config(normalized)
        with manager.activity_lock(normalized, policy, shared=False):
            if not dry_run and not bool(
                ExperimentConfig.get_value(normalized, "experiment.rerun_completed", False)
            ):
                run_dir = self.experiment_run_dir(normalized)
                completed = self._completed_result(normalized, run_dir)
                if completed is not None:
                    print(f"Skipping completed run: {run_dir}", flush=True)
                    return completed.with_updates(skipped_existing=True)
            manager.invalidate_receipt(normalized)
            return self._run_single_experiment_unlocked(
                normalized,
                dry_run=dry_run,
                stop_event=stop_event,
            )

    def _run_single_experiment_unlocked(
        self,
        config: Mapping[str, Any],
        dry_run: bool = False,
        stop_event: Any | None = None,
    ) -> RunResult:
        r"""Run one materialized experiment config.

        ``stop_event`` is forwarded to the runner so a
        :class:`~lambdaforge.training.orchestration.TrainingOrchestrator.TrainingOrchestrator`
        can request a graceful stop across parallel jobs.
        """
        run_dir = self.experiment_run_dir(config)
        config_fingerprint = RunFingerprint.digest(config)

        if dry_run:
            timestamp = datetime.now(timezone.utc).isoformat()
            self._write_materialized_config(config, run_dir)
            self._write_environment_manifest(run_dir)
            return RunResult(
                result_version=2,
                name=ExperimentConfig.get_value(config, "experiment.name", "experiment"),
                run_dir=run_dir,
                variant=ExperimentConfig.get_value(config, "experiment.variant"),
                seed=ExperimentConfig.get_value(config, "experiment.seed"),
                status=RunStatus.DRY_RUN,
                config_fingerprint=config_fingerprint,
                started_at_utc=timestamp,
                finished_at_utc=timestamp,
            )

        if not bool(ExperimentConfig.get_value(config, "experiment.rerun_completed", False)):
            completed = self._completed_result(config, run_dir)
            if completed is not None:
                print(f"Skipping completed run: {run_dir}", flush=True)
                return completed.with_updates(skipped_existing=True)

        config = self._prepare_rerun_config(config, run_dir)
        config_fingerprint = RunFingerprint.digest(config)
        self._archive_previous_result(run_dir)
        self._write_materialized_config(config, run_dir)
        environment = self._write_environment_manifest(run_dir)

        self._write_hparams(config, run_dir)

        t0 = time.perf_counter()
        started_at_utc = datetime.now(timezone.utc).isoformat()
        attempt_id = (
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-"
            f"{config_fingerprint.removeprefix('sha256:')[:12]}-{uuid4().hex[:8]}"
        )
        seed = ExperimentConfig.get_value(config, "experiment.seed")
        self._set_seed(seed)

        plugins = PluginRegistry.default()
        with plugins.usage_session() as plugin_usage:
            try:
                datamodule = self._build_datamodule(config, plugins)
                model = ObjectFactory.build(config["model"], plugins=plugins)
                task, metrics = self._build_task(config, model, plugins)
                runner = self._build_runner(config, metrics, plugins)

                # Persist the complete static graph before long-running training,
                # then capture any later lazy resolutions again on terminal exit.
                self._write_environment_manifest(
                    run_dir,
                    environment.with_plugins(plugin_usage.descriptors()),
                )

                ckpt_path = ExperimentConfig.get_value(config, "experiment.ckpt_path")
                trainer = runner.fit(
                    task=task,
                    datamodule=datamodule,
                    ckpt_path=ckpt_path,
                    stop_event=stop_event,
                )

                checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
                best_model_path = getattr(checkpoint_callback, "best_model_path", None) or None
                last_model_path = getattr(checkpoint_callback, "last_model_path", None) or None
                interrupted = bool(
                    stop_event is not None and hasattr(stop_event, "is_set") and stop_event.is_set()
                )
                if not interrupted and bool(
                    ExperimentConfig.get_value(config, "experiment.test_after_fit", False)
                ):
                    runner.test(
                        task=task,
                        datamodule=datamodule,
                        ckpt_path=best_model_path or last_model_path,
                        stop_event=stop_event,
                    )
                interrupted = interrupted or bool(
                    stop_event is not None and hasattr(stop_event, "is_set") and stop_event.is_set()
                )

                directions = self._metric_directions(task)
                best_metric = self._best_metric(trainer)
                monitor = best_metric.get("monitor")
                monitor_mode = directions.get(monitor, "max") if monitor is not None else "max"
                result = RunResult(
                    result_version=2,
                    name=ExperimentConfig.get_value(config, "experiment.name", "experiment"),
                    run_dir=run_dir,
                    variant=ExperimentConfig.get_value(config, "experiment.variant"),
                    seed=seed,
                    status=RunStatus.INTERRUPTED if interrupted else RunStatus.OK,
                    seconds=time.perf_counter() - t0,
                    best_model_path=best_model_path,
                    last_model_path=last_model_path,
                    best_metric=best_metric,
                    best_epoch_metrics=self._best_epoch_metrics(
                        run_dir,
                        monitor,
                        monitor_mode,
                    ),
                    final_metrics=self._scalar_metrics(trainer.callback_metrics),
                    best_metrics=self._best_metrics_from_csv(run_dir, directions),
                    error="Cooperative stop requested." if interrupted else None,
                    attempt_id=attempt_id,
                    config_fingerprint=config_fingerprint,
                    started_at_utc=started_at_utc,
                    finished_at_utc=datetime.now(timezone.utc).isoformat(),
                )
            except BaseException:
                try:
                    self._write_environment_manifest(
                        run_dir,
                        environment.with_plugins(plugin_usage.descriptors()),
                    )
                except Exception as manifest_error:
                    print(
                        "WARNING: could not update plugin provenance after run failure: "
                        f"{manifest_error}",
                        flush=True,
                    )
                raise

            self._write_environment_manifest(
                run_dir,
                environment.with_plugins(plugin_usage.descriptors()),
            )

        result.write_json(run_dir / "result.json")
        return result

    def run_experiment_config(
        self,
        config: Mapping[str, Any],
        dry_run: bool = False,
        execution_overrides: Mapping[str, Any] | None = None,
        aggregate_plots: bool = True,
        on_run_finished: Callable[[Mapping[str, Any], Mapping[str, Any]], None] | None = None,
    ) -> list[RunResult]:
        r"""Expand and run all variants from one experiment config.

        Dry runs stay in-process and sequential. Real runs are dispatched through
        :mod:`lambdaforge.experiments.execution`, which honors the ``execution:`` block
        (sequential / parallel / ddp) and writes each run's ``train.log``. Once
        every run finishes, :mod:`lambdaforge.experiments.aggregate` summarizes seeds.

        ``execution_overrides`` (from the CLI) take precedence over the YAML
        ``execution:`` block.
        """
        normalized_config = (
            config if isinstance(config, ExperimentConfig) else ExperimentConfig(config)
        )
        config = normalized_config
        run_configs = normalized_config.expand()

        if dry_run:
            return [
                self.run_single_experiment(run_config, dry_run=True) for run_config in run_configs
            ]

        # Lazy imports break the runner <-> execution/aggregate import cycle.
        from lambdaforge.experiments.ExecutionConfig import ExecutionConfig
        from lambdaforge.experiments.ExperimentAggregator import ExperimentAggregator
        from lambdaforge.experiments.ExperimentExecutor import ExperimentExecutor
        from lambdaforge.experiments.retention.ArtifactRetentionManager import (
            ArtifactRetentionManager,
        )
        from lambdaforge.experiments.retention.ArtifactRetentionPolicy import (
            ArtifactRetentionPolicy,
        )

        aggregator = ExperimentAggregator()
        retention_manager = ArtifactRetentionManager()
        retention_policy = ArtifactRetentionPolicy.from_config(config)

        if not bool(ExperimentConfig.get_value(config, "experiment.rerun_completed", False)):
            existing_results = self._existing_terminal_results(run_configs)
            if existing_results is not None:
                print(
                    "All expected runs already have terminal result.json files; "
                    "regenerating aggregates and plots without launching training.",
                    flush=True,
                )
                aggregator.write(
                    config,
                    existing_results,
                    make_plots=aggregate_plots,
                    global_plots=True,
                    variant_plot_policy="available",
                )
                return existing_results

        postprocess_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="experiment-postprocess"
        )
        postprocess_futures: list[Future] = []
        queued_variants: set[str] = set()

        def _postprocess_variant(
            variant: str,
            run_config: Mapping[str, Any],
            result: Mapping[str, Any],
        ) -> None:
            try:
                print(
                    f"Refreshing aggregates after completed variant: {variant}",
                    flush=True,
                )
                aggregator.write(
                    config,
                    make_plots=aggregate_plots,
                    global_plots=False,
                    variant_plot_policy="terminal",
                    final=False,
                )
                if on_run_finished is not None:
                    on_run_finished(run_config, result)
            except Exception as exc:
                print(
                    f"WARNING: post-run postprocessing failed for variant={variant}: {exc}",
                    flush=True,
                )

        def _refresh_after_run(run_config: Mapping[str, Any], result: Mapping[str, Any]) -> None:
            variant = ExperimentConfig.get_value(run_config, "experiment.variant", "base")
            seed = ExperimentConfig.get_value(run_config, "experiment.seed")
            variant = str(variant)
            if variant in queued_variants:
                return

            if not self._variant_terminal(run_configs, variant):
                return

            queued_variants.add(variant)
            print(
                f"Queueing postprocessing after completed variant: variant={variant}, seed={seed}",
                flush=True,
            )
            postprocess_futures.append(
                postprocess_executor.submit(
                    _postprocess_variant, variant, dict(run_config), dict(result)
                )
            )

        activity_lock = retention_manager.activity_lock(
            normalized_config,
            retention_policy,
            shared=False,
        )
        activity_lock.acquire()
        try:
            retention_manager.invalidate_receipt(normalized_config)
            execution_config = ExecutionConfig.from_mapping(config, execution_overrides)
            scheduled_configs = self._prepare_scheduled_attempts(run_configs)
            results = ExperimentExecutor().run(
                scheduled_configs,
                execution_config,
                on_run_finished=_refresh_after_run,
            )
        finally:
            try:
                for future in postprocess_futures:
                    future.result()
            finally:
                postprocess_executor.shutdown(wait=True)
                activity_lock.release()

        aggregator.write(
            config,
            results,
            make_plots=aggregate_plots,
            global_plots=True,
            variant_plot_policy="available",
        )

        return results

    def run_experiment_file(
        self,
        path: str | Path,
        dry_run: bool = False,
        execution_overrides: Mapping[str, Any] | None = None,
        aggregate_plots: bool = True,
        on_run_finished: Callable[[Mapping[str, Any], Mapping[str, Any]], None] | None = None,
    ) -> list[RunResult]:
        r"""Load, expand and run a YAML experiment file."""
        config = ExperimentConfig.from_yaml(path)
        return self.run_experiment_config(
            config=config,
            dry_run=dry_run,
            execution_overrides=execution_overrides,
            aggregate_plots=aggregate_plots,
            on_run_finished=on_run_finished,
        )
