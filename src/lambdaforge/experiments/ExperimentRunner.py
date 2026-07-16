"""Implementation of the ExperimentRunner object."""

from __future__ import annotations

import copy
import csv
import json
import random
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
from lambdaforge.experiments.ObjectFactory import ObjectFactory
from lambdaforge.experiments.RunStatus import RunStatus


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

    def _build_datamodule(self, config: Mapping[str, Any]):
        from lambdaforge.training.data.LightningDataModule import LightningDataModule

        data_cfg = config.get("data", {})
        if not isinstance(data_cfg, Mapping):
            raise TypeError("'data' must be a mapping.")

        train = ObjectFactory.build(data_cfg.get("train")) if "train" in data_cfg else None
        val = ObjectFactory.build(data_cfg.get("val")) if "val" in data_cfg else None
        test = ObjectFactory.build(data_cfg.get("test")) if "test" in data_cfg else None

        datamodule_cfg = data_cfg.get("datamodule", {})
        if datamodule_cfg is None:
            datamodule_cfg = {}
        if not isinstance(datamodule_cfg, Mapping):
            raise TypeError("data.datamodule must be a mapping.")

        target = datamodule_cfg.get("target")
        params = ObjectFactory.build(datamodule_cfg.get("params", {}))
        if not isinstance(params, Mapping):
            raise TypeError("data.datamodule.params must be a mapping.")

        params = dict(params)
        params.setdefault("train", train)
        params.setdefault("val", val)
        params.setdefault("test", test)

        cls = ObjectFactory.import_object(str(target)) if target else LightningDataModule
        return cls(**params)

    def _build_task(self, config: Mapping[str, Any], model: torch.nn.Module):
        from lambdaforge.training.LightningTask import LightningTask

        losses = self._as_list(ObjectFactory.build(config.get("losses")))
        if not losses:
            raise RuntimeError("Experiment config must define at least one loss.")

        shared_metrics = self._as_list(ObjectFactory.build(config.get("metrics", [])))
        train_metrics_spec = config.get("train_metrics")
        val_metrics_spec = config.get("val_metrics")
        test_metrics_spec = config.get("test_metrics")
        train_metrics = (
            self._as_list(ObjectFactory.build(train_metrics_spec))
            if train_metrics_spec is not None
            else shared_metrics
        )
        val_metrics = (
            self._as_list(ObjectFactory.build(val_metrics_spec))
            if val_metrics_spec is not None
            else None
        )
        test_metrics = (
            self._as_list(ObjectFactory.build(test_metrics_spec))
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
        optimizer_kwargs = ObjectFactory.build(optimizer_cfg.get("params", {}))

        scheduler_cfg = config.get("scheduler")
        scheduler_cls = None
        scheduler_kwargs = {}
        scheduler_config = None
        if scheduler_cfg is not None:
            if not isinstance(scheduler_cfg, Mapping):
                raise TypeError("'scheduler' must be a mapping.")
            scheduler_cls = ObjectFactory.import_object(str(scheduler_cfg["ref"]))
            scheduler_kwargs = ObjectFactory.build(scheduler_cfg.get("params", {}))
            scheduler_config = ObjectFactory.build(scheduler_cfg.get("lightning", None))

        task_cfg = config.get("task", {})
        if task_cfg is None:
            task_cfg = {}
        if not isinstance(task_cfg, Mapping):
            raise TypeError("'task' must be a mapping.")

        target = task_cfg.get("target")
        params = ObjectFactory.build(task_cfg.get("params", {}))
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

    def _build_runner(self, config: Mapping[str, Any], metrics: list[Any]):
        from lambdaforge.training.LightningRunner import LightningRunner
        from lambdaforge.training.LightningTrainConfig import LightningTrainConfig

        trainer_cfg = dict(config.get("trainer", {}) or {})
        run_dir = self.experiment_run_dir(config)
        trainer_cfg.setdefault("default_root_dir", str(run_dir))

        train_config = LightningTrainConfig(**ObjectFactory.build(trainer_cfg))
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
        params = ObjectFactory.build(runner_cfg.get("params", {}))
        if not isinstance(params, Mapping):
            raise TypeError("runner.params must be a mapping.")

        params = dict(params)
        if "callbacks" in config:
            params.setdefault("callbacks", self._as_list(ObjectFactory.build(config["callbacks"])))
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

    def _read_result(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _existing_terminal_results(
        self, run_configs: list[dict[str, Any]]
    ) -> list[dict[str, Any]] | None:
        """Return existing results only when every expected run already finished.

        A terminal run can be successful or failed: failed/OOM runs are excluded by
        aggregation but should not force retraining when the user only relaunches an
        already executed suite to regenerate summaries and plots.
        """
        results: list[dict[str, Any]] = []
        for run_config in run_configs:
            run_dir = self.experiment_run_dir(run_config)
            result = self._read_result(run_dir / "result.json")
            if result is None:
                return None

            if result.get("status") not in {RunStatus.OK.value, RunStatus.FAILED.value}:
                return None

            result = dict(result)
            result.setdefault(
                "name",
                ExperimentConfig.get_value(run_config, "experiment.name", "experiment"),
            )
            result.setdefault("run_dir", str(run_dir))
            result.setdefault(
                "variant", ExperimentConfig.get_value(run_config, "experiment.variant")
            )
            result.setdefault("seed", ExperimentConfig.get_value(run_config, "experiment.seed"))
            results.append(result)

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

    def _path_exists(self, value: Any) -> bool:
        if value in (None, ""):
            return False
        return Path(str(value)).exists()

    def _completed_result(self, config: Mapping[str, Any], run_dir: Path) -> dict[str, Any] | None:
        result = self._read_result(run_dir / "result.json")
        if not result or result.get("status") != RunStatus.OK.value:
            return None

        checkpoint_policy = str(
            ExperimentConfig.get_value(config, "trainer.checkpoint_policy", "last_and_best")
        )
        if checkpoint_policy != "none":
            checkpoint_paths = [
                result.get("best_model_path"),
                result.get("last_model_path"),
            ]
            if not any(self._path_exists(path) for path in checkpoint_paths):
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

        return result

    def _latest_checkpoint(self, run_dir: Path) -> Path | None:
        checkpoint_dir = run_dir / "checkpoints"
        candidates = [
            checkpoint_dir / "last.ckpt",
        ]

        result = self._read_result(run_dir / "result.json") or {}
        for key in ("last_model_path", "best_model_path"):
            value = result.get(key)
            if value not in (None, ""):
                candidates.append(Path(str(value)))

        if checkpoint_dir.exists():
            candidates.extend(
                sorted(
                    checkpoint_dir.glob("*.ckpt"),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
            )

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _prepare_rerun_config(self, config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
        prepared = copy.deepcopy(dict(config))
        if ExperimentConfig.get_value(prepared, "experiment.ckpt_path") is not None:
            return prepared

        if not bool(ExperimentConfig.get_value(prepared, "experiment.resume", True)):
            return prepared

        checkpoint = self._latest_checkpoint(run_dir)
        if checkpoint is None:
            return prepared

        ExperimentConfig.set_value(prepared, "experiment.ckpt_path", str(checkpoint))
        print(f"Resuming incomplete run from checkpoint: {checkpoint}", flush=True)
        return prepared

    def run_single_experiment(
        self,
        config: Mapping[str, Any],
        dry_run: bool = False,
        stop_event: Any | None = None,
    ) -> dict[str, Any]:
        r"""Run one materialized experiment config.

        ``stop_event`` is forwarded to the runner so a
        :class:`~lambdaforge.training.orchestration.TrainingOrchestrator.TrainingOrchestrator`
        can request a graceful stop across parallel jobs.
        """
        run_dir = self.experiment_run_dir(config)

        if dry_run:
            self._write_materialized_config(config, run_dir)
            return {
                "name": ExperimentConfig.get_value(config, "experiment.name", "experiment"),
                "run_dir": str(run_dir),
                "variant": ExperimentConfig.get_value(config, "experiment.variant"),
                "seed": ExperimentConfig.get_value(config, "experiment.seed"),
                "status": RunStatus.DRY_RUN.value,
            }

        if not bool(ExperimentConfig.get_value(config, "experiment.rerun_completed", False)):
            completed = self._completed_result(config, run_dir)
            if completed is not None:
                print(f"Skipping completed run: {run_dir}", flush=True)
                completed = dict(completed)
                completed["skipped_existing"] = True
                return completed

        config = self._prepare_rerun_config(config, run_dir)
        self._write_materialized_config(config, run_dir)

        self._write_hparams(config, run_dir)

        t0 = time.perf_counter()
        seed = ExperimentConfig.get_value(config, "experiment.seed")
        self._set_seed(seed)

        datamodule = self._build_datamodule(config)
        model = ObjectFactory.build(config["model"])
        task, metrics = self._build_task(config, model)
        runner = self._build_runner(config, metrics)

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
        if bool(ExperimentConfig.get_value(config, "experiment.test_after_fit", False)):
            runner.test(
                task=task,
                datamodule=datamodule,
                ckpt_path=best_model_path or last_model_path,
                stop_event=stop_event,
            )

        directions = self._metric_directions(task)
        best_metric = self._best_metric(trainer)
        monitor = best_metric.get("monitor")
        monitor_mode = directions.get(monitor, "max") if monitor is not None else "max"
        result = {
            "name": ExperimentConfig.get_value(config, "experiment.name", "experiment"),
            "run_dir": str(run_dir),
            "variant": ExperimentConfig.get_value(config, "experiment.variant"),
            "seed": seed,
            "status": RunStatus.OK.value,
            "seconds": time.perf_counter() - t0,
            "best_model_path": best_model_path,
            "last_model_path": last_model_path,
            "best_metric": best_metric,
            "best_epoch_metrics": self._best_epoch_metrics(run_dir, monitor, monitor_mode),
            "final_metrics": self._scalar_metrics(trainer.callback_metrics),
            "best_metrics": self._best_metrics_from_csv(run_dir, directions),
        }
        with open(run_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return result

    def run_experiment_config(
        self,
        config: Mapping[str, Any],
        dry_run: bool = False,
        execution_overrides: Mapping[str, Any] | None = None,
        aggregate_plots: bool = True,
        on_run_finished: Callable[[Mapping[str, Any], Mapping[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        r"""Expand and run all variants from one experiment config.

        Dry runs stay in-process and sequential. Real runs are dispatched through
        :mod:`lambdaforge.experiments.execution`, which honors the ``execution:`` block
        (sequential / parallel / ddp) and writes each run's ``train.log``. Once
        every run finishes, :mod:`lambdaforge.experiments.aggregate` summarizes seeds.

        ``execution_overrides`` (from the CLI) take precedence over the YAML
        ``execution:`` block.
        """
        run_configs = ExperimentConfig.expand_mapping(config)

        if dry_run:
            return [
                self.run_single_experiment(run_config, dry_run=True) for run_config in run_configs
            ]

        # Lazy imports break the runner <-> execution/aggregate import cycle.
        from lambdaforge.experiments.ExecutionConfig import ExecutionConfig
        from lambdaforge.experiments.ExperimentAggregator import ExperimentAggregator
        from lambdaforge.experiments.ExperimentExecutor import ExperimentExecutor

        aggregator = ExperimentAggregator()

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

        try:
            execution_config = ExecutionConfig.from_mapping(config, execution_overrides)
            results = ExperimentExecutor().run(
                run_configs,
                execution_config,
                on_run_finished=_refresh_after_run,
            )
        finally:
            for future in postprocess_futures:
                future.result()
            postprocess_executor.shutdown(wait=True)

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
    ) -> list[dict[str, Any]]:
        r"""Load, expand and run a YAML experiment file."""
        config = ExperimentConfig.from_yaml(path)
        return self.run_experiment_config(
            config=config,
            dry_run=dry_run,
            execution_overrides=execution_overrides,
            aggregate_plots=aggregate_plots,
            on_run_finished=on_run_finished,
        )
