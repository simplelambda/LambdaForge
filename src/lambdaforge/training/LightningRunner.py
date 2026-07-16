"""Implementation of the LightningRunner object."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from lambdaforge.integrations.Lightning import (
    CallbackBase,
    Lightning,
    LightningDataModuleBase,
    LightningModuleBase,
    LoggerType,
    TrainerType,
)
from lambdaforge.metrics.Metric import Metric
from lambdaforge.training.callbacks.EpochLogPrinter import EpochLogPrinter
from lambdaforge.training.callbacks.EpochMetricsCSV import EpochMetricsCSV
from lambdaforge.training.callbacks.EpochStats import EpochStats
from lambdaforge.training.callbacks.StopEventCallback import StopEventCallback
from lambdaforge.training.CheckpointPolicy import CheckpointPolicy
from lambdaforge.training.LightningTrainConfig import LightningTrainConfig
from lambdaforge.training.LoggerMode import LoggerMode

pl = Lightning.module
CSVLogger = Lightning.CSVLogger
EarlyStopping = Lightning.EarlyStopping
ModelCheckpoint = Lightning.ModelCheckpoint


class LightningRunner:
    r"""Builds and runs a Lightning ``Trainer`` from project conventions.

    Translates :class:`LightningTrainConfig` settings and project
    ``Metric`` objects into the Lightning-native ``Trainer``,
    ``ModelCheckpoint``, and ``EarlyStopping`` callbacks.

    Parameters
    ----------
    config : LightningTrainConfig | None
        Training configuration. Defaults to :class:`LightningTrainConfig`
        with all defaults.
    early_stopping_metric : Metric | None
        Metric monitored for early stopping. The Lightning monitor key will
        be ``val_{metric.name}``. Without one, ``val_loss`` is used whenever
        early-stopping patience is configured.
    checkpoint_metric : Metric | None
        Metric used to select the best checkpoint. ``None`` uses
        ``val_loss`` (minimised).
    callbacks : Sequence[Callback] | None
        Extra Lightning callbacks appended after the built-in ones.
    """

    def __init__(
        self,
        config: LightningTrainConfig | None = None,
        early_stopping_metric: Metric | None = None,
        checkpoint_metric: Metric | None = None,
        callbacks: Sequence[CallbackBase] | None = None,
    ) -> None:
        self.config = config or LightningTrainConfig()
        self.early_stopping_metric = early_stopping_metric
        self.checkpoint_metric = checkpoint_metric
        self.extra_callbacks = list(callbacks or [])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        task: LightningModuleBase,
        datamodule: LightningDataModuleBase,
        ckpt_path: str | Path | None = None,
        stop_event: Any | None = None,
    ) -> TrainerType:
        """Run ``trainer.fit`` and return the trainer."""
        trainer = self.build_trainer(stop_event=stop_event)
        trainer.fit(model=task, datamodule=datamodule, ckpt_path=ckpt_path)
        return trainer

    def test(
        self,
        task: LightningModuleBase,
        datamodule: LightningDataModuleBase,
        ckpt_path: str | Path | None = "best",
        stop_event: Any | None = None,
    ) -> TrainerType:
        """Run ``trainer.test`` and return the trainer."""
        trainer = self.build_trainer(stop_event=stop_event)
        trainer.test(model=task, datamodule=datamodule, ckpt_path=ckpt_path)
        return trainer

    def build_trainer(self, stop_event: Any | None = None) -> TrainerType:
        """Construct a ``pl.Trainer`` with all configured callbacks."""
        if self.config.matmul_precision:
            torch.set_float32_matmul_precision(self.config.matmul_precision)

        # Documented in the run log so every run records its numeric setup.
        print(
            f"[precision] trainer_precision={self.config.precision} "
            f"float32_matmul_precision={self.config.matmul_precision}",
            flush=True,
        )

        callbacks: list[CallbackBase] = []
        callbacks.extend(self._build_checkpoint_callbacks())
        callbacks.extend(self._build_early_stopping_callbacks())

        if self.config.track_epoch_stats:
            callbacks.append(EpochStats())

        if self.config.write_epoch_metrics_csv:
            callbacks.append(
                EpochMetricsCSV(
                    include=self.config.epoch_metrics_include,
                    exclude=self.config.epoch_metrics_exclude,
                )
            )

        if self.config.print_epoch_table:
            callbacks.append(
                EpochLogPrinter(
                    include=self.config.epoch_console_include,
                    exclude=self.config.epoch_console_exclude,
                )
            )

        if stop_event is not None:
            callbacks.append(StopEventCallback(stop_event))

        callbacks.extend(self.extra_callbacks)

        framework_kwargs: dict[str, Any] = {
            "max_epochs": self.config.max_epochs,
            "accelerator": self.config.accelerator,
            "devices": self.config.devices,
            "strategy": self.config.strategy,
            "precision": self.config.precision,
            "accumulate_grad_batches": self.config.accumulate_grad_batches,
            "gradient_clip_val": self.config.gradient_clip_val,
            "check_val_every_n_epoch": self.config.check_val_every_n_epoch,
            "default_root_dir": self.config.default_root_dir,
            "log_every_n_steps": self.config.log_every_n_steps,
            "logger": self._build_logger(),
            "callbacks": callbacks,
            "enable_checkpointing": (self.config.checkpoint_policy != CheckpointPolicy.NONE.value),
            "num_sanity_val_steps": self.config.num_sanity_val_steps,
            "enable_progress_bar": self.config.enable_progress_bar,
            "deterministic": self.config.deterministic,
        }
        extra_kwargs = dict(self.config.trainer_kwargs or {})
        conflicts = sorted(framework_kwargs.keys() & extra_kwargs.keys())
        if conflicts:
            raise ValueError(
                "trainer_kwargs cannot override framework-managed keys: " + ", ".join(conflicts)
            )
        return pl.Trainer(**framework_kwargs, **extra_kwargs)

    def _build_logger(self) -> LoggerType | bool:
        """Build the metrics logger.

        ``"csv"`` is handled by :class:`EpochMetricsCSV` so every epoch is one
        dense row. ``"lightning_csv"`` keeps Lightning's native CSVLogger.
        """
        if not isinstance(self.config.logger, str):
            return self.config.logger

        if self.config.logger in {LoggerMode.NONE.value, LoggerMode.CSV.value}:
            return False

        if self.config.logger == LoggerMode.LIGHTNING_CSV.value:
            return CSVLogger(save_dir=str(self.config.default_root_dir), name="", version="")

        raise ValueError(f"Unknown logger mode: {self.config.logger!r}.")

    # ------------------------------------------------------------------
    # Callback builders
    # ------------------------------------------------------------------

    def _build_checkpoint_callbacks(self) -> list[CallbackBase]:
        policy = self.config.checkpoint_policy

        if policy == CheckpointPolicy.NONE.value:
            return []

        monitor, mode = self._resolve_monitor(
            self.checkpoint_metric,
            self.config.checkpoint_monitor,
            self.config.checkpoint_mode,
            "checkpoint",
        )
        dirpath = Path(self.config.default_root_dir) / "checkpoints"

        if policy == CheckpointPolicy.LAST.value:
            return [ModelCheckpoint(dirpath=dirpath, filename="last", save_last=True, save_top_k=0)]

        if policy == CheckpointPolicy.BEST.value:
            return [
                ModelCheckpoint(
                    dirpath=dirpath,
                    filename="best-{epoch:03d}",
                    monitor=monitor,
                    mode=mode,
                    save_top_k=1,
                    save_last=False,
                )
            ]

        if policy == CheckpointPolicy.LAST_AND_BEST.value:
            return [
                ModelCheckpoint(
                    dirpath=dirpath,
                    filename="best-{epoch:03d}",
                    monitor=monitor,
                    mode=mode,
                    save_top_k=1,
                    save_last=True,
                )
            ]

        if policy == CheckpointPolicy.ALL.value:
            return [
                ModelCheckpoint(
                    dirpath=dirpath,
                    filename="epoch-{epoch:03d}",
                    save_top_k=-1,
                    save_last=False,
                )
            ]

        raise ValueError(f"Unknown checkpoint_policy: {policy!r}")

    def _build_early_stopping_callbacks(self) -> list[CallbackBase]:
        if self.config.early_stopping_patience is None:
            return []

        monitor, mode = self._resolve_monitor(
            self.early_stopping_metric,
            self.config.early_stopping_monitor,
            self.config.early_stopping_mode,
            "early_stopping",
        )

        return [
            EarlyStopping(
                monitor=monitor,
                mode=mode,
                patience=self.config.early_stopping_patience,
                min_delta=self.config.early_stopping_min_delta,
            )
        ]

    @staticmethod
    def _resolve_monitor(
        metric: Metric | None,
        explicit_monitor: str | None,
        explicit_mode: str | None,
        purpose: str,
    ) -> tuple[str, str]:
        default_monitor = f"val_{metric.name}" if metric is not None else "val_loss"
        default_mode = "max" if metric is not None and metric.higher_is_better else "min"
        monitor = explicit_monitor or default_monitor
        if explicit_mode is not None:
            return monitor, explicit_mode
        if monitor == default_monitor:
            return monitor, default_mode
        if monitor.endswith("loss"):
            return monitor, "min"
        raise ValueError(
            f"trainer.{purpose}_mode is required when {purpose}_monitor "
            f"selects the custom key {monitor!r}."
        )
