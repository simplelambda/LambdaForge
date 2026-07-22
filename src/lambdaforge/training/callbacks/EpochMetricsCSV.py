"""Callback that writes dense epoch metrics to CSV."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from lambdaforge.integrations.Lightning import (
    CallbackBase,
    LightningModuleBase,
    TrainerType,
)
from lambdaforge.training.callbacks.LogKeyFilter import LogKeyFilter


class EpochMetricsCSV(CallbackBase):
    r"""Write one dense ``metrics.csv`` row per completed epoch.

    Lightning's built-in CSV logger can emit several sparse rows for the same
    epoch because losses, metrics and runtime stats are logged from different
    hooks. This callback mirrors ``trainer.callback_metrics`` after each epoch
    into a single wide row, which is easier to plot and post-process.
    """

    def __init__(
        self,
        filename: str = "metrics.csv",
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        continue_existing: bool = False,
    ) -> None:
        super().__init__()
        self.filename = filename
        self.key_filter = LogKeyFilter(include=include, exclude=exclude)
        self.continue_existing = bool(continue_existing)
        self._last_completed_epoch: int | None = None
        self._last_validation_epoch: int | None = None
        self._last_written_epoch: int | None = None
        self._rows: list[dict[str, float | int | str]] = []

    def on_fit_start(self, trainer: TrainerType, pl_module: LightningModuleBase) -> None:
        if self._is_global_zero(trainer):
            path = self._path(trainer)
            if path.exists():
                if self._should_continue_existing_file(trainer):
                    self._rows = self._read_existing_rows(path)
                    self._last_written_epoch = self._last_epoch_in_rows()
                else:
                    path.unlink()

    def on_train_epoch_end(self, trainer: TrainerType, pl_module: LightningModuleBase) -> None:
        self._last_completed_epoch = int(trainer.current_epoch)

    def on_validation_epoch_end(self, trainer: TrainerType, pl_module: LightningModuleBase) -> None:
        if not trainer.sanity_checking:
            self._last_validation_epoch = int(trainer.current_epoch)

    def on_train_epoch_start(self, trainer: TrainerType, pl_module: LightningModuleBase) -> None:
        self._write_epoch_if_ready(trainer)

    def on_fit_end(self, trainer: TrainerType, pl_module: LightningModuleBase) -> None:
        self._write_epoch_if_ready(trainer)

    def _write_epoch_if_ready(self, trainer: TrainerType) -> None:
        epoch = self._last_completed_epoch
        if epoch is None or epoch == self._last_written_epoch:
            return
        if not self._is_global_zero(trainer):
            self._last_written_epoch = epoch
            return

        row: dict[str, float | int] = {"epoch": epoch + 1}
        row.update(self._scalar_metrics(trainer.callback_metrics))
        self._upsert_row(row)
        self._rewrite_file(trainer)
        self._last_written_epoch = epoch

    def _rewrite_file(self, trainer: TrainerType) -> None:
        path = self._path(trainer)
        path.parent.mkdir(parents=True, exist_ok=True)

        fields = ["epoch"]
        for row in self._rows:
            for key in row:
                if key not in fields:
                    fields.append(key)

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in self._rows:
                writer.writerow(row)

    def _path(self, trainer: TrainerType) -> Path:
        return Path(str(trainer.default_root_dir)) / self.filename

    def _should_continue_existing_file(self, trainer: TrainerType) -> bool:
        del trainer
        return self.continue_existing

    def _read_existing_rows(self, path: Path) -> list[dict[str, float | int | str]]:
        with open(path, encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]

    def _last_epoch_in_rows(self) -> int | None:
        epochs = [
            int(float(row["epoch"])) - 1 for row in self._rows if row.get("epoch") not in (None, "")
        ]
        return max(epochs) if epochs else None

    def _upsert_row(self, row: dict[str, float | int]) -> None:
        epoch = row.get("epoch")
        for index, existing in enumerate(self._rows):
            if existing.get("epoch") in {epoch, str(epoch)}:
                merged = dict(existing)
                merged.update(row)
                self._rows[index] = merged
                return
        normalized: dict[str, float | int | str] = dict(row)
        self._rows.append(normalized)

    @staticmethod
    def _is_global_zero(trainer: TrainerType) -> bool:
        return bool(getattr(trainer, "is_global_zero", True))

    def _scalar_metrics(self, raw_metrics: dict[str, Any]) -> dict[str, float]:
        metrics: dict[str, float] = {}

        for key, value in raw_metrics.items():
            if key in {"epoch", "step", "hp_metric"} or str(key).endswith("_step"):
                continue
            if not self.key_filter.accepts(str(key)):
                continue

            scalar = self._to_float(value)
            if scalar is not None:
                metrics[str(key)] = scalar

        return metrics

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if torch.is_tensor(value):
            if value.numel() != 1:
                return None
            return float(value.detach().cpu().item())

        if isinstance(value, (int, float)):
            return float(value)

        return None
