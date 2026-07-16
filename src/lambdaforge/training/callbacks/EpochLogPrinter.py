"""Console callback for compact epoch summaries."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from lambdaforge.integrations.Lightning import (
    CallbackBase,
    LightningModuleBase,
    TrainerType,
)
from lambdaforge.training.callbacks.LogKeyFilter import LogKeyFilter


class EpochLogPrinter(CallbackBase):
    r"""Print a compact per-epoch metrics table to stdout.

    Lightning's CSV logger writes scalar metrics to ``metrics.csv`` but does
    not make those values visible in captured ``train.log`` files. This
    callback mirrors the epoch-level scalars to stdout in a human-readable
    table, so the same information is available while tailing a log.

    The callback is problem-agnostic: it reads whatever scalar metrics the
    task/logger exposed (``train_*``, ``val_*``, ``test_*`` plus runtime stats)
    and does not know anything about a specific dataset or model.
    """

    def __init__(
        self,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.key_filter = LogKeyFilter(include=include, exclude=exclude)
        self._last_completed_epoch: int | None = None
        self._last_validation_epoch: int | None = None
        self._last_printed_epoch: int | None = None

    def on_train_epoch_end(self, trainer: TrainerType, pl_module: LightningModuleBase) -> None:
        self._last_completed_epoch = int(trainer.current_epoch)

    def on_validation_epoch_end(self, trainer: TrainerType, pl_module: LightningModuleBase) -> None:
        if not trainer.sanity_checking:
            self._last_validation_epoch = int(trainer.current_epoch)

    def on_train_epoch_start(self, trainer: TrainerType, pl_module: LightningModuleBase) -> None:
        self._print_epoch_if_ready(trainer)

    def on_fit_end(self, trainer: TrainerType, pl_module: LightningModuleBase) -> None:
        self._print_epoch_if_ready(trainer)

    def _print_epoch_if_ready(self, trainer: TrainerType) -> None:
        epoch = self._last_completed_epoch
        if epoch is None or epoch == self._last_printed_epoch:
            return
        if not self._is_global_zero(trainer):
            self._last_printed_epoch = epoch
            return

        metrics = self._scalar_metrics(trainer.callback_metrics)
        lines = self._format_epoch(epoch, trainer.max_epochs, metrics)
        if lines:
            print("\n".join(lines), flush=True)

        self._last_printed_epoch = epoch

    def _format_epoch(
        self,
        epoch: int,
        max_epochs: int | None,
        metrics: dict[str, float],
    ) -> list[str]:
        train = self._split_metrics(metrics, "train")
        val = self._split_metrics(metrics, "val") if self._last_validation_epoch == epoch else {}
        test = self._split_metrics(metrics, "test")

        if self._has_level_metrics(train, val, test):
            return self._format_level_epoch(epoch, max_epochs, metrics, train, val, test)

        metric_names = self._ordered_metric_names(train, val, test)
        rows: list[list[str]] = []
        for split_name, values in (("train", train), ("val", val), ("test", test)):
            if values:
                rows.append(
                    [split_name] + [self._format_value(values.get(name)) for name in metric_names]
                )

        stats = self._runtime_stats(metrics)
        title = self._epoch_title(epoch, max_epochs, stats)
        if not rows:
            return [title]

        header = ["split"] + metric_names
        widths = [max(len(row[index]) for row in [header] + rows) for index in range(len(header))]

        def render(row: list[str]) -> str:
            return " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))

        divider = "-+-".join("-" * width for width in widths)
        body = [
            title,
            render(header),
            divider,
            *[render(row) for row in rows],
        ]
        return self._with_epoch_separator(body)

    def _format_level_epoch(
        self,
        epoch: int,
        max_epochs: int | None,
        metrics: dict[str, float],
        train: dict[str, float],
        val: dict[str, float],
        test: dict[str, float],
    ) -> list[str]:
        row_maps: list[tuple[str, str, dict[str, float]]] = []
        for split_name, values in (("train", train), ("val", val), ("test", test)):
            row_maps.extend(
                (split_name, level, level_values)
                for level, level_values in self._metrics_by_level(values)
            )

        stats = self._runtime_stats(metrics)
        title = self._epoch_title(epoch, max_epochs, stats)
        if not row_maps:
            return self._with_epoch_separator([title])

        metric_names = self._ordered_metric_names(*(values for _, _, values in row_maps))
        rows = [
            [split_name, level] + [self._format_value(values.get(name)) for name in metric_names]
            for split_name, level, values in row_maps
        ]
        header = ["split", "level"] + metric_names

        return self._with_epoch_separator([title, *self._render_table(header, rows)])

    @staticmethod
    def _is_global_zero(trainer: TrainerType) -> bool:
        return bool(getattr(trainer, "is_global_zero", True))

    def _scalar_metrics(self, raw_metrics: dict[str, Any]) -> dict[str, float]:
        metrics: dict[str, float] = {}

        for key, value in raw_metrics.items():
            if key in {"epoch", "step", "hp_metric"} or key.endswith("_step"):
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

    @staticmethod
    def _split_metrics(metrics: dict[str, float], split: str) -> dict[str, float]:
        prefix = f"{split}_"
        return {
            key.removeprefix(prefix): value
            for key, value in metrics.items()
            if key.startswith(prefix)
        }

    @staticmethod
    def _ordered_metric_names(*groups: dict[str, float]) -> list[str]:
        names = {name for group in groups for name in group}
        preferred = EpochLogPrinter._preferred_metric_names()
        ordered = [name for name in preferred if name in names]
        ordered.extend(sorted(names - set(ordered)))
        return ordered

    @classmethod
    def _has_level_metrics(cls, *groups: dict[str, float]) -> bool:
        return any(cls._metric_level(name)[0] is not None for group in groups for name in group)

    @classmethod
    def _metrics_by_level(cls, values: dict[str, float]) -> list[tuple[str, dict[str, float]]]:
        if not values:
            return []

        known_levels = {
            level for name in values for level, _ in [cls._metric_level(name)] if level is not None
        }

        by_level: dict[str, dict[str, float]] = {}
        for name, value in values.items():
            level, base_name = cls._metric_level(name)
            if level is None:
                level = cls._default_level(base_name, known_levels)
            by_level.setdefault(level, {})[base_name] = value

        return sorted(
            by_level.items(),
            key=lambda item: (item[0] == "summary", item[0]),
        )

    @classmethod
    def _metric_level(cls, name: str) -> tuple[str | None, str]:
        preferred = cls._preferred_metric_names()
        if name in preferred:
            return None, name

        for base_name in preferred:
            suffix = f"_{base_name}"
            if name.endswith(suffix) and name != base_name:
                return name[: -len(suffix)], base_name
        return None, name

    @staticmethod
    def _default_level(name: str, known_levels: set[str]) -> str:
        if name == "loss" and known_levels:
            return sorted(known_levels)[0]
        return "summary"

    @staticmethod
    def _preferred_metric_names() -> tuple[str, ...]:
        return (
            "loss",
            "auroc",
            "auprc",
            "balanced_accuracy",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "specificity",
            "mcc",
            "kappa",
        )

    @staticmethod
    def _render_table(header: list[str], rows: list[list[str]]) -> list[str]:
        widths = [max(len(row[index]) for row in [header] + rows) for index in range(len(header))]

        def render(row: list[str]) -> str:
            return " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))

        divider = "-+-".join("-" * width for width in widths)
        return [
            render(header),
            divider,
            *[render(row) for row in rows],
        ]

    @staticmethod
    def _with_epoch_separator(lines: list[str]) -> list[str]:
        width = max(80, *(len(line) for line in lines))
        separator = "=" * width
        return ["", separator, *lines, separator]

    @staticmethod
    def _runtime_stats(metrics: dict[str, float]) -> list[str]:
        stats: list[str] = []

        if "epoch_time_s" in metrics:
            stats.append(f"time={metrics['epoch_time_s']:.2f}s")
        if "gpu_mem_mb" in metrics:
            stats.append(f"gpu_peak={metrics['gpu_mem_mb']:.1f} MB")
        if "cpu_rss_mb" in metrics:
            stats.append(f"cpu_rss={metrics['cpu_rss_mb']:.1f} MB")

        return stats

    @staticmethod
    def _epoch_title(epoch: int, max_epochs: int | None, stats: list[str]) -> str:
        current = epoch + 1
        if max_epochs is None or max_epochs < 0:
            label = f"Epoch {current}"
        else:
            label = f"Epoch {current}/{max_epochs}"

        return f"{label} | {' | '.join(stats)}" if stats else label

    @staticmethod
    def _format_value(value: float | None) -> str:
        if value is None:
            return "-"
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"

        absolute = abs(value)
        if absolute >= 100:
            return f"{value:.2f}"
        if absolute >= 10:
            return f"{value:.3f}"
        return f"{value:.4f}"
