"""Bridge Lightning validation metrics and cooperative adaptive-HPO stops."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

from lambdaforge.integrations.Lightning import CallbackBase


class AdaptiveHpoCallback(CallbackBase):
    """Observe one objective after validation and honor a file-based stop request."""

    def __init__(
        self,
        metric: str | None,
        metrics_path: Path | None,
        stop_path: Path | None,
        training_metrics_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.metric = metric
        self.metrics_path = metrics_path
        self.stop_path = stop_path
        self.training_metrics_path = training_metrics_path
        self._validation_started: float | None = None
        self._last_training_step: int | None = None

    @classmethod
    def from_environment(cls) -> AdaptiveHpoCallback | None:
        metric = os.environ.get("LAMBDAFORGE_HPO_OBJECTIVE")
        metrics = os.environ.get("LAMBDAFORGE_HPO_METRICS_PATH")
        stop = os.environ.get("LAMBDAFORGE_STOP_REQUEST_PATH")
        training = os.environ.get("LAMBDAFORGE_TRAINING_METRICS_PATH")
        adaptive = bool(metric and metrics and stop)
        if not adaptive and not training:
            return None
        return cls(
            metric if adaptive else None,
            Path(metrics) if adaptive and metrics else None,
            Path(stop) if adaptive and stop else None,
            Path(training) if training else None,
        )

    def on_train_batch_start(self, trainer: Any, *_: Any) -> None:
        self._stop(trainer)

    def on_validation_batch_start(self, trainer: Any, *_: Any) -> None:
        self._stop(trainer)

    def on_validation_epoch_start(self, trainer: Any, *_: Any) -> None:
        if not bool(getattr(trainer, "sanity_checking", False)):
            self._validation_started = time.perf_counter()

    def on_validation_epoch_end(self, trainer: Any, *_: Any) -> None:
        if bool(getattr(trainer, "sanity_checking", False)):
            return
        step = int(getattr(trainer, "current_epoch", 0)) + 1
        scalars = self._scalars(getattr(trainer, "callback_metrics", {}))
        if self._validation_started is not None:
            scalars["validation_time_s"] = time.perf_counter() - self._validation_started
        if bool(getattr(trainer, "is_global_zero", True)):
            self._write_training(scalars, step)
            if self.metric is not None and self.metrics_path is not None and self.metric in scalars:
                self._append(self.metrics_path, self.metric, scalars[self.metric], step)
        self._stop(trainer)

    def on_fit_end(self, trainer: Any, *_: Any) -> None:
        if not bool(getattr(trainer, "is_global_zero", True)):
            return
        step = int(getattr(trainer, "current_epoch", 0)) + 1
        if step != self._last_training_step:
            self._write_training(self._scalars(getattr(trainer, "callback_metrics", {})), step)

    def _stop(self, trainer: Any) -> None:
        if self.stop_path is not None and self.stop_path.is_file():
            trainer.should_stop = True

    def _write_training(self, values: dict[str, float], step: int) -> None:
        if self.training_metrics_path is None or not values:
            return
        self.training_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.training_metrics_path.open("a", encoding="utf-8") as handle:
            for name, value in values.items():
                record = {"name": name, "value": value, "step": step, "split": None}
                handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._last_training_step = step

    @staticmethod
    def _scalars(values: Any) -> dict[str, float]:
        if not isinstance(values, dict):
            return {}
        output: dict[str, float] = {}
        for raw_name, value in values.items():
            name = str(raw_name)
            if name in {"epoch", "step", "hp_metric"} or name.endswith("_step"):
                continue
            try:
                scalar = float(value.detach().cpu().item() if hasattr(value, "detach") else value)
            except (TypeError, ValueError, RuntimeError):
                continue
            if math.isfinite(scalar):
                output[name] = scalar
        return output

    @staticmethod
    def _append(path: Path, name: str, value: float, step: int, *, sync: bool = True) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"name": name, "value": value, "step": step, "split": None}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            if sync:
                os.fsync(handle.fileno())


__all__ = ["AdaptiveHpoCallback"]
