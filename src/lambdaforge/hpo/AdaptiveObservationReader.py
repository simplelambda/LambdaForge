"""Read an adaptive observation from canonical experiment artifacts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from lambdaforge.experiments.RunResult import RunResult
from lambdaforge.experiments.RunStatus import RunStatus
from lambdaforge.hpo.AdaptiveAction import AdaptiveAction
from lambdaforge.hpo.AdaptiveObservation import AdaptiveObservation
from lambdaforge.hpo.AdaptiveOptimizerConfig import AdaptiveOptimizerConfig
from lambdaforge.hpo.AdaptiveTrialStatus import AdaptiveTrialStatus


class AdaptiveObservationReader:
    """Convert result, dense curve and memory telemetry into controller evidence."""

    def read(
        self,
        action: AdaptiveAction,
        run_dir: str | Path,
        optimizer: AdaptiveOptimizerConfig,
        *,
        exit_code: int | None,
    ) -> AdaptiveObservation:
        """Read one terminal process outcome without selecting metrics by mtime."""
        directory = Path(run_dir)
        result = self._result(directory / "result.json")
        curve = self._curve(directory / "metrics.csv", optimizer.metric)
        budget = max((epoch for epoch, _ in curve), default=action.current_budget)
        score = curve[-1][1] if curve else self._result_score(result, optimizer.metric)
        resource = self._resource(directory / "adaptive-resource.json")
        error = (result.error if result is not None else "") or ""
        oom = "out of memory" in error.lower() or "cuda oom" in error.lower()
        if oom:
            status = AdaptiveTrialStatus.OOM_GPU
        elif exit_code is None or (result is not None and result.status is RunStatus.INTERRUPTED):
            status = AdaptiveTrialStatus.CANCELLED
        elif exit_code != 0 or result is None or result.status is not RunStatus.OK:
            status = AdaptiveTrialStatus.FAILED
        elif score is None:
            status = AdaptiveTrialStatus.FAILED
            error = (
                f"Objective metric {optimizer.metric!r} was not found in metrics.csv "
                "or terminal result metrics."
            )
        elif budget < action.target_budget:
            status = AdaptiveTrialStatus.EARLY_STOPPED
        elif budget < optimizer.max_budget:
            status = AdaptiveTrialStatus.PAUSED
        else:
            status = AdaptiveTrialStatus.COMPLETED
        seconds = float(result.seconds or 0.0) if result is not None else 0.0
        return AdaptiveObservation(
            action.action_id,
            action.config_id,
            action.parameters,
            action.seed,
            budget,
            score,
            curve,
            status,
            seconds=seconds,
            gpu_seconds=seconds if resource.get("peak_reserved_bytes", 0) else 0.0,
            peak_allocated_bytes=int(resource.get("peak_allocated_bytes", 0)),
            peak_reserved_bytes=int(resource.get("peak_reserved_bytes", 0)),
            oom=oom,
            run_dir=str(directory),
            error=error or None,
        )

    @staticmethod
    def _curve(path: Path, metric: str) -> tuple[tuple[int, float], ...]:
        if not path.exists():
            return ()
        output: list[tuple[int, float]] = []
        with open(path, encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                try:
                    point = (int(float(row["epoch"])), float(row[metric]))
                    if point[0] >= 0 and math.isfinite(point[1]):
                        output.append(point)
                except (KeyError, TypeError, ValueError):
                    continue
        return tuple(output)

    @staticmethod
    def _result(path: Path) -> RunResult | None:
        try:
            return RunResult.read_json(path)
        except (OSError, TypeError, ValueError):
            return None

    @staticmethod
    def _resource(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, TypeError, ValueError):
            return {}

    @staticmethod
    def _result_score(result: RunResult | None, metric: str) -> float | None:
        if result is None:
            return None
        for source in (result.final_metrics, result.best_metrics):
            if source is not None and metric in source:
                value = source[metric]
                if isinstance(value, dict):
                    value = value.get("value")
                try:
                    number = float(value)
                    return number if math.isfinite(number) else None
                except (TypeError, ValueError):
                    pass
        return None
