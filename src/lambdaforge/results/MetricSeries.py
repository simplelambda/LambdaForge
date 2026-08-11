"""Standard metric-series representation backed by existing dense CSV files."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from lambdaforge.experiments.results.ResultRecord import ResultRecord
from lambdaforge.results.MetricPoint import MetricPoint


@dataclass(frozen=True, slots=True)
class MetricSeries:
    """Collect normalized points without duplicating the authoritative metric log."""

    points: tuple[MetricPoint, ...]

    @classmethod
    def from_records(cls, records: Iterable[ResultRecord]) -> MetricSeries:
        """Read canonical dense CSV rows for selected result records."""
        points: list[MetricPoint] = []
        for record in records:
            metrics_path = Path(record.run_dir) / "metrics.csv"
            if not metrics_path.is_file() or record.archived:
                continue
            try:
                with metrics_path.open(encoding="utf-8", newline="") as handle:
                    rows = tuple(csv.DictReader(handle))
            except (OSError, csv.Error):
                continue
            for index, row in enumerate(rows, start=1):
                step = cls._number(row.get("epoch"))
                if step is None:
                    step = cls._number(row.get("step"))
                step = float(index) if step is None else step
                timestamp = row.get("timestamp") or row.get("timestamp_utc")
                for metric, raw in row.items():
                    if metric in {"epoch", "step", "timestamp", "timestamp_utc", "hp_metric"}:
                        continue
                    if metric.endswith("_step"):
                        continue
                    value = cls._number(raw)
                    if value is None:
                        continue
                    points.append(
                        MetricPoint(
                            run=record.attempt_id,
                            seed=(
                                int(record.result.seed) if record.result.seed is not None else None
                            ),
                            variant=record.result.variant or "base",
                            split=cls._split(metric),
                            metric=metric,
                            step=step,
                            value=value,
                            timestamp=timestamp,
                        )
                    )
        return cls(tuple(points))

    def metrics(self) -> tuple[str, ...]:
        """Return available metric names in stable order."""
        return tuple(sorted({point.metric for point in self.points}))

    def select(self, *metrics: str) -> MetricSeries:
        """Filter exact names and report useful alternatives on failure."""
        if not metrics:
            return self
        available = self.metrics()
        missing = tuple(metric for metric in metrics if metric not in available)
        if missing:
            raise KeyError(
                f"Metric {missing[0]!r} was not found. Available metrics: {', '.join(available)}"
            )
        selected = set(metrics)
        return MetricSeries(tuple(point for point in self.points if point.metric in selected))

    def to_rows(self) -> tuple[dict[str, object], ...]:
        """Return JSON/CSV/DataFrame-friendly rows."""
        return tuple(cast(dict[str, object], point.to_dict()) for point in self.points)

    @staticmethod
    def _number(value: object) -> float | None:
        if value in (None, ""):
            return None
        try:
            result = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    @staticmethod
    def _split(metric: str) -> str | None:
        prefix = metric.split("_", 1)[0]
        return prefix if prefix in {"train", "val", "test", "predict"} else None
