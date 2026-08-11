"""Stable service for human-facing result queries, series, comparison and export."""

from __future__ import annotations

import csv
import importlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml

from lambdaforge.experiments.results.ResultCatalog import ResultCatalog
from lambdaforge.experiments.results.ResultRecord import ResultRecord
from lambdaforge.registry.ExperimentComparator import ExperimentComparator
from lambdaforge.results.MetricSeries import MetricSeries
from lambdaforge.results.ResultSelectionError import ResultSelectionError


class ResultService:
    """Locate and analyse local/synchronized results without a second database."""

    def __init__(self, roots: Sequence[str | Path] = ("runs",)) -> None:
        self.roots = tuple(Path(root).resolve() for root in roots)

    def records(self, *, status: str | None = None) -> tuple[ResultRecord, ...]:
        """Return a de-duplicated deterministic view across configured roots."""
        records: dict[tuple[str, str], ResultRecord] = {}
        for root in self.roots:
            for record in ResultCatalog(root).records(status=status):
                records[(record.result_path, record.attempt_id)] = record
        return tuple(
            sorted(records.values(), key=lambda record: (record.result.name, record.attempt_id))
        )

    def resolve(self, selector: str | Path) -> tuple[ResultRecord, ...]:
        """Resolve a path, attempt/job-like ID, fingerprint, run name or experiment name."""
        path = Path(selector)
        if path.exists():
            if path.suffix.lower() in {".yaml", ".yml"}:
                from lambdaforge.configuration.AuthoringConfig import AuthoringConfig
                from lambdaforge.configuration.ConfigurationKind import ConfigurationKind
                from lambdaforge.experiments.Experiment import Experiment
                from lambdaforge.tasks.TaskRun import TaskRun

                materialized = AuthoringConfig.from_yaml(path).materialize()
                catalog = (
                    TaskRun.from_yaml(path).result_catalog()
                    if materialized.kind is ConfigurationKind.TASK
                    else Experiment.from_yaml(path).result_catalog()
                )
            else:
                catalog = ResultCatalog(path)
            matches = catalog.records()
        else:
            raw = str(selector)
            if raw.startswith("job-"):
                from lambdaforge.results.RemoteResultService import RemoteResultService

                synchronized = RemoteResultService().sync(raw)
                matches = ResultCatalog(synchronized.destination).records()
                if matches:
                    return matches
            matches = tuple(
                record
                for record in self.records()
                if raw
                in {
                    record.attempt_id,
                    record.config_fingerprint,
                    record.result.name,
                    record.result.variant,
                    Path(record.run_dir).name,
                }
                or record.result.name.startswith(f"{raw}__")
            )
        if not matches:
            available = tuple(sorted({record.result.name for record in self.records()}))
            raise ResultSelectionError(
                f"No result matches {str(selector)!r}. Available names: {available}."
            )
        return tuple(matches)

    def show(self, selector: str | Path) -> dict[str, Any]:
        """Return all candidates explicitly; never hide an ambiguous selector."""
        matches = self.resolve(selector)
        return {
            "selector": str(selector),
            "ambiguous": len(matches) > 1,
            "records": [record.to_dict() for record in matches],
            "available_metrics": list(MetricSeries.from_records(matches).metrics()),
        }

    def metric_series(self, selector: str | Path, *metrics: str) -> MetricSeries:
        """Return normalized points for a selected result group."""
        return MetricSeries.from_records(self.resolve(selector)).select(*metrics)

    def compare(
        self,
        selectors: Sequence[str | Path],
        *,
        metrics: Sequence[str] = (),
        confidence_level: float = 0.95,
        direction: str | None = None,
    ) -> dict[str, Any]:
        """Compare one or more exact metrics across human-selected groups."""
        if not selectors:
            raise ValueError("Result comparison requires at least one selector.")
        if direction not in {None, "minimize", "maximize"}:
            raise ValueError("Comparison direction must be minimize, maximize or omitted.")
        groups = {
            str(selector): self._registry_rows(self.resolve(selector)) for selector in selectors
        }
        available = [
            set(key for record in records for key in record.get("metrics", {}))
            for records in groups.values()
        ]
        selected = tuple(metrics) or tuple(
            sorted(set.intersection(*available) if available else ())
        )
        if not selected:
            raise ValueError("No common numeric result metric is available for comparison.")
        comparisons = [
            ExperimentComparator().compare(groups, metric=metric, confidence_level=confidence_level)
            for metric in selected
        ]
        for comparison in comparisons:
            table = comparison["groups"]
            baseline = str(selectors[0])
            baseline_mean = float(table[baseline]["mean"])
            for summary in table.values():
                summary["delta_vs_baseline"] = float(summary["mean"]) - baseline_mean
            comparison["baseline"] = baseline
            comparison["direction"] = direction
            if direction is not None:
                ordered = sorted(
                    table,
                    key=lambda label: float(table[label]["mean"]),
                    reverse=direction == "maximize",
                )
                comparison["best_group"] = ordered[0]
                comparison["worst_group"] = ordered[-1]
        return {
            "selectors": [str(value) for value in selectors],
            "comparisons": comparisons,
        }

    def export(
        self,
        selector: str | Path,
        destination: str | Path,
        *,
        metric_series: bool = False,
    ) -> Path:
        """Export selected summaries or normalized metric rows as JSON/CSV/Parquet."""
        path = Path(destination)
        rows: Iterable[dict[str, Any]] = (
            self.metric_series(selector).to_rows()
            if metric_series
            else (record.to_dict() for record in self.resolve(selector))
        )
        materialized = tuple(dict(row) for row in rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".json":
            path.write_text(
                json.dumps(materialized, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        elif path.suffix.lower() == ".csv":
            fields = tuple(sorted({key for row in materialized for key in row}))
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for row in materialized:
                    writer.writerow(
                        {
                            key: json.dumps(value, sort_keys=True)
                            if isinstance(value, (dict, list, tuple))
                            else value
                            for key, value in row.items()
                        }
                    )
        elif path.suffix.lower() == ".parquet":
            try:
                pandas = importlib.import_module("pandas")
            except ImportError as error:
                raise ImportError("Parquet export requires lambdaforge[parquet].") from error
            pandas.DataFrame(materialized).to_parquet(path, index=False)
        else:
            raise ValueError("Result export must end in .json, .csv or .parquet.")
        return path

    @staticmethod
    def _registry_rows(records: Sequence[ResultRecord]) -> tuple[dict[str, Any], ...]:
        rows = []
        for record in records:
            config_path = Path(record.run_dir) / "config.yaml"
            try:
                value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError):
                value = {}
            rows.append(
                {
                    **record.to_dict(),
                    "config": dict(value) if isinstance(value, dict) else {},
                }
            )
        return tuple(rows)
