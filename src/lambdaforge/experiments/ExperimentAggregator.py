"""Cross-seed aggregation and sweep plots for experiment runs.

The experiment runner writes one directory per ``(variant, seed)`` with a
``result.json`` summary and a dense per-epoch ``metrics.csv``. This module is the
framework-level post-processor: it rebuilds the expected sweep from the YAML
config, reads every completed run from disk, and writes comparison artifacts that
do not depend on any concrete project.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lambdaforge.experiments.AggregateResult import AggregateResult
from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
from lambdaforge.experiments.RunStatus import RunStatus
from lambdaforge.experiments.statistics.paired.PairedAlternative import PairedAlternative
from lambdaforge.experiments.statistics.paired.PairedTestMethod import PairedTestMethod
from lambdaforge.experiments.statistics.StatisticalComparisonConfig import (
    StatisticalComparisonConfig,
)
from lambdaforge.experiments.statistics.StatisticalComparisonEngine import (
    StatisticalComparisonEngine,
)
from lambdaforge.experiments.VariantAggregateResult import VariantAggregateResult

AGGREGATE_SCHEMA_VERSION = 4


class ExperimentAggregator:
    """Generate cross-seed statistics, reliability reports and plots.

    Aggregation is deliberately state-free between calls; all source data is
    reconstructed from materialized configs and run artifacts on disk.
    """

    def _seed_segment(self, config: Mapping[str, Any]) -> str:
        seed = ExperimentConfig.get_value(config, "experiment.seed")
        return f"seed={seed}" if seed is not None else "seed=none"

    def _run_dir(self, config: Mapping[str, Any]) -> Path:
        variant = str(ExperimentConfig.get_value(config, "experiment.variant", "base"))
        return ExperimentConfig.suite_dir_for(config) / variant / self._seed_segment(config)

    def _maybe_float(self, value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if math.isfinite(out) else None

    def _agg(self, values: Sequence[float | None]) -> dict[str, Any]:
        numeric = [float(v) for v in values if v is not None and math.isfinite(float(v))]
        if not numeric:
            return {
                "mean": None,
                "std": None,
                "median": None,
                "min": None,
                "max": None,
                "n": 0,
                "values": [],
            }

        return {
            "mean": statistics.fmean(numeric),
            "std": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
            "median": statistics.median(numeric),
            "min": min(numeric),
            "max": max(numeric),
            "n": len(numeric),
            "values": numeric,
        }

    def _safe_mean(self, values: Sequence[float]) -> float | None:
        return statistics.fmean(values) if values else None

    def _safe_std(self, values: Sequence[float]) -> float | None:
        return statistics.stdev(values) if len(values) > 1 else 0.0 if values else None

    def _is_min_metric(self, metric: str, meta: Mapping[str, str]) -> bool:
        mode = str(meta.get("mode", "") or "").lower()
        if mode in {"min", "max"}:
            return mode == "min"

        raw = str(meta.get("raw_metric", "") or metric).lower()
        name = metric.lower()
        return any(
            token in raw or token in name for token in ("loss", "time", "seconds", "mem", "rss")
        )

    def _metric_mode(self, metric: str, meta: Mapping[str, str]) -> str:
        return "min" if self._is_min_metric(metric, meta) else "max"

    def _recommended_paired_n(
        self,
        improvements: Sequence[float],
        protocol: StatisticalComparisonConfig,
    ) -> tuple[int | None, str]:
        n = len(improvements)
        if n < 2:
            return None, "need_at_least_2_pairs"

        mean = statistics.fmean(improvements)
        if abs(mean) <= protocol.zero_tolerance:
            return None, "no_observed_effect"

        std = statistics.stdev(improvements)
        # If every observed paired difference is identical, the paired-mean formula
        # gives zero variance. Still require enough non-zero paired signs for
        # the configured one- or two-sided alpha.
        if std <= protocol.zero_tolerance:
            tail_alpha = (
                protocol.alpha / 2.0
                if protocol.paired_alternative
                in {PairedAlternative.TWO_SIDED, PairedAlternative.OBSERVED_DIRECTION}
                else protocol.alpha
            )
            sign_n = math.ceil(math.log(tail_alpha) / math.log(0.5))
            return max(n, sign_n), "zero_observed_variance"

        two_sided = protocol.paired_alternative in {
            PairedAlternative.TWO_SIDED,
            PairedAlternative.OBSERVED_DIRECTION,
        }
        alpha_probability = 1.0 - protocol.alpha / (2.0 if two_sided else 1.0)
        z_alpha = statistics.NormalDist().inv_cdf(alpha_probability)
        z_power = statistics.NormalDist().inv_cdf(protocol.target_power)
        required = math.ceil(((z_alpha + z_power) * std / abs(mean)) ** 2)
        return max(n, required), "paired_mean_power_approx"

    def _comparison_verdict(
        self,
        n_pairs: int,
        mean_improvement: float | None,
        p_value: float | None,
        recommended_n: int | None,
        protocol: StatisticalComparisonConfig,
    ) -> str:
        if n_pairs < protocol.min_pairs_for_verdict:
            return "insufficient_pairs"
        if mean_improvement is None or abs(mean_improvement) <= protocol.zero_tolerance:
            return "no_clear_effect"
        if p_value is not None and p_value <= protocol.alpha:
            if protocol.paired_alternative is PairedAlternative.GREATER:
                return "better_than_baseline" if mean_improvement > 0 else "inconclusive"
            if protocol.paired_alternative is PairedAlternative.LESS:
                return "worse_than_baseline" if mean_improvement < 0 else "inconclusive"
            return "better_than_baseline" if mean_improvement > 0 else "worse_than_baseline"
        if recommended_n is not None and recommended_n > n_pairs:
            return "needs_more_seeds"
        return "inconclusive"

    def _with_bh_q_values(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        out = [dict(row) for row in rows]
        indexed: list[tuple[int, float]] = []
        for idx, row in enumerate(out):
            p_value = self._maybe_float(row.get("p_value_directional"))
            if p_value is not None:
                indexed.append((idx, p_value))
        if not indexed:
            return out

        ordered = sorted(indexed, key=lambda item: item[1])
        m = len(ordered)
        adjusted: list[tuple[int, float]] = []
        running = 1.0
        for rank_from_end, (idx, p_value) in enumerate(reversed(ordered), start=1):
            rank = m - rank_from_end + 1
            q_value = min(running, p_value * m / rank)
            running = q_value
            adjusted.append((idx, min(1.0, q_value)))

        for idx, q_value in adjusted:
            out[idx]["q_value_bh_directional"] = q_value
        return out

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _read_epoch_stats(self, run_dir: Path) -> dict[str, float]:
        """Read timing/memory statistics from one run's dense ``metrics.csv``."""
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            return {}

        epoch_times: list[float] = []
        gpu_mems: list[float] = []
        cpu_rss: list[float] = []
        epochs: list[float] = []

        with open(metrics_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                epoch = self._maybe_float(row.get("epoch"))
                if epoch is not None:
                    epochs.append(epoch)

                time_cell = self._maybe_float(row.get("epoch_time_s"))
                if time_cell is not None:
                    epoch_times.append(time_cell)

                mem_cell = self._maybe_float(row.get("gpu_mem_mb"))
                if mem_cell is not None:
                    gpu_mems.append(mem_cell)

                cpu_cell = self._maybe_float(row.get("cpu_rss_mb"))
                if cpu_cell is not None:
                    cpu_rss.append(cpu_cell)

        stats: dict[str, float] = {}
        if epochs:
            stats["epoch_count"] = float(len(set(epochs)))
        if epoch_times:
            stats["epoch_time_s"] = statistics.fmean(epoch_times)
            stats["epoch_time_s_total"] = sum(epoch_times)
        if gpu_mems:
            stats["gpu_mem_mb"] = max(gpu_mems)
        if cpu_rss:
            stats["cpu_rss_mb"] = max(cpu_rss)
        return stats

    def _read_epoch_metric_rows(self, run_dir: Path) -> list[tuple[int, dict[str, float]]]:
        """Read all numeric per-epoch metrics from one valid run."""
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            return []

        rows: list[tuple[int, dict[str, float]]] = []
        with open(metrics_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                epoch = self._maybe_float(row.get("epoch"))
                if epoch is None:
                    continue

                metrics: dict[str, float] = {}
                for key, cell in row.items():
                    if key in {"epoch", "step", "hp_metric"} or str(key).endswith("_step"):
                        continue

                    value = self._maybe_float(cell)
                    if value is not None:
                        metrics[str(key)] = value

                if metrics:
                    rows.append((int(epoch), metrics))

        return rows

    def _metric_mode_from_result(self, result: Mapping[str, Any], monitor: str) -> str:
        best_epoch = result.get("best_epoch_metrics")
        if isinstance(best_epoch, Mapping) and best_epoch.get("mode") in {"min", "max"}:
            return str(best_epoch["mode"])

        best_metrics = result.get("best_metrics")
        if isinstance(best_metrics, Mapping):
            payload = best_metrics.get(monitor)
            if isinstance(payload, Mapping) and payload.get("mode") in {"min", "max"}:
                return str(payload["mode"])

        if monitor.endswith("_loss") or monitor in {
            "epoch_time_s",
            "gpu_mem_mb",
            "cpu_rss_mb",
            "seconds",
        }:
            return "min"
        return "max"

    def _best_epoch_from_metrics_csv(
        self, run_dir: Path, monitor: str, mode: str
    ) -> dict[str, Any]:
        metrics_path = run_dir / "metrics.csv"
        if monitor in {"", "None"} or not metrics_path.exists():
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

        if best_row is None or best_score is None:
            return {}

        metrics = {
            key: numeric
            for key, cell in best_row.items()
            if key != "epoch" and (numeric := self._maybe_float(cell)) is not None
        }
        epoch = self._maybe_float(best_row.get("epoch"))
        return {
            "monitor": monitor,
            "mode": mode,
            "epoch": int(epoch) if epoch is not None else -1,
            "best_score": best_score,
            "metrics": metrics,
            "source": "metrics.csv",
            "reconstructed": True,
        }

    def _has_numeric_metric_values(self, best_epoch: Mapping[str, Any]) -> bool:
        metrics = best_epoch.get("metrics")
        if not isinstance(metrics, Mapping):
            return False
        return any(self._maybe_float(value) is not None for value in metrics.values())

    def _normalise_best_epoch_metrics(
        self, result: Mapping[str, Any], run_dir: Path
    ) -> dict[str, Any]:
        best_epoch = result.get("best_epoch_metrics")
        if isinstance(best_epoch, Mapping):
            best_epoch_dict = dict(best_epoch)
            if self._maybe_float(
                best_epoch_dict.get("best_score")
            ) is not None and self._has_numeric_metric_values(best_epoch_dict):
                return best_epoch_dict
        else:
            best_epoch_dict = {}

        best_metric = result.get("best_metric")
        monitor = ""
        if isinstance(best_metric, Mapping):
            monitor = str(best_metric.get("monitor", "") or "")
        if not monitor:
            monitor = str(best_epoch_dict.get("monitor", "") or "")

        if monitor:
            rebuilt = self._best_epoch_from_metrics_csv(
                run_dir=run_dir,
                monitor=monitor,
                mode=self._metric_mode_from_result(result, monitor),
            )
            if rebuilt:
                return rebuilt

        return best_epoch_dict

    def _has_numeric_alternative_metrics(self, result: Mapping[str, Any]) -> bool:
        best_metrics = result.get("best_metrics")
        if isinstance(best_metrics, Mapping):
            for payload in best_metrics.values():
                if (
                    isinstance(payload, Mapping)
                    and self._maybe_float(payload.get("value")) is not None
                ):
                    return True

        final_metrics = result.get("final_metrics")
        if isinstance(final_metrics, Mapping):
            return any(self._maybe_float(value) is not None for value in final_metrics.values())

        return False

    def _metric_meta_for_best_payload(
        self, payload: Mapping[str, Any], name: str
    ) -> dict[str, str]:
        return {
            "source": "best_metrics",
            "raw_metric": name,
            "mode": str(payload.get("mode", "")),
            "monitor": "",
        }

    def _collect_seed_metrics(
        self, result: Mapping[str, Any], run_dir: Path
    ) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
        """Flatten one run's result into a single wide seed row."""
        row: dict[str, Any] = {
            "variant": result.get("variant", "base"),
            "seed": result.get("seed"),
            "status": result.get("status"),
            "run_dir": str(run_dir),
        }
        meta: dict[str, dict[str, str]] = {}

        best_metric = result.get("best_metric") or {}
        monitor = str(best_metric.get("monitor", "") or "")
        best_score = self._maybe_float(best_metric.get("best_score"))
        if best_score is not None:
            row["best_score"] = best_score
            meta["best_score"] = {
                "source": "best_metric",
                "raw_metric": monitor,
                "mode": "",
                "monitor": monitor,
            }

        best_epoch = self._normalise_best_epoch_metrics(result, run_dir)
        best_epoch_monitor = str(best_epoch.get("monitor", monitor) or "")
        best_epoch_mode = str(best_epoch.get("mode", "") or "")
        epoch = self._maybe_float(best_epoch.get("epoch"))
        if epoch is not None:
            row["best_epoch"] = epoch
            meta["best_epoch"] = {
                "source": "best_epoch_metrics",
                "raw_metric": "epoch",
                "mode": best_epoch_mode,
                "monitor": best_epoch_monitor,
            }

        epoch_score = self._maybe_float(best_epoch.get("best_score"))
        if epoch_score is not None:
            row["best_epoch_score"] = epoch_score
            meta["best_epoch_score"] = {
                "source": "best_epoch_metrics",
                "raw_metric": best_epoch_monitor,
                "mode": best_epoch_mode,
                "monitor": best_epoch_monitor,
            }

        for name, value in (best_epoch.get("metrics") or {}).items():
            numeric = self._maybe_float(value)
            if numeric is None:
                continue
            key = f"best_epoch_{name}"
            row[key] = numeric
            meta[key] = {
                "source": "best_epoch_metrics",
                "raw_metric": str(name),
                "mode": best_epoch_mode,
                "monitor": best_epoch_monitor,
            }

        for name, payload in (result.get("best_metrics") or {}).items():
            if not isinstance(payload, Mapping):
                continue
            numeric = self._maybe_float(payload.get("value"))
            if numeric is None:
                continue
            key = f"best_{name}"
            row[key] = numeric
            meta[key] = self._metric_meta_for_best_payload(payload, str(name))

        for name, value in (result.get("final_metrics") or {}).items():
            numeric = self._maybe_float(value)
            if numeric is None:
                continue
            key = f"final_{name}"
            row[key] = numeric
            meta[key] = {
                "source": "final_metrics",
                "raw_metric": str(name),
                "mode": "",
                "monitor": "",
            }

        seconds = self._maybe_float(result.get("seconds"))
        if seconds is not None:
            row["seconds"] = seconds
            meta["seconds"] = {
                "source": "runtime",
                "raw_metric": "seconds",
                "mode": "min",
                "monitor": "",
            }

        for key, value in self._read_epoch_stats(run_dir).items():
            row[key] = value
            meta[key] = {
                "source": "metrics.csv",
                "raw_metric": key,
                "mode": "min",
                "monitor": "",
            }

        return row, meta

    def _expected_runs(self, config: Mapping[str, Any]) -> list[dict[str, Any]]:
        expected: list[dict[str, Any]] = []
        for run_config in ExperimentConfig.expand_mapping(config):
            run_dir = self._run_dir(run_config)
            expected.append(
                {
                    "variant": str(
                        ExperimentConfig.get_value(run_config, "experiment.variant", "base")
                    ),
                    "seed": ExperimentConfig.get_value(run_config, "experiment.seed"),
                    "run_dir": str(run_dir),
                    "config": run_config,
                }
            )
        return expected

    def _result_lookup(self, results: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
        lookup: dict[str, Mapping[str, Any]] = {}
        for result in results:
            run_dir = result.get("run_dir")
            if run_dir not in (None, ""):
                lookup[str(Path(str(run_dir)))] = result
        return lookup

    def _load_run_result(
        self, expected: Mapping[str, Any], lookup: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, Any]:
        run_dir = Path(str(expected["run_dir"]))
        result = lookup.get(str(run_dir))
        if result is None:
            result = self._read_json(run_dir / "result.json")

        if result is None:
            return {
                "variant": expected["variant"],
                "seed": expected["seed"],
                "run_dir": str(run_dir),
                "status": "missing",
            }

        out = dict(result)
        out.setdefault("variant", expected["variant"])
        out.setdefault("seed", expected["seed"])
        out.setdefault("run_dir", str(run_dir))
        out.setdefault("status", "unknown")
        return out

    def _aggregate_state(self, result: Mapping[str, Any], run_dir: Path) -> tuple[str, str]:
        """Return ``(aggregate_status, reason)`` for one expected run.

        ``ok`` contributes to statistics. ``ignored`` is terminal but excluded
        (failed/OOM/corrupt partial result). ``pending`` means no result exists yet,
        so variant-level plots should wait because the combination is still running.
        """
        status = result.get("status")
        if status == "missing":
            return "pending", "missing_result"

        if status == RunStatus.FAILED.value:
            return "ignored", f"status={status}"

        if status != RunStatus.OK.value:
            return "pending", f"status={status or RunStatus.UNKNOWN.value}"

        if not run_dir.exists():
            return "ignored", "run_dir_missing"

        best_metric = result.get("best_metric") or {}
        monitor = best_metric.get("monitor")
        if monitor in (None, ""):
            # Generic experiments may not checkpoint on a named metric. In that
            # case a clean status is enough for runtime/final-metric aggregation.
            return "ok", ""

        best_epoch = self._normalise_best_epoch_metrics(result, run_dir)
        if not isinstance(best_epoch, Mapping) or not best_epoch:
            if self._has_numeric_alternative_metrics(result):
                return "ok", ""
            return "ignored", "missing_best_epoch_metrics"

        if self._maybe_float(best_epoch.get("best_score")) is None:
            if self._has_numeric_alternative_metrics(result):
                return "ok", ""
            return "ignored", "missing_best_epoch_score"

        metrics = best_epoch.get("metrics")
        if not isinstance(metrics, Mapping) or not metrics:
            if self._has_numeric_alternative_metrics(result):
                return "ok", ""
            return "ignored", "missing_best_epoch_metric_values"

        numeric_metrics = [self._maybe_float(value) for value in metrics.values()]
        if not any(value is not None for value in numeric_metrics):
            if self._has_numeric_alternative_metrics(result):
                return "ok", ""
            return "ignored", "no_numeric_best_epoch_metrics"

        return "ok", ""

    def _collect_rows(
        self,
        config: Mapping[str, Any],
        results: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, str]]]:
        lookup = self._result_lookup(results)
        run_status_rows: list[dict[str, Any]] = []
        seed_metric_rows: list[dict[str, Any]] = []
        metric_meta: dict[str, dict[str, str]] = {}

        for expected in self._expected_runs(config):
            run_dir = Path(str(expected["run_dir"]))
            result = self._load_run_result(expected, lookup)
            aggregate_status, aggregate_reason = self._aggregate_state(result, run_dir)
            run_status_rows.append(
                {
                    "variant": result.get("variant", expected["variant"]),
                    "seed": result.get("seed", expected["seed"]),
                    "status": result.get("status", "unknown"),
                    "aggregate_status": aggregate_status,
                    "ignore_reason": aggregate_reason,
                    "run_dir": str(run_dir),
                    "error": result.get("error", ""),
                }
            )

            if aggregate_status != "ok":
                continue

            row, row_meta = self._collect_seed_metrics(result, run_dir)
            seed_metric_rows.append(row)
            for key, value in row_meta.items():
                metric_meta.setdefault(key, value)

        return run_status_rows, seed_metric_rows, metric_meta

    def _metric_keys(self, rows: Sequence[Mapping[str, Any]]) -> list[str]:
        reserved = {"variant", "seed", "status", "run_dir", "error"}
        keys: set[str] = set()
        for row in rows:
            for key, value in row.items():
                if key in reserved:
                    continue
                if self._maybe_float(value) is not None:
                    keys.add(key)
        return sorted(keys)

    def _group_by_variant(
        self, rows: Sequence[Mapping[str, Any]]
    ) -> dict[str, list[Mapping[str, Any]]]:
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row.get("variant", "base")), []).append(row)
        return grouped

    def _seed_lookup(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
        lookup: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            seed = row.get("seed")
            if seed not in (None, ""):
                lookup[str(seed)] = row
        return lookup

    def _baseline_variant_for(self, variant: str, available_variants: set[str]) -> str | None:
        if variant == "base":
            return None

        if "__" in variant:
            candidate = variant.rsplit("__", 1)[0]
            if candidate in available_variants:
                return candidate

        if "base" in available_variants:
            return "base"

        return None

    def _primary_metric_key(
        self, metric_keys: Sequence[str], metric_meta: Mapping[str, Mapping[str, str]]
    ) -> str | None:
        coherent = [
            key
            for key in metric_keys
            if key.startswith("best_epoch_")
            and key not in {"best_epoch", "best_epoch_score"}
            and metric_meta.get(key, {}).get("monitor")
            and metric_meta.get(key, {}).get("raw_metric")
            == metric_meta.get(key, {}).get("monitor")
        ]
        if coherent:
            return sorted(coherent)[0]

        if "best_epoch_score" in metric_keys:
            return "best_epoch_score"

        best_epoch = [
            key for key in metric_keys if key.startswith("best_epoch_") and key != "best_epoch"
        ]
        if best_epoch:
            return sorted(best_epoch)[0]

        return metric_keys[0] if metric_keys else None

    def _baseline_comparison_rows(
        self,
        seed_rows: Sequence[Mapping[str, Any]],
        metric_keys: Sequence[str],
        metric_meta: Mapping[str, Mapping[str, str]],
        engine: StatisticalComparisonEngine,
        protocol: StatisticalComparisonConfig,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rows_by_variant = self._group_by_variant(seed_rows)
        available_variants = set(rows_by_variant)
        primary_metric = self._primary_metric_key(metric_keys, metric_meta)

        if not available_variants:
            return [], {
                "available": False,
                "reason": "no_completed_seeds",
                "primary_metric": primary_metric,
            }

        comparisons: list[dict[str, Any]] = []
        missing_baselines: list[str] = []

        for variant in sorted(available_variants):
            baseline_variant = self._baseline_variant_for(variant, available_variants)
            if baseline_variant is None:
                continue

            baseline_by_seed = self._seed_lookup(rows_by_variant.get(baseline_variant, []))
            variant_by_seed = self._seed_lookup(rows_by_variant.get(variant, []))
            common_seeds = sorted(set(baseline_by_seed) & set(variant_by_seed))
            if not common_seeds:
                missing_baselines.append(variant)
                continue

            for metric in metric_keys:
                meta = dict(metric_meta.get(metric, {}))
                mode = self._metric_mode(metric, meta)
                paired: list[tuple[str, float, float, float, float]] = []

                for seed in common_seeds:
                    baseline_value = self._maybe_float(baseline_by_seed[seed].get(metric))
                    variant_value = self._maybe_float(variant_by_seed[seed].get(metric))
                    if baseline_value is None or variant_value is None:
                        continue

                    delta = variant_value - baseline_value
                    improvement = -delta if mode == "min" else delta
                    paired.append((seed, baseline_value, variant_value, delta, improvement))

                if not paired:
                    continue

                improvements = [item[4] for item in paired]
                deltas = [item[3] for item in paired]
                baseline_values = [item[1] for item in paired]
                variant_values = [item[2] for item in paired]
                mean_improvement = self._safe_mean(improvements)
                std_improvement = self._safe_std(improvements)
                identity = (baseline_variant, variant, metric)
                legacy_interval = engine.legacy_ci95(improvements)
                interval = engine.confidence_interval(improvements, identity=identity)
                recommended_n, recommendation_reason = self._recommended_paired_n(
                    improvements,
                    protocol,
                )
                sign = engine.legacy_sign_test(improvements)
                paired_test = engine.paired_test(improvements)
                additional = (
                    max(0, int(recommended_n) - len(paired)) if recommended_n is not None else None
                )

                comparisons.append(
                    {
                        "variant": variant,
                        "baseline_variant": baseline_variant,
                        "metric": metric,
                        "primary_metric": metric == primary_metric,
                        "source": meta.get("source", ""),
                        "raw_metric": meta.get("raw_metric", ""),
                        "monitor": meta.get("monitor", ""),
                        "mode": mode,
                        "n_pairs": len(paired),
                        "paired_seeds": ",".join(seed for seed, *_ in paired),
                        "baseline_mean_paired": self._safe_mean(baseline_values),
                        "variant_mean_paired": self._safe_mean(variant_values),
                        "mean_delta": self._safe_mean(deltas),
                        "std_delta": self._safe_std(deltas),
                        "mean_improvement": mean_improvement,
                        "std_improvement": std_improvement,
                        "ci95_improvement_low": legacy_interval.lower,
                        "ci95_improvement_high": legacy_interval.upper,
                        "wins": sign.wins,
                        "losses": sign.losses,
                        "ties": sign.ties,
                        "p_value_sign_two_sided": sign.p_value_two_sided,
                        "p_value_sign_better": sign.p_value_better,
                        "p_value_sign_worse": sign.p_value_worse,
                        "p_value_directional": paired_test.p_value,
                        "effect_size_dz": (
                            mean_improvement / std_improvement
                            if mean_improvement is not None and std_improvement not in (None, 0.0)
                            else None
                        ),
                        "recommended_total_seeds": recommended_n,
                        "recommended_additional_seeds": additional,
                        "recommendation_reason": recommendation_reason,
                        "verdict": self._comparison_verdict(
                            n_pairs=len(paired),
                            mean_improvement=mean_improvement,
                            p_value=paired_test.p_value,
                            recommended_n=recommended_n,
                            protocol=protocol,
                        ),
                        "confidence_interval_method": interval.method,
                        "confidence_level": interval.confidence_level,
                        "confidence_interval_low": interval.lower,
                        "confidence_interval_high": interval.upper,
                        "confidence_interval_standard_error": interval.standard_error,
                        "confidence_interval_status": interval.status,
                        "confidence_interval_reason": interval.reason,
                        "bootstrap_resamples": interval.resamples,
                        "bootstrap_seed": interval.base_seed,
                        "bootstrap_effective_seed": interval.effective_seed,
                        "paired_test_method": paired_test.method,
                        "paired_test_alternative": paired_test.alternative,
                        "paired_test_calculation_requested": paired_test.calculation_requested,
                        "paired_test_calculation_used": paired_test.calculation_used,
                        "paired_test_statistic": paired_test.statistic,
                        "paired_test_positive_statistic": paired_test.positive_statistic,
                        "paired_test_negative_statistic": paired_test.negative_statistic,
                        "paired_test_effective_n": paired_test.n_effective,
                        "paired_test_zero_count": paired_test.n_zero,
                        "paired_test_has_rank_ties": paired_test.has_rank_ties,
                        "paired_test_z_statistic": paired_test.z_statistic,
                        "paired_test_p_value_two_sided": paired_test.p_value_two_sided,
                        "paired_test_p_value_better": paired_test.p_value_better,
                        "paired_test_p_value_worse": paired_test.p_value_worse,
                        "paired_test_status": paired_test.status,
                        "paired_test_reason": paired_test.reason,
                        "paired_test_zero_method": paired_test.zero_method,
                        "paired_test_continuity_correction": (paired_test.continuity_correction),
                        "paired_test_exact_max_pairs": paired_test.exact_max_pairs,
                        "paired_test_zero_tolerance": paired_test.zero_tolerance,
                        "paired_test_round_decimals": paired_test.round_decimals,
                        "p_value_wilcoxon_two_sided": (
                            paired_test.p_value_two_sided
                            if paired_test.method == PairedTestMethod.WILCOXON.value
                            else None
                        ),
                        "p_value_wilcoxon_better": (
                            paired_test.p_value_better
                            if paired_test.method == PairedTestMethod.WILCOXON.value
                            else None
                        ),
                        "p_value_wilcoxon_worse": (
                            paired_test.p_value_worse
                            if paired_test.method == PairedTestMethod.WILCOXON.value
                            else None
                        ),
                    }
                )

        comparisons = self._with_bh_q_values(comparisons)
        return comparisons, {
            "available": bool(comparisons),
            "reason": "" if comparisons else "baseline_missing_or_no_common_seeds",
            "primary_metric": primary_metric,
            "missing_baseline_or_pairs": missing_baselines,
        }

    def _aggregate_variant(
        self,
        variant: str,
        expected_rows: Sequence[Mapping[str, Any]],
        seed_rows: Sequence[Mapping[str, Any]],
        metric_keys: Sequence[str],
        metric_meta: Mapping[str, Mapping[str, str]],
    ) -> dict[str, Any]:
        completed_seeds = [row.get("seed") for row in seed_rows]
        failed_rows = [row for row in expected_rows if row.get("aggregate_status") == "ignored"]
        missing_rows = [row for row in expected_rows if row.get("status") == "missing"]
        pending_rows = [row for row in expected_rows if row.get("aggregate_status") == "pending"]

        metrics: dict[str, Any] = {}
        for key in metric_keys:
            values = [self._maybe_float(row.get(key)) for row in seed_rows]
            metrics[key] = {
                **dict(metric_meta.get(key, {})),
                **self._agg(values),
            }

        return {
            "aggregate_version": AGGREGATE_SCHEMA_VERSION,
            "variant": variant,
            "complete": len(seed_rows) == len(expected_rows) and len(expected_rows) > 0,
            "terminal": len(pending_rows) == 0 and len(expected_rows) > 0,
            "expected_n": len(expected_rows),
            "n_seeds": len(seed_rows),
            "seeds": completed_seeds,
            "missing_seeds": [row.get("seed") for row in missing_rows],
            "pending_seeds": [row.get("seed") for row in pending_rows],
            "failed_seeds": [
                row.get("seed") for row in failed_rows if row.get("status") != "missing"
            ],
            "ignored_seeds": [row.get("seed") for row in failed_rows],
            "ignored_runs": [
                {
                    "seed": row.get("seed"),
                    "status": row.get("status"),
                    "reason": row.get("ignore_reason", ""),
                    "error": row.get("error", ""),
                }
                for row in failed_rows
            ],
            "metrics": metrics,
        }

    def _epoch_curve_rows(
        self, variant: str, seed_rows: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        values: dict[str, dict[int, list[float]]] = {}

        for seed_row in seed_rows:
            run_dir = Path(str(seed_row["run_dir"]))
            for epoch, metrics in self._read_epoch_metric_rows(run_dir):
                for metric, value in metrics.items():
                    values.setdefault(metric, {}).setdefault(epoch, []).append(value)

        rows: list[dict[str, Any]] = []
        for metric in sorted(values):
            for epoch in sorted(values[metric]):
                stats = self._agg(values[metric][epoch])
                rows.append(
                    {
                        "variant": variant,
                        "epoch": epoch,
                        "metric": metric,
                        "mean": stats["mean"],
                        "std": stats["std"],
                        "median": stats["median"],
                        "min": stats["min"],
                        "max": stats["max"],
                        "n": stats["n"],
                    }
                )

        return rows

    def _write_epoch_curve_csvs(
        self, variant_dir: Path, rows: Sequence[Mapping[str, Any]]
    ) -> dict[str, str | None]:
        if not rows:
            return {"epoch_metrics_csv": None, "epoch_metrics_wide_csv": None}

        long_fields = [
            "variant",
            "epoch",
            "metric",
            "mean",
            "std",
            "median",
            "min",
            "max",
            "n",
        ]
        self._write_csv(variant_dir / "epoch_metrics.csv", rows, long_fields)

        metric_names = sorted({str(row["metric"]) for row in rows})
        epoch_values: set[float] = set()
        for row in rows:
            epoch_value = self._maybe_float(row["epoch"])
            if epoch_value is not None:
                epoch_values.add(epoch_value)
        epochs = sorted(epoch_values)
        lookup = {
            (int(float(row["epoch"])), str(row["metric"])): row
            for row in rows
            if self._maybe_float(row.get("epoch")) is not None
        }

        wide_fields = ["variant", "epoch"]
        for metric in metric_names:
            wide_fields.extend([f"{metric}__mean", f"{metric}__std", f"{metric}__n"])

        wide_rows: list[dict[str, Any]] = []
        variant = str(rows[0]["variant"])
        for epoch_value in epochs:
            epoch = int(epoch_value)
            wide_row: dict[str, Any] = {"variant": variant, "epoch": epoch}
            for metric in metric_names:
                stats = lookup.get((epoch, metric), {})
                wide_row[f"{metric}__mean"] = stats.get("mean", "")
                wide_row[f"{metric}__std"] = stats.get("std", "")
                wide_row[f"{metric}__n"] = stats.get("n", 0)
            wide_rows.append(wide_row)

        self._write_csv(variant_dir / "epoch_metrics_wide.csv", wide_rows, wide_fields)
        return {
            "epoch_metrics_csv": str(variant_dir / "epoch_metrics.csv"),
            "epoch_metrics_wide_csv": str(variant_dir / "epoch_metrics_wide.csv"),
        }

    def _read_existing_epoch_curve_rows(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []

        rows: list[dict[str, Any]] = []
        try:
            with open(path, encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    rows.append(dict(row))
        except OSError:
            return []
        return rows

    def _latest_variant_input_mtime(self, expected_rows: Sequence[Mapping[str, Any]]) -> float:
        latest = 0.0
        for row in expected_rows:
            run_dir = Path(str(row["run_dir"]))
            for path in (run_dir / "result.json", run_dir / "metrics.csv"):
                if path.exists():
                    latest = max(latest, path.stat().st_mtime)
        return latest

    def _plot_index_current(self, index_path: Path, latest_input_mtime: float) -> bool:
        if not index_path.exists() or index_path.stat().st_mtime < latest_input_mtime:
            return False

        index = self._read_json(index_path)
        if not index or index.get("error"):
            return False

        for plot_path in index.get("plots", []) or []:
            path = Path(str(plot_path))
            if not path.exists() or path.stat().st_mtime < latest_input_mtime:
                return False
        return True

    def _variant_artifacts_current(
        self,
        variant_dir: Path,
        expected_rows: Sequence[Mapping[str, Any]],
        aggregate: Mapping[str, Any],
        make_plots: bool,
        variant_plot_policy: str,
    ) -> bool:
        latest_input_mtime = self._latest_variant_input_mtime(expected_rows)
        required = [
            variant_dir / "aggregate.json",
            variant_dir / "epoch_metrics.csv",
            variant_dir / "epoch_metrics_wide.csv",
        ]
        for path in required:
            if not path.exists() or path.stat().st_mtime < latest_input_mtime:
                return False

        existing_aggregate = self._read_json(variant_dir / "aggregate.json")
        if (
            not existing_aggregate
            or existing_aggregate.get("aggregate_version") != AGGREGATE_SCHEMA_VERSION
        ):
            return False
        existing_state = {
            key: value for key, value in existing_aggregate.items() if key != "artifacts"
        }
        if existing_state != dict(aggregate):
            return False

        plots_required = (
            make_plots
            and variant_plot_policy != "none"
            and (variant_plot_policy == "available" or bool(aggregate.get("terminal", False)))
        )
        if plots_required:
            return self._plot_index_current(
                variant_dir / "plots" / "epoch_curves" / "index.json",
                latest_input_mtime,
            )

        return True

    def _global_plots_current(
        self, aggregate_dir: Path, status_rows: Sequence[Mapping[str, Any]]
    ) -> bool:
        latest_input_mtime = self._latest_variant_input_mtime(status_rows)
        return self._plot_index_current(
            aggregate_dir / "plots" / "summary" / "index.json", latest_input_mtime
        ) and self._plot_index_current(
            aggregate_dir / "plots" / "epoch_curves" / "index.json", latest_input_mtime
        )

    def _read_plot_index(self, index_path: Path) -> dict[str, Any]:
        index = self._read_json(index_path)
        return index if index is not None else {"plots": [], "error": None}

    def _write_csv(
        self, path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _write_seed_metrics_csv(
        self, aggregate_dir: Path, rows: Sequence[Mapping[str, Any]]
    ) -> None:
        fields = ["variant", "seed", "status", "run_dir"]
        for key in self._metric_keys(rows):
            if key not in fields:
                fields.append(key)
        self._write_csv(aggregate_dir / "seed_metrics.csv", rows, fields)

    def _summary_rows(self, aggregates: Mapping[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for variant in sorted(aggregates):
            aggregate = aggregates[variant]
            for metric_name, stats in sorted(aggregate["metrics"].items()):
                rows.append(
                    {
                        "variant": variant,
                        "metric": metric_name,
                        "source": stats.get("source", ""),
                        "raw_metric": stats.get("raw_metric", ""),
                        "monitor": stats.get("monitor", ""),
                        "mode": stats.get("mode", ""),
                        "mean": stats.get("mean", ""),
                        "std": stats.get("std", ""),
                        "median": stats.get("median", ""),
                        "min": stats.get("min", ""),
                        "max": stats.get("max", ""),
                        "n": stats.get("n", 0),
                        "expected_n": aggregate.get("expected_n", 0),
                        "complete": aggregate.get("complete", False),
                    }
                )
        return rows

    def _write_summary_csvs(
        self, base_dir: Path, aggregate_dir: Path, aggregates: Mapping[str, Any]
    ) -> None:
        fields = [
            "variant",
            "metric",
            "source",
            "raw_metric",
            "monitor",
            "mode",
            "mean",
            "std",
            "median",
            "min",
            "max",
            "n",
            "expected_n",
            "complete",
        ]
        rows = self._summary_rows(aggregates)

        # Keep the historical root-level path and also place all new post-processing
        # artifacts under aggregate/ for discoverability.
        self._write_csv(base_dir / "summary.csv", rows, fields)
        self._write_csv(aggregate_dir / "summary.csv", rows, fields)

    def _write_summary_wide_csv(self, aggregate_dir: Path, aggregates: Mapping[str, Any]) -> None:
        metric_names = sorted({metric for agg in aggregates.values() for metric in agg["metrics"]})
        fields = ["variant", "complete", "n_seeds", "expected_n"]
        for metric in metric_names:
            fields.extend([f"{metric}__mean", f"{metric}__std", f"{metric}__n"])

        rows: list[dict[str, Any]] = []
        for variant in sorted(aggregates):
            aggregate = aggregates[variant]
            row: dict[str, Any] = {
                "variant": variant,
                "complete": aggregate.get("complete", False),
                "n_seeds": aggregate.get("n_seeds", 0),
                "expected_n": aggregate.get("expected_n", 0),
            }
            for metric in metric_names:
                stats = aggregate["metrics"].get(metric, {})
                row[f"{metric}__mean"] = stats.get("mean", "")
                row[f"{metric}__std"] = stats.get("std", "")
                row[f"{metric}__n"] = stats.get("n", 0)
            rows.append(row)

        self._write_csv(aggregate_dir / "summary_wide.csv", rows, fields)

    def _write_baseline_comparison_csv(
        self, aggregate_dir: Path, rows: Sequence[Mapping[str, Any]]
    ) -> Path:
        fields = [
            "variant",
            "baseline_variant",
            "metric",
            "primary_metric",
            "source",
            "raw_metric",
            "monitor",
            "mode",
            "n_pairs",
            "paired_seeds",
            "baseline_mean_paired",
            "variant_mean_paired",
            "mean_delta",
            "std_delta",
            "mean_improvement",
            "std_improvement",
            "ci95_improvement_low",
            "ci95_improvement_high",
            "wins",
            "losses",
            "ties",
            "p_value_sign_two_sided",
            "p_value_sign_better",
            "p_value_sign_worse",
            "p_value_directional",
            "q_value_bh_directional",
            "effect_size_dz",
            "recommended_total_seeds",
            "recommended_additional_seeds",
            "recommendation_reason",
            "verdict",
            "confidence_interval_method",
            "confidence_level",
            "confidence_interval_low",
            "confidence_interval_high",
            "confidence_interval_standard_error",
            "confidence_interval_status",
            "confidence_interval_reason",
            "bootstrap_resamples",
            "bootstrap_seed",
            "bootstrap_effective_seed",
            "paired_test_method",
            "paired_test_alternative",
            "paired_test_calculation_requested",
            "paired_test_calculation_used",
            "paired_test_statistic",
            "paired_test_positive_statistic",
            "paired_test_negative_statistic",
            "paired_test_effective_n",
            "paired_test_zero_count",
            "paired_test_has_rank_ties",
            "paired_test_z_statistic",
            "paired_test_p_value_two_sided",
            "paired_test_p_value_better",
            "paired_test_p_value_worse",
            "paired_test_status",
            "paired_test_reason",
            "paired_test_zero_method",
            "paired_test_continuity_correction",
            "paired_test_exact_max_pairs",
            "paired_test_zero_tolerance",
            "paired_test_round_decimals",
            "p_value_wilcoxon_two_sided",
            "p_value_wilcoxon_better",
            "p_value_wilcoxon_worse",
        ]
        path = aggregate_dir / "baseline_comparisons.csv"
        self._write_csv(path, rows, fields)
        return path

    def _write_reliability_json(
        self,
        aggregate_dir: Path,
        rows: Sequence[Mapping[str, Any]],
        metadata: Mapping[str, Any],
        protocol: StatisticalComparisonConfig,
    ) -> Path:
        primary_metric = metadata.get("primary_metric")
        primary_rows = [
            dict(row)
            for row in rows
            if primary_metric is not None and row.get("metric") == primary_metric
        ]
        needs_more = [
            {
                "variant": row.get("variant"),
                "baseline_variant": row.get("baseline_variant"),
                "metric": row.get("metric"),
                "n_pairs": row.get("n_pairs"),
                "recommended_total_seeds": row.get("recommended_total_seeds"),
                "recommended_additional_seeds": row.get("recommended_additional_seeds"),
                "mean_improvement": row.get("mean_improvement"),
                "std_improvement": row.get("std_improvement"),
                "p_value_directional": row.get("p_value_directional"),
            }
            for row in rows
            if row.get("verdict") == "needs_more_seeds"
        ]

        payload = {
            "aggregate_version": AGGREGATE_SCHEMA_VERSION,
            "available": bool(rows),
            "reason": metadata.get("reason", ""),
            "baseline_rule": (
                "variant__ablation is compared with variant when that prefix exists; "
                "otherwise variants are compared with literal base when present"
            ),
            "alpha": protocol.alpha,
            "target_power": protocol.target_power,
            "statistical_protocol": protocol.to_dict(),
            "tests": {
                "paired_difference": "variant - baseline",
                "improvement": "delta for max metrics, -delta for min metrics",
                "confidence_interval": protocol.confidence_interval_method.value,
                "p_value": protocol.paired_test_method.value,
                "alternative": protocol.paired_alternative.value,
                "q_value": "Benjamini-Hochberg FDR over selected comparison p-values",
                "recommended_total_seeds": (
                    "paired normal-approx power estimate using observed std and effect; "
                    "zero-variance effects require enough signs for one-sided alpha"
                ),
            },
            "primary_metric": primary_metric,
            "primary_comparisons": primary_rows,
            "needs_more_seeds": needs_more,
            "comparisons": [dict(row) for row in rows],
            "missing_baseline_or_pairs": metadata.get("missing_baseline_or_pairs", []),
        }
        path = aggregate_dir / "reliability.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path

    def _sanitize_filename(self, value: str) -> str:
        out = "".join(ch if ch.isalnum() or ch in "._=-" else "_" for ch in value)
        return out.strip("._") or "metric"

    def _group_rows_by_metric(
        self, rows: Sequence[Mapping[str, Any]]
    ) -> dict[str, list[Mapping[str, Any]]]:
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            if self._maybe_float(row.get("mean")) is None:
                continue
            grouped.setdefault(str(row["metric"]), []).append(row)
        return grouped

    def _plot_summary_metric(
        self, metric: str, rows: Sequence[Mapping[str, Any]], path: Path
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        variants = [str(row["variant"]) for row in rows]
        means = [float(row["mean"]) for row in rows]
        stds = [float(row["std"]) if row.get("std") not in (None, "") else 0.0 for row in rows]

        height = max(3.2, 0.45 * len(variants) + 1.4)
        fig, ax = plt.subplots(figsize=(9.0, height))
        y_pos = list(range(len(variants)))
        ax.errorbar(
            means,
            y_pos,
            xerr=stds,
            fmt="o",
            color="#4C78A8",
            ecolor="#9ECAE9",
            capsize=4,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(variants)
        ax.invert_yaxis()
        ax.set_xlabel(metric)
        ax.set_title(metric)
        ax.grid(axis="x", alpha=0.25)

        for y, mean, std in zip(y_pos, means, stds, strict=True):
            ax.text(mean, y, f"  {mean:.4g} +/- {std:.3g}", va="center", fontsize=8)

        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)

    def _plot_epoch_curves(
        self,
        metric: str,
        rows_by_label: Mapping[str, Sequence[Mapping[str, Any]]],
        path: Path,
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        color_map = plt.get_cmap("tab20")
        colors = [color_map(index / 20.0) for index in range(20)]

        for index, (label, rows) in enumerate(sorted(rows_by_label.items())):
            ordered = sorted(rows, key=lambda row: int(float(row["epoch"])))
            epochs = [int(float(row["epoch"])) for row in ordered]
            means = [float(row["mean"]) for row in ordered]
            stds = [
                float(row["std"]) if row.get("std") not in (None, "") else 0.0 for row in ordered
            ]
            color = colors[index % len(colors)]

            ax.plot(epochs, means, label=label, color=color, linewidth=1.8)
            if any(std > 0 for std in stds):
                low = [mean - std for mean, std in zip(means, stds, strict=True)]
                high = [mean + std for mean, std in zip(means, stds, strict=True)]
                ax.fill_between(epochs, low, high, color=color, alpha=0.12, linewidth=0)

        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(alpha=0.25)
        if len(rows_by_label) <= 18:
            ax.legend(loc="best", fontsize=8)
        else:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)

        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)

    def _write_variant_epoch_plots(
        self, variant_dir: Path, rows: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        plots_dir = variant_dir / "plots" / "epoch_curves"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_index: dict[str, Any] = {"plots": [], "error": None}

        try:
            for metric, metric_rows in sorted(self._group_rows_by_metric(rows).items()):
                path = plots_dir / f"{self._sanitize_filename(metric)}.png"
                label = str(metric_rows[0]["variant"]) if metric_rows else "variant"
                self._plot_epoch_curves(metric, {label: metric_rows}, path)
                plot_index["plots"].append(str(path))
        except Exception as exc:  # pragma: no cover - depends on optional plotting stack.
            plot_index["error"] = f"{type(exc).__name__}: {exc}"

        with open(plots_dir / "index.json", "w", encoding="utf-8") as f:
            json.dump(plot_index, f, indent=2)
        return plot_index

    def _write_summary_plots(
        self, aggregate_dir: Path, summary_rows: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        plots_dir = aggregate_dir / "plots" / "summary"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_index: dict[str, Any] = {"plots": [], "error": None}

        try:
            for metric, rows in sorted(self._group_rows_by_metric(summary_rows).items()):
                path = plots_dir / f"{self._sanitize_filename(metric)}.png"
                self._plot_summary_metric(metric, rows, path)
                plot_index["plots"].append(str(path))
        except Exception as exc:  # pragma: no cover - depends on optional plotting stack.
            plot_index["error"] = f"{type(exc).__name__}: {exc}"

        with open(plots_dir / "index.json", "w", encoding="utf-8") as f:
            json.dump(plot_index, f, indent=2)
        return plot_index

    def _write_global_epoch_plots(
        self, aggregate_dir: Path, rows: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        plots_dir = aggregate_dir / "plots" / "epoch_curves"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_index: dict[str, Any] = {"plots": [], "error": None}

        try:
            by_metric: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
            for row in rows:
                if self._maybe_float(row.get("mean")) is None:
                    continue
                metric = str(row["metric"])
                variant = str(row["variant"])
                by_metric.setdefault(metric, {}).setdefault(variant, []).append(row)

            for metric, variant_rows in sorted(by_metric.items()):
                path = plots_dir / f"{self._sanitize_filename(metric)}.png"
                self._plot_epoch_curves(metric, variant_rows, path)
                plot_index["plots"].append(str(path))
        except Exception as exc:  # pragma: no cover - depends on optional plotting stack.
            plot_index["error"] = f"{type(exc).__name__}: {exc}"

        with open(plots_dir / "index.json", "w", encoding="utf-8") as f:
            json.dump(plot_index, f, indent=2)
        return plot_index

    def write(
        self,
        config: Mapping[str, Any],
        results: Sequence[Mapping[str, Any]] = (),
        make_plots: bool = True,
        global_plots: bool = True,
        variant_plot_policy: str = "available",
        *,
        final: bool = True,
    ) -> AggregateResult:
        """Serialize aggregate publication and optionally finalize retention."""
        from lambdaforge.experiments.retention.AggregationReceipt import AggregationReceipt
        from lambdaforge.experiments.retention.ArtifactRetentionManager import (
            ArtifactRetentionManager,
        )
        from lambdaforge.experiments.retention.ArtifactRetentionMode import (
            ArtifactRetentionMode,
        )
        from lambdaforge.experiments.retention.ArtifactRetentionPolicy import (
            ArtifactRetentionPolicy,
        )

        normalized = config if isinstance(config, ExperimentConfig) else ExperimentConfig(config)
        policy = ArtifactRetentionPolicy.from_config(normalized)
        manager = ArtifactRetentionManager()
        if final:
            with manager.activity_lock(normalized, policy, shared=True):
                with manager.aggregation_lock(normalized, policy):
                    manager.invalidate_receipt(normalized)
                    aggregate = self._write_unlocked(
                        normalized,
                        results,
                        make_plots=make_plots,
                        global_plots=global_plots,
                        variant_plot_policy=variant_plot_policy,
                    )
                    receipt = AggregationReceipt.build(normalized)
                    receipt.write_json(AggregationReceipt.path_for(normalized))
                    if policy.mode is ArtifactRetentionMode.PREVIEW:
                        try:
                            print(
                                manager.preview(normalized).summary(),
                                file=sys.stderr,
                                flush=True,
                            )
                        except Exception as error:
                            print(
                                "WARNING: artifact-retention preview failed: "
                                f"{error.__class__.__name__}: {error}",
                                file=sys.stderr,
                                flush=True,
                            )
        else:
            with manager.aggregation_lock(normalized, policy):
                manager.invalidate_receipt(normalized)
                aggregate = self._write_unlocked(
                    normalized,
                    results,
                    make_plots=make_plots,
                    global_plots=global_plots,
                    variant_plot_policy=variant_plot_policy,
                )

        if final and policy.mode is ArtifactRetentionMode.APPLY:
            retention = manager.apply(normalized, explicit=False)
            if retention.status.value not in {"applied", "already_applied"}:
                print(
                    "WARNING: artifact retention did not commit: "
                    f"{retention.status.value}; {', '.join(retention.errors)}",
                    flush=True,
                )
        return aggregate

    def _write_unlocked(
        self,
        config: Mapping[str, Any],
        results: Sequence[Mapping[str, Any]] = (),
        make_plots: bool = True,
        global_plots: bool = True,
        variant_plot_policy: str = "available",
    ) -> AggregateResult:
        """Write cross-seed aggregates for every sweep variant.

        The function intentionally reconstructs the sweep from ``config`` and reads
        ``result.json`` from disk. The optional ``results`` list from the current
        runner is only a fresh cache. This means aggregation also works after
        resuming an experiment or when it is run as a standalone post-processing
        step over already completed seeds.
        """
        if variant_plot_policy not in {"available", "terminal", "none"}:
            raise ValueError("variant_plot_policy must be 'available', 'terminal' or 'none'.")

        config = config if isinstance(config, ExperimentConfig) else ExperimentConfig(config)
        statistical_protocol = StatisticalComparisonConfig.from_mapping(config)
        statistical_engine = StatisticalComparisonEngine(statistical_protocol)
        base_dir = ExperimentConfig.suite_dir_for(config)
        aggregate_dir = base_dir / "aggregate"

        status_rows, seed_rows, metric_meta = self._collect_rows(config, results)
        expected_by_variant = self._group_by_variant(status_rows)
        seeds_by_variant = self._group_by_variant(seed_rows)
        metric_keys = self._metric_keys(seed_rows)
        all_epoch_rows: list[dict[str, Any]] = []

        aggregates: dict[str, Any] = {}
        for variant in sorted(expected_by_variant):
            variant_seed_rows = seeds_by_variant.get(variant, [])
            aggregate = self._aggregate_variant(
                variant,
                expected_by_variant[variant],
                variant_seed_rows,
                metric_keys,
                metric_meta,
            )

            variant_dir = base_dir / variant
            variant_dir.mkdir(parents=True, exist_ok=True)

            if self._variant_artifacts_current(
                variant_dir,
                expected_by_variant[variant],
                aggregate,
                make_plots,
                variant_plot_policy,
            ):
                existing_aggregate = self._read_json(variant_dir / "aggregate.json")
                if existing_aggregate is not None:
                    aggregate = existing_aggregate
                    aggregate.setdefault("artifacts", {})["skipped_existing"] = True
                all_epoch_rows.extend(
                    self._read_existing_epoch_curve_rows(variant_dir / "epoch_metrics.csv")
                )
                aggregates[variant] = aggregate
                continue

            epoch_rows = self._epoch_curve_rows(variant, variant_seed_rows)
            all_epoch_rows.extend(epoch_rows)
            epoch_artifacts: dict[str, Any] = self._write_epoch_curve_csvs(variant_dir, epoch_rows)

            write_variant_plots = (
                make_plots
                and bool(epoch_rows)
                and variant_plot_policy != "none"
                and (variant_plot_policy == "available" or bool(aggregate.get("terminal", False)))
            )
            if write_variant_plots:
                epoch_artifacts["epoch_plots"] = self._write_variant_epoch_plots(
                    variant_dir, epoch_rows
                )
            else:
                epoch_artifacts["epoch_plots"] = {
                    "plots": [],
                    "error": None,
                    "skipped": (
                        "no_valid_epoch_metrics"
                        if not epoch_rows
                        else "variant_not_terminal"
                        if variant_plot_policy == "terminal"
                        and not aggregate.get("terminal", False)
                        else "plots_disabled"
                    ),
                }

            aggregate["artifacts"] = epoch_artifacts
            aggregates[variant] = aggregate

            VariantAggregateResult.from_mapping(aggregate).write_json(
                variant_dir / "aggregate.json"
            )

        aggregate_dir.mkdir(parents=True, exist_ok=True)
        self._write_csv(
            aggregate_dir / "run_status.csv",
            status_rows,
            [
                "variant",
                "seed",
                "status",
                "aggregate_status",
                "ignore_reason",
                "run_dir",
                "error",
            ],
        )
        self._write_seed_metrics_csv(aggregate_dir, seed_rows)
        self._write_summary_csvs(base_dir, aggregate_dir, aggregates)
        self._write_summary_wide_csv(aggregate_dir, aggregates)
        baseline_rows, reliability_meta = self._baseline_comparison_rows(
            seed_rows,
            metric_keys,
            metric_meta,
            statistical_engine,
            statistical_protocol,
        )
        baseline_csv = self._write_baseline_comparison_csv(aggregate_dir, baseline_rows)
        reliability_json = self._write_reliability_json(
            aggregate_dir,
            baseline_rows,
            reliability_meta,
            statistical_protocol,
        )

        from lambdaforge.experiments.retention.ArtifactRetentionPolicy import (
            ArtifactRetentionPolicy,
        )

        retention_policy = ArtifactRetentionPolicy.from_config(config)
        summary = {
            "aggregate_version": AGGREGATE_SCHEMA_VERSION,
            "experiment": str(ExperimentConfig.get_value(config, "experiment.name", "experiment")),
            "base_dir": str(base_dir),
            "aggregate_dir": str(aggregate_dir),
            "expected_runs": len(status_rows),
            "completed_runs": sum(1 for row in status_rows if row.get("aggregate_status") == "ok"),
            "ignored_runs": sum(
                1 for row in status_rows if row.get("aggregate_status") == "ignored"
            ),
            "variants": aggregates,
            "artifacts": {
                "run_status_csv": str(aggregate_dir / "run_status.csv"),
                "seed_metrics_csv": str(aggregate_dir / "seed_metrics.csv"),
                "summary_csv": str(aggregate_dir / "summary.csv"),
                "summary_wide_csv": str(aggregate_dir / "summary_wide.csv"),
                "baseline_comparisons_csv": str(baseline_csv),
                "reliability_json": str(reliability_json),
                "legacy_summary_csv": str(base_dir / "summary.csv"),
            },
            "reliability": {
                "available": reliability_meta.get("available", False),
                "reason": reliability_meta.get("reason", ""),
                "primary_metric": reliability_meta.get("primary_metric"),
                "statistical_protocol": statistical_protocol.to_dict(),
                "n_comparisons": len(baseline_rows),
                "n_primary_comparisons": sum(
                    1 for row in baseline_rows if row.get("primary_metric") is True
                ),
            },
            "plots": None,
            "retention": {
                "mode": retention_policy.mode.value,
                "status": "not_applied",
                "latest_manifest": None,
            },
        }

        rows = self._summary_rows(aggregates)
        if make_plots and global_plots:
            if self._global_plots_current(aggregate_dir, status_rows):
                summary_plots = self._read_plot_index(
                    aggregate_dir / "plots" / "summary" / "index.json"
                )
                epoch_plots = self._read_plot_index(
                    aggregate_dir / "plots" / "epoch_curves" / "index.json"
                )
                summary_plots["skipped_existing"] = True
                epoch_plots["skipped_existing"] = True
                summary["plots"] = {
                    "summary": summary_plots,
                    "epoch_curves": epoch_plots,
                }
            else:
                summary["plots"] = {
                    "summary": self._write_summary_plots(aggregate_dir, rows),
                    "epoch_curves": self._write_global_epoch_plots(aggregate_dir, all_epoch_rows),
                }

        result = AggregateResult(
            aggregates,
            summary={key: value for key, value in summary.items() if key != "variants"},
        )
        result.write_summary_json(aggregate_dir / "summary.json")
        return result
