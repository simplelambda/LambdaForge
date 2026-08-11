"""Scientific plot planning and stateless file rendering."""

from __future__ import annotations

import importlib
import json
import math
import os
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
from lambdaforge.results.ResultService import ResultService
from lambdaforge.visualization.PlotSpec import PlotSpec


class VisualizationService:
    """Build testable plot specifications and render outside training loops."""

    def __init__(self, results: ResultService | None = None) -> None:
        self.results = results or ResultService()

    def learning(
        self,
        selector: str | Path,
        *,
        metrics: Sequence[str] = (),
        aggregate: str = "mean",
        uncertainty: str = "std",
    ) -> PlotSpec:
        """Build individual or cross-seed learning curves from dense epoch CSVs."""
        if aggregate not in {"individual", "mean"}:
            raise ValueError("Learning aggregation must be individual or mean.")
        if uncertainty not in {"none", "std", "ci"}:
            raise ValueError("Learning uncertainty must be none, std or ci.")
        raw = self.results.metric_series(selector)
        selected_metrics = tuple(metrics) or self._default_learning_metrics(raw.metrics())
        series = raw.select(*selected_metrics)
        rows: list[dict[str, Any]] = []
        if aggregate == "individual":
            for point in series.points:
                rows.append(
                    {
                        **point.to_dict(),
                        "group": f"{point.variant}/seed={point.seed}",
                        "n": 1,
                        "lower": None,
                        "upper": None,
                    }
                )
        else:
            grouped: dict[tuple[str, str, float], list[float]] = defaultdict(list)
            for point in series.points:
                grouped[(point.metric, point.variant, point.step)].append(point.value)
            critical = statistics.NormalDist().inv_cdf(0.975)
            for (metric, variant, step), values in sorted(grouped.items()):
                mean = statistics.fmean(values)
                spread = statistics.stdev(values) if len(values) > 1 else None
                half = None
                if spread is not None and uncertainty == "std":
                    half = spread
                elif spread is not None and uncertainty == "ci":
                    half = critical * spread / math.sqrt(len(values))
                rows.append(
                    {
                        "metric": metric,
                        "variant": variant,
                        "step": step,
                        "value": mean,
                        "group": variant,
                        "n": len(values),
                        "lower": mean - half if half is not None else None,
                        "upper": mean + half if half is not None else None,
                    }
                )
        return PlotSpec(
            "learning",
            data_references=tuple(sorted({point.run for point in series.points})),
            x="step",
            y="value",
            metrics=selected_metrics,
            aggregation=aggregate,
            uncertainty=uncertainty,
            labels={"x": "Epoch / step", "y": "Metric value"},
            data=tuple(rows),
            metadata={"selector": str(selector), "n=1_has_uncertainty": False},
        )

    def sweep(
        self,
        config_path: str | Path,
        *,
        x: str,
        metrics: Sequence[str],
        y: str | None = None,
        uncertainty: str = "std",
        interpolate: bool = False,
        view: str = "auto",
        normalize: bool = False,
    ) -> PlotSpec:
        """Aggregate final metrics over seeds for one- or two-dimensional sweeps."""
        if not metrics:
            raise ValueError("Sweep plots require at least one --metric.")
        if uncertainty not in {"none", "std", "ci"}:
            raise ValueError("Sweep uncertainty must be none, std or ci.")
        allowed_views = (
            {"auto", "line"}
            if y is None
            else {
                "auto",
                "scatter",
                "heatmap",
                "contour",
                "surface",
            }
        )
        if view not in allowed_views:
            raise ValueError(f"Sweep view must be one of {sorted(allowed_views)}.")
        config = ExperimentConfig.from_yaml(config_path)
        records = self.results.resolve(config_path)
        lookup = {
            (record.result.variant or "base", record.result.seed): record for record in records
        }
        grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        raw_values: dict[tuple[str, str], tuple[Any, Any]] = {}
        for run in config.expand():
            variant = str(ExperimentConfig.get_value(run, "experiment.variant", "base"))
            seed = ExperimentConfig.get_value(run, "experiment.seed")
            record = lookup.get((variant, seed))
            if record is None or record.status != "ok":
                continue
            x_value = ExperimentConfig.get_value(run, x)
            y_value = ExperimentConfig.get_value(run, y) if y is not None else None
            cell = (json.dumps(x_value, sort_keys=True), json.dumps(y_value, sort_keys=True))
            raw_values[cell] = (x_value, y_value)
            for metric in metrics:
                if metric not in record.metrics:
                    continue
                grouped[(cell[0], cell[1], metric)].append(float(record.metrics[metric]))
        available = sorted({key[2] for key in grouped})
        missing = [metric for metric in metrics if metric not in available]
        if missing:
            raise KeyError(f"Metric {missing[0]!r} was not found. Available metrics: {available}.")
        critical = statistics.NormalDist().inv_cdf(0.975)
        rows: list[dict[str, Any]] = []
        for (x_key, y_key, metric), values in sorted(grouped.items()):
            mean = statistics.fmean(values)
            spread = statistics.stdev(values) if len(values) > 1 else None
            half = None
            if spread is not None and uncertainty == "std":
                half = spread
            elif spread is not None and uncertainty == "ci":
                half = critical * spread / math.sqrt(len(values))
            x_value, y_value = raw_values[(x_key, y_key)]
            rows.append(
                {
                    "x": x_value,
                    "y": y_value,
                    "metric": metric,
                    "value": mean,
                    "std": spread,
                    "lower": mean - half if half is not None else None,
                    "upper": mean + half if half is not None else None,
                    "n": len(values),
                }
            )
        if normalize:
            for metric in metrics:
                metric_rows = [row for row in rows if row["metric"] == metric]
                observed = [float(row["value"]) for row in metric_rows]
                low, high = min(observed), max(observed)
                span = high - low
                for row in metric_rows:
                    row["raw_value"] = row["value"]
                    row["raw_lower"] = row["lower"]
                    row["raw_upper"] = row["upper"]
                    for field in ("value", "lower", "upper", "std"):
                        value = row[field]
                        if value is None:
                            continue
                        row[field] = (
                            float(value) / span
                            if field == "std" and span
                            else (float(value) - low) / span
                            if span
                            else 0.0
                        )
        return PlotSpec(
            "sweep-2d" if y else "sweep-1d",
            data_references=tuple(record.attempt_id for record in records),
            x=x,
            y=y,
            metrics=tuple(metrics),
            aggregation="mean",
            uncertainty=uncertainty,
            labels={"x": x, "y": y or ("normalized metric [0,1]" if normalize else "metric")},
            data=tuple(rows),
            metadata={
                "config": str(Path(config_path).resolve()),
                "interpolate": bool(interpolate),
                "view": ("line" if y is None else "scatter") if view == "auto" else view,
                "interpolation_method": "matplotlib triangulation" if interpolate else None,
                "missing_cells_are_masked": not interpolate,
                "normalized": normalize,
                "normalization_method": "per-metric min-max over observed cells"
                if normalize
                else None,
            },
        )

    def seed_distribution(
        self, selector: str | Path, *, metric: str, kind: str = "box"
    ) -> PlotSpec:
        """Build a box/violin/strip view of terminal values across seeds."""
        if kind not in {"box", "violin", "strip"}:
            raise ValueError("Seed distribution kind must be box, violin or strip.")
        records = self.results.resolve(selector)
        available = sorted({key for record in records for key in record.metrics})
        if metric not in available:
            raise KeyError(f"Metric {metric!r} was not found. Available metrics: {available}.")
        rows = tuple(
            {
                "variant": record.result.variant or "base",
                "seed": record.result.seed,
                "value": float(record.metrics[metric]),
            }
            for record in records
            if metric in record.metrics
        )
        return PlotSpec(
            f"seed-{kind}",
            data_references=tuple(record.attempt_id for record in records),
            x="variant",
            y="value",
            metrics=(metric,),
            data=rows,
            metadata={"selector": str(selector)},
        )

    def hpo(
        self,
        study: str | Path,
        *,
        parameter: str | None = None,
        direction: str = "minimize",
    ) -> PlotSpec:
        """Visualize objective, best-so-far, budget, state and an optional parameter."""
        if direction not in {"minimize", "maximize"}:
            raise ValueError("HPO plot direction must be minimize or maximize.")
        root = Path(study)
        state_path = root if root.name == "state.json" else root / "state.json"
        if not state_path.is_file():
            candidates = tuple(sorted(root.rglob(".lambdaforge/adaptive/*/state.json")))
            if len(candidates) != 1:
                raise LookupError(
                    f"Expected one adaptive state below {root}, found {len(candidates)}."
                )
            state_path = candidates[0]
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        observations = payload.get("observations", ())
        if not isinstance(observations, list):
            raise TypeError("Adaptive state observations must be a list.")
        rows: list[dict[str, Any]] = []
        best: float | None = None
        for trial, observation in enumerate(observations, start=1):
            if not isinstance(observation, dict):
                continue
            score = observation.get("score")
            numeric = float(score) if score is not None else None
            if numeric is not None:
                best = (
                    numeric
                    if best is None
                    else min(best, numeric)
                    if direction == "minimize"
                    else max(best, numeric)
                )
            parameters = observation.get("parameters", {})
            rows.append(
                {
                    "trial": trial,
                    "objective": numeric,
                    "best_so_far": best,
                    "budget": int(observation.get("budget", 0)),
                    "cumulative_gpu_seconds": sum(
                        float(item.get("gpu_seconds", 0.0))
                        for item in observations[:trial]
                        if isinstance(item, dict)
                    ),
                    "status": observation.get("status"),
                    "parameter": parameters.get(parameter)
                    if parameter is not None and isinstance(parameters, dict)
                    else None,
                }
            )
        return PlotSpec(
            "hpo",
            data_references=(str(state_path.resolve()),),
            x="trial",
            y="objective",
            metrics=("objective", "best_so_far", "budget"),
            data=tuple(rows),
            metadata={
                "parameter": parameter,
                "phase": payload.get("phase"),
                "direction": direction,
            },
        )

    def resources(self, selector: str | Path) -> PlotSpec:
        """Build curves only from resource metrics already present in dense logs."""
        series = self.results.metric_series(selector)
        keywords = ("cpu", "rss", "ram", "gpu", "vram", "memory", "seconds", "runtime")
        metrics = tuple(
            metric
            for metric in series.metrics()
            if any(word in metric.lower() for word in keywords)
        )
        if not metrics:
            raise KeyError(
                "No resource metric was recorded. Available metrics: "
                f"{', '.join(series.metrics())}."
            )
        spec = self.learning(selector, metrics=metrics, aggregate="mean", uncertainty="none")
        return PlotSpec(
            "resources",
            spec.data_references,
            spec.x,
            spec.y,
            spec.metrics,
            spec.aggregation,
            spec.uncertainty,
            spec.labels,
            spec.data,
            spec.metadata,
        )

    def render(self, spec: PlotSpec, output: str | Path) -> Path:
        """Atomically render PNG/SVG/PDF or optional self-contained Plotly HTML."""
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        sidecar = destination.with_suffix(destination.suffix + ".plot.json")
        if destination.is_file() and sidecar.is_file():
            try:
                previous = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = {}
            if previous.get("fingerprint") == spec.fingerprint:
                return destination
        if destination.suffix.lower() == ".html":
            self._render_plotly(spec, destination)
        elif destination.suffix.lower() in {".png", ".svg", ".pdf"}:
            self._render_matplotlib(spec, destination)
        else:
            raise ValueError("Plot output must end in .png, .svg, .pdf or .html.")
        self._atomic_text(
            sidecar,
            json.dumps(
                {
                    "fingerprint": spec.fingerprint,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "spec": spec.to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        return destination

    @staticmethod
    def _default_learning_metrics(available: Sequence[str]) -> tuple[str, ...]:
        preferred = tuple(metric for metric in ("train_loss", "val_loss") if metric in available)
        return preferred or tuple(available[:2])

    def _render_matplotlib(self, spec: PlotSpec, destination: Path) -> None:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        pyplot = importlib.import_module("matplotlib.pyplot")
        metrics = spec.metrics or ("value",)
        figure, axes = pyplot.subplots(
            len(metrics), 1, squeeze=False, figsize=(7, 4 * len(metrics))
        )
        for index, metric in enumerate(metrics):
            axis = axes[index][0]
            rows = [row for row in spec.data if row.get("metric", metric) == metric]
            if spec.plot_type in {"learning", "resources"}:
                groups = sorted({str(row.get("group", "series")) for row in rows})
                for group in groups:
                    selected = sorted(
                        (row for row in rows if str(row.get("group", "series")) == group),
                        key=lambda row: float(row["step"]),
                    )
                    axis.plot(
                        [row["step"] for row in selected],
                        [row["value"] for row in selected],
                        label=f"{group} ({metric})",
                    )
                    bounded = [row for row in selected if row.get("lower") is not None]
                    if bounded:
                        axis.fill_between(
                            [row["step"] for row in bounded],
                            [row["lower"] for row in bounded],
                            [row["upper"] for row in bounded],
                            alpha=0.2,
                        )
            elif spec.plot_type == "sweep-1d":
                ordered = sorted(rows, key=lambda row: row["x"])
                axis.errorbar(
                    [row["x"] for row in ordered],
                    [row["value"] for row in ordered],
                    yerr=[
                        0.0
                        if row.get("lower") is None
                        else float(row["value"]) - float(row["lower"])
                        for row in ordered
                    ],
                    marker="o",
                )
            elif spec.plot_type == "sweep-2d":
                axis = self._render_sweep_2d(figure, axis, rows, spec)
            elif spec.plot_type.startswith("seed-"):
                grouped = defaultdict(list)
                for row in rows:
                    grouped[str(row["variant"])].append(float(row["value"]))
                labels = sorted(grouped)
                values = [grouped[label] for label in labels]
                if spec.plot_type == "seed-violin":
                    axis.violinplot(values, showmeans=True)
                    axis.set_xticks(range(1, len(labels) + 1), labels)
                elif spec.plot_type == "seed-strip":
                    for position, group_values in enumerate(values, start=1):
                        axis.scatter([position] * len(group_values), group_values)
                    axis.set_xticks(range(1, len(labels) + 1), labels)
                else:
                    axis.boxplot(values, tick_labels=labels)
            elif spec.plot_type == "hpo":
                for field in ("objective", "best_so_far", "budget"):
                    selected = [row for row in rows if row.get(field) is not None]
                    axis.plot(
                        [row["trial"] for row in selected],
                        [row[field] for row in selected],
                        marker="o",
                        label=field,
                    )
            elif spec.plot_type in {"point-cloud", "graph", "mesh"}:
                axis.remove()
                axis = figure.add_subplot(len(metrics), 1, index + 1, projection="3d")
                vertices = [row for row in rows if row.get("kind") in {None, "node", "vertex"}]
                axis.scatter(
                    [row["x"] for row in vertices],
                    [row["y"] for row in vertices],
                    [row["z"] for row in vertices],
                    s=4,
                )
                indexed = {int(row["index"]): row for row in vertices if "index" in row}
                for edge in (row for row in rows if row.get("kind") == "edge"):
                    left, right = indexed.get(int(edge["source"])), indexed.get(int(edge["target"]))
                    if left is not None and right is not None:
                        axis.plot(
                            [left["x"], right["x"]],
                            [left["y"], right["y"]],
                            [left["z"], right["z"]],
                            linewidth=0.5,
                        )
            axis.set_title(metric)
            axis.set_xlabel(spec.labels.get("x", spec.x or ""))
            axis.set_ylabel(spec.labels.get("y", spec.y or "value"))
            if axis.get_legend_handles_labels()[0]:
                axis.legend()
        figure.tight_layout()
        temporary = destination.with_name(
            f".{destination.stem}.{os.getpid()}.{uuid4().hex}{destination.suffix}"
        )
        try:
            figure.savefig(temporary)
            os.replace(temporary, destination)
        finally:
            pyplot.close(figure)
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _render_sweep_2d(
        figure: Any, axis: Any, rows: list[Mapping[str, Any]], spec: PlotSpec
    ) -> Any:
        """Render exact cells as scatter/masked grid, interpolating only when requested."""
        view = str(spec.metadata.get("view", "scatter"))
        if view == "scatter":
            plotted = axis.scatter(
                [row["x"] for row in rows],
                [row["y"] for row in rows],
                c=[row["value"] for row in rows],
            )
            figure.colorbar(plotted, ax=axis)
            return axis
        if view == "surface":
            axis.remove()
            axis = figure.add_subplot(1, 1, 1, projection="3d")
            if bool(spec.metadata.get("interpolate")):
                axis.plot_trisurf(
                    [float(row["x"]) for row in rows],
                    [float(row["y"]) for row in rows],
                    [float(row["value"]) for row in rows],
                )
            else:
                axis.scatter(
                    [row["x"] for row in rows],
                    [row["y"] for row in rows],
                    [row["value"] for row in rows],
                )
            return axis
        numpy = importlib.import_module("numpy")
        x_values = sorted({row["x"] for row in rows}, key=str)
        y_values = sorted({row["y"] for row in rows}, key=str)
        matrix = numpy.full((len(y_values), len(x_values)), numpy.nan)
        for row in rows:
            matrix[y_values.index(row["y"]), x_values.index(row["x"])] = row["value"]
        if view == "contour":
            if numpy.isnan(matrix).any() and not bool(spec.metadata.get("interpolate")):
                raise ValueError("Contour over an incomplete grid requires --interpolate.")
            plotted = axis.contourf(range(len(x_values)), range(len(y_values)), matrix)
        else:
            plotted = axis.imshow(matrix, aspect="auto", origin="lower")
        axis.set_xticks(range(len(x_values)), [str(value) for value in x_values])
        axis.set_yticks(range(len(y_values)), [str(value) for value in y_values])
        figure.colorbar(plotted, ax=axis)
        return axis

    @staticmethod
    def _render_plotly(spec: PlotSpec, destination: Path) -> None:
        try:
            graph_objects = importlib.import_module("plotly.graph_objects")
        except ImportError as error:
            raise ImportError("HTML/interactive plots require lambdaforge[viz].") from error
        figure = graph_objects.Figure()
        if spec.plot_type == "sweep-2d":
            rows = list(spec.data)
            view = str(spec.metadata.get("view", "scatter"))
            if view in {"surface", "heatmap", "contour"}:
                x_values = sorted({row["x"] for row in rows}, key=str)
                y_values = sorted({row["y"] for row in rows}, key=str)
                cells = {(row["x"], row["y"]): row["value"] for row in rows}
                z_values = [
                    [cells.get((x_value, y_value)) for x_value in x_values] for y_value in y_values
                ]
                provider = (
                    graph_objects.Surface
                    if view == "surface"
                    else graph_objects.Contour
                    if view == "contour"
                    else graph_objects.Heatmap
                )
                figure.add_trace(provider(x=x_values, y=y_values, z=z_values))
            else:
                figure.add_trace(
                    graph_objects.Scatter3d(
                        x=[row["x"] for row in rows],
                        y=[row["y"] for row in rows],
                        z=[row["value"] for row in rows],
                        mode="markers",
                    )
                )
        elif spec.plot_type in {"point-cloud", "graph", "mesh"}:
            rows = [row for row in spec.data if row.get("kind") in {None, "node", "vertex"}]
            figure.add_trace(
                graph_objects.Scatter3d(
                    x=[row["x"] for row in rows],
                    y=[row["y"] for row in rows],
                    z=[row["z"] for row in rows],
                    mode="markers",
                )
            )
        else:
            rows = list(spec.data)
            figure.add_trace(
                graph_objects.Scatter(
                    x=[row.get("step", row.get("x")) for row in rows],
                    y=[row["value"] for row in rows],
                    mode="lines+markers",
                )
            )
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        figure.write_html(temporary, include_plotlyjs=True, full_html=True)
        os.replace(temporary, destination)

    @staticmethod
    def _atomic_text(path: Path, value: str) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(value, encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
