"""Result, plot and scientific-artifact CLI commands."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

from lambdaforge.artifacts.ArtifactPluginRegistry import ArtifactPluginRegistry
from lambdaforge.artifacts.ArtifactService import ArtifactService
from lambdaforge.artifacts.NumpyArtifactValidator import NumpyArtifactValidator
from lambdaforge.artifacts.RemoteArtifactService import RemoteArtifactService
from lambdaforge.experiments.Experiment import Experiment
from lambdaforge.experiments.results.ResultCatalog import ResultCatalog
from lambdaforge.results.RemoteResultService import RemoteResultService
from lambdaforge.results.ResultService import ResultService
from lambdaforge.tasks.TaskConfig import TaskConfig
from lambdaforge.tasks.TaskRun import TaskRun
from lambdaforge.visualization.VisualizationService import VisualizationService


def run_evidence_command(arguments: argparse.Namespace) -> int:
    """Dispatch result, plot and artifact actions to their domain services."""
    if arguments.command == "results":
        return run_results(arguments)
    if arguments.command == "plot":
        return run_plot(arguments)
    return run_artifact(arguments)


def run_results(arguments: argparse.Namespace) -> int:
    """Execute the parsed ``results`` action."""
    if arguments.results_command == "list":
        records = ResultService(arguments.root or ("runs",)).records(status=arguments.status)
        result_payload = [record.to_dict() for record in records]
        if arguments.json:
            print(json.dumps(result_payload, indent=2))
        else:
            for result_record in records:
                print(
                    f"{result_record.attempt_id:<32} {result_record.status:<11} "
                    f"{result_record.result.name} seed={result_record.result.seed}"
                )
        return 0
    if arguments.results_command == "show":
        shown = ResultService(arguments.root or ("runs",)).show(arguments.selector)
        if arguments.json:
            print(json.dumps(shown, indent=2))
        else:
            print(f"Selector: {shown['selector']}  ambiguous={shown['ambiguous']}")
            for record in shown["records"]:
                metrics = ", ".join(
                    f"{key}={value}" for key, value in record.get("metrics", {}).items()
                )
                print(
                    f"{record['attempt_id']:<32} {record['status']:<11} "
                    f"{record['name']} seed={record.get('seed')}  {metrics}"
                )
        return 0
    if arguments.results_command == "compare":
        compared = ResultService(arguments.root or ("runs",)).compare(
            arguments.selectors,
            metrics=arguments.metric,
            confidence_level=arguments.confidence,
            direction=arguments.direction,
        )
        if arguments.json:
            print(json.dumps(compared, indent=2))
        else:
            for comparison in compared["comparisons"]:
                print(f"Metric: {comparison['metric']}")
                print("GROUP  N  MEAN  STD  CI  DELTA VS BASELINE")
                for label, summary in comparison["groups"].items():
                    interval = summary["confidence_interval"]
                    print(
                        f"{label}  {summary['count']}  {summary['mean']:.6g}  "
                        f"{summary['stdev']:.6g}  [{interval[0]:.6g}, "
                        f"{interval[1]:.6g}]  {summary['delta_vs_baseline']:+.6g}"
                    )
                if comparison.get("direction"):
                    print(f"Best: {comparison['best_group']}  Worst: {comparison['worst_group']}")
        return 0
    if arguments.results_command == "export":
        suffix = arguments.format
        destination = arguments.output or Path(
            f"{str(arguments.selector).replace('/', '-')}-results.{suffix}"
        )
        print(
            ResultService(arguments.root or ("runs",)).export(
                arguments.selector,
                destination,
                metric_series=arguments.series,
            )
        )
        return 0
    if arguments.results_command == "sync":
        synced = RemoteResultService(root=arguments.destination).sync(arguments.job_id)
        print(json.dumps(synced.to_dict(), indent=2))
        return 0
    source = arguments.source
    result_catalog = (
        TaskRun.from_yaml(source).result_catalog()
        if source.suffix.lower() in {".yaml", ".yml"} and TaskConfig.is_task_file(source)
        else Experiment.from_yaml(source).result_catalog()
        if source.suffix.lower() in {".yaml", ".yml"}
        else ResultCatalog(source)
    )
    result_records = result_catalog.records(
        status=arguments.status,
        include_archived=not arguments.no_archived,
    )
    if arguments.duplicates:
        duplicate_ids = {
            record.attempt_id
            for group in result_catalog.duplicate_groups().values()
            for record in group
        }
        result_records = tuple(
            record for record in result_records if record.attempt_id in duplicate_ids
        )
    index_path = result_catalog.write_index() if arguments.write_index else None
    if arguments.json:
        print(
            json.dumps(
                {
                    "summary": result_catalog.summary(),
                    "index_path": str(index_path) if index_path else None,
                    "records": [record.to_dict() for record in result_records],
                },
                indent=2,
            )
        )
    else:
        print(result_catalog.summary())
        for result_record in result_records:
            location = "archived" if result_record.archived else "current"
            print(
                f"{result_record.attempt_id}  {result_record.status:<11} "
                f"{location:<8} {result_record.result_path}"
            )
        if index_path is not None:
            print(f"Wrote result index: {index_path}")
    return 2 if arguments.fail_on_ambiguous and result_catalog.ambiguous_successes() else 0


def run_plot(arguments: argparse.Namespace) -> int:
    """Execute the parsed ``plot`` action."""
    visualization = VisualizationService()
    if arguments.plot_command == "learning":
        selector: str | Path = arguments.selector
        if str(selector).startswith("job-"):
            selector = Path(RemoteResultService().sync(str(selector)).destination)
        spec = visualization.learning(
            selector,
            metrics=arguments.metric,
            aggregate=arguments.aggregate,
            uncertainty=arguments.uncertainty,
        )
    elif arguments.plot_command == "sweep":
        spec = visualization.sweep(
            arguments.config,
            x=arguments.x,
            y=arguments.y,
            metrics=arguments.metric,
            uncertainty=arguments.uncertainty,
            interpolate=arguments.interpolate,
            view=arguments.kind,
            normalize=arguments.normalize,
        )
    elif arguments.plot_command == "seeds":
        spec = visualization.seed_distribution(
            arguments.selector, metric=arguments.metric, kind=arguments.kind
        )
    elif arguments.plot_command == "hpo":
        spec = visualization.hpo(
            arguments.study,
            parameter=arguments.parameter,
            direction=arguments.direction,
        )
    else:
        spec = visualization.resources(arguments.selector)
    if arguments.json:
        print(json.dumps(spec.to_dict(), indent=2))
        return 0
    output = arguments.output or Path(f"{spec.plot_type}.png")
    if getattr(arguments, "follow", False):
        return follow_plot(arguments, visualization, output)
    print(visualization.render(spec, output))
    return 0


def run_artifact(arguments: argparse.Namespace) -> int:
    """Execute the parsed ``artifact`` action."""
    artifact_service = ArtifactService()
    if arguments.artifact_command == "inspect":
        artifact_inspection = artifact_service.inspect(
            arguments.path,
            array=arguments.array,
            rows=arguments.rows,
            slice_expression=arguments.slice,
            inspector=arguments.inspector,
        )
        if arguments.json:
            print(json.dumps(artifact_inspection.to_dict(), indent=2))
        else:
            print(
                f"Type: {artifact_inspection.artifact_type}\n"
                f"Path: {artifact_inspection.path}\n"
                f"Size: {artifact_inspection.size_bytes} bytes"
            )
            for item in artifact_inspection.items:
                if "name" in item:
                    print(
                        f"{item['name']:<24} shape={tuple(item['shape'])!s:<18} "
                        f"dtype={item['dtype']:<12} elements={item['elements']}"
                    )
                else:
                    print(json.dumps(dict(item), indent=2))
    elif arguments.artifact_command == "export":
        destination = arguments.output or arguments.path.with_suffix(f".{arguments.format}")
        print(
            artifact_service.export_array(
                arguments.path, array=arguments.array, destination=destination
            )
        )
    elif arguments.artifact_command == "visualize":
        roles = {
            key: value
            for key, value in {
                "positions": arguments.positions,
                "nodes": arguments.nodes,
                "edges": arguments.edges,
            }.items()
            if value is not None
        }
        spec = artifact_service.visualization_spec(
            arguments.path,
            visualization_type=arguments.type,
            roles=roles,
            visualizer=arguments.visualizer,
        )
        if arguments.json:
            print(json.dumps(spec.to_dict(), indent=2))
        else:
            print(VisualizationService().render(spec, arguments.output))
    elif arguments.artifact_command == "list":
        artifact_payload = (
            RemoteArtifactService().list(arguments.selector)
            if str(arguments.selector).startswith("job-")
            else artifact_service.list(arguments.selector)
        )
        if arguments.json:
            print(json.dumps(artifact_payload, indent=2))
        else:
            for artifact in artifact_payload:
                print(
                    f"{str(artifact.get('logical_name')):<32} "
                    f"{str(artifact.get('type', 'artifact')):<14} "
                    f"{str(artifact.get('size_bytes', '?')):<12} "
                    f"{artifact.get('location', artifact.get('path', ''))}"
                )
    elif arguments.artifact_command == "fetch":
        print(
            RemoteArtifactService().fetch(
                arguments.job_id, arguments.logical_name, arguments.output
            )
        )
    elif arguments.artifact_command == "plugins":
        print(json.dumps(ArtifactPluginRegistry().names(arguments.kind), indent=2))
    else:
        shapes = artifact_shapes(arguments.shape)
        artifact_validator = NumpyArtifactValidator(
            required_arrays=arguments.require_array,
            shapes=shapes,
            finite=arguments.finite,
        )
        print(
            json.dumps(
                artifact_service.validate(arguments.path, (artifact_validator,)).to_dict(),
                indent=2,
            )
        )
    return 0


def follow_plot(
    arguments: argparse.Namespace,
    visualization: VisualizationService,
    output: Path,
) -> int:
    """Synchronize metrics and atomically refresh a remote-job learning plot."""
    job_id = str(arguments.selector)
    if not job_id.startswith("job-"):
        raise ValueError("--follow requires a persistent JOB selector.")
    remote = RemoteResultService()
    while True:
        synced = remote.sync(job_id)
        spec = visualization.learning(
            synced.destination,
            metrics=arguments.metric,
            aggregate=arguments.aggregate,
            uncertainty=arguments.uncertainty,
        )
        visualization.render(spec, output)
        if remote.jobs.get(job_id).state.terminal:
            print(output)
            return 0
        time.sleep(arguments.interval)


def artifact_shapes(values: Sequence[str]) -> dict[str, tuple[int | None, ...]]:
    """Parse repeated ARRAY=DIM,DIM constraints without expressions."""
    output: dict[str, tuple[int | None, ...]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--shape must use ARRAY=DIM,DIM with * for any dimension.")
        name, raw = value.split("=", 1)
        output[name] = tuple(None if part.strip() == "*" else int(part) for part in raw.split(","))
    return output
