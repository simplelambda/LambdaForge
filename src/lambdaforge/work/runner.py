"""Plan and execute the single Work runtime without historical execution adapters."""

from __future__ import annotations

import json
import multiprocessing
import os
import random
import shutil
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any
from uuid import uuid4

import tomli

from lambdaforge._version import VERSION
from lambdaforge.data.DatasetReference import DatasetReference
from lambdaforge.data.DatasetRegistry import DatasetRegistry
from lambdaforge.data.DatasetResolver import DatasetResolver
from lambdaforge.EnvironmentManifest import EnvironmentManifest
from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.reproducibility.CodeIdentity import CodeIdentity
from lambdaforge.reproducibility.ScientificIdentity import ScientificIdentity
from lambdaforge.work.cache import WorkCache
from lambdaforge.work.checkpoints import CheckpointCollection
from lambdaforge.work.config import RunDefinition, WorkConfig, import_work_class
from lambdaforge.work.managed import fingerprint
from lambdaforge.work.models import (
    WorkConfiguration,
    WorkInput,
    WorkResources,
    WorkResult,
    WorkTrial,
    atomic_json,
)
from lambdaforge.work.runtime import WorkRuntime


@dataclass(frozen=True, slots=True)
class WorkExecutionPlan:
    """Read-only expansion and placement plan for one Work document."""

    name: str
    source: Path
    execution_id: str
    scientific_fingerprint: str
    levels: tuple[tuple[Mapping[str, Any], ...], ...]
    resources: Mapping[str, Any]
    reuse: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON planning envelope."""
        return {
            "plan_version": 1,
            "name": self.name,
            "source": str(self.source),
            "execution_id": self.execution_id,
            "scientific_fingerprint": self.scientific_fingerprint,
            "levels": [[dict(run) for run in level] for level in self.levels],
            "resources": dict(self.resources),
            "reuse": self.reuse,
        }


@dataclass(frozen=True, slots=True)
class WorkExecutionResult:
    """Aggregate all Runs in one Work Execution."""

    name: str
    execution_id: str
    scientific_fingerprint: str
    status: str
    execution_dir: Path
    runs: tuple[WorkResult, ...]
    outputs: Mapping[str, Mapping[str, Any]]
    summary: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the durable human/machine execution view."""
        return {
            "execution_result_version": 1,
            "name": self.name,
            "execution_id": self.execution_id,
            "scientific_fingerprint": self.scientific_fingerprint,
            "status": self.status,
            "execution_dir": str(self.execution_dir),
            "runs": [result.to_dict() for result in self.runs],
            "outputs": {name: dict(values) for name, values in self.outputs.items()},
            "summary": dict(self.summary),
        }


class WorkRunner:
    """Establish runtime, invoke Work.run and finalize owned scientific evidence."""

    def plan(self, config: WorkConfig, *, rerun: bool = False) -> WorkExecutionPlan:
        """Resolve expansion/identity without constructing a Work or creating state."""
        source = self._source(config)
        study_identity = self._study_identity(config)
        execution_id = self._execution_id(study_identity, rerun=rerun)
        execution_dir = self._execution_root(source, config.name) / execution_id
        levels: list[tuple[Mapping[str, Any], ...]] = []
        for level in config.levels:
            planned = []
            for definition in level.runs:
                for index, variant, seed in self._expanded(definition):
                    planned.append(
                        {
                            "name": definition.name,
                            "class": definition.work_class,
                            "seed": seed,
                            "trial": index,
                            "parameters": {**dict(definition.parameters), **dict(variant)},
                            "resources": definition.resources.to_dict(),
                        }
                    )
            levels.append(tuple(planned))
        existing = execution_dir / "result.json"
        reusable = False
        if existing.is_file() and not rerun:
            try:
                reusable = (
                    json.loads(existing.read_text(encoding="utf-8")).get("status") == "succeeded"
                )
            except (OSError, ValueError, TypeError):
                reusable = False
        return WorkExecutionPlan(
            config.name,
            source,
            execution_id,
            study_identity,
            tuple(levels),
            config.resources.to_dict(),
            reusable,
        )

    def run(
        self,
        config: WorkConfig,
        *,
        dry_run: bool = False,
        rerun: bool = False,
        restart: bool = False,
    ) -> WorkExecutionPlan | WorkExecutionResult:
        """Execute sequential levels and process-isolated parallel Work Runs."""
        plan = self.plan(config, rerun=rerun)
        if dry_run:
            return plan
        execution_dir = self._execution_root(plan.source, config.name) / plan.execution_id
        existing = execution_dir / "result.json"
        if existing.is_file() and not rerun:
            prior_execution = self._read_execution(existing)
            if prior_execution.status == "succeeded":
                return prior_execution
        execution_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(
            execution_dir / "execution.json",
            {
                "execution_manifest_version": 1,
                "name": config.name,
                "execution_id": plan.execution_id,
                "scientific_fingerprint": plan.scientific_fingerprint,
                "code_identity": _code_identity(self._project_root(plan.source.parent)),
                "consumer_package": _consumer_package(self._project_root(plan.source.parent)),
                "lambdaforge_version": VERSION,
                "source": str(plan.source),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "planned_runs": config.planned_runs,
                "resources": config.resources.to_dict(),
                "ownership": {
                    "execution_dir": "owned",
                    "published_datasets": "durable-independent",
                    "environment": "shared",
                    "cache": "reconstructible",
                },
            },
        )
        # Retry must use the exact materialized authoring document that created this
        # Execution, even when the researcher later edits or removes the source YAML.
        atomic_json(execution_dir / "configuration.json", config.raw)
        outcomes: list[WorkResult] = []
        named_outputs: dict[str, Mapping[str, Any]] = {}
        for level in config.levels:
            groups: list[list[dict[str, Any]]] = []
            for definition in level.runs:
                parameters = self._resolve_references(definition.parameters, named_outputs)
                specifications: list[dict[str, Any]] = []
                for trial_index, variant, seed in self._expanded(definition):
                    specifications.append(
                        {
                            "definition": {
                                "name": definition.name,
                                "work_class": definition.work_class,
                                "resources": definition.resources.to_dict(),
                                "variants": [dict(item) for item in definition.variants],
                                "objective": dict(definition.objective or {}),
                            },
                            "parameters": {**parameters, **dict(variant)},
                            "trial_parameters": dict(variant),
                            "seed": seed,
                            "trial_index": trial_index,
                            "execution_id": plan.execution_id,
                            "execution_dir": execution_dir,
                            "source": plan.source,
                            "restart": restart,
                        }
                    )
                groups.append(specifications)
            level_results: list[WorkResult] = []
            if len(groups) == 1:
                level_results.extend(_execute_group(groups[0]))
            else:
                worker_limit = max(1, min(len(groups), os.cpu_count() or 1))
                with ProcessPoolExecutor(
                    max_workers=worker_limit,
                    mp_context=multiprocessing.get_context("spawn"),
                ) as pool:
                    futures = {
                        pool.submit(_execute_group, group): index
                        for index, group in enumerate(groups)
                    }
                    ordered: dict[int, tuple[WorkResult, ...]] = {}
                    for future in as_completed(futures):
                        ordered[futures[future]] = future.result()
                    for index in range(len(groups)):
                        level_results.extend(ordered[index])
            outcomes.extend(level_results)
            by_definition: dict[str, list[WorkResult]] = {}
            for result in level_results:
                by_definition.setdefault(result.name, []).append(result)
            for name, results in by_definition.items():
                if len(results) != 1:
                    named_outputs[name] = {"__multiple_runs__": len(results)}
                    continue
                selected = results[0]
                if selected.ok:
                    named_outputs[name] = {
                        **dict(selected.outputs),
                        **{
                            artifact.name: {"file": str(selected.run_dir / artifact.path)}
                            for artifact in selected.artifacts
                        },
                        **{
                            name: {"dataset": f"{record['name']}@{record['version']}"}
                            for name, record in selected.datasets.items()
                        },
                    }
            if any(not result.ok for result in level_results):
                break
        status = "succeeded" if outcomes and all(result.ok for result in outcomes) else "failed"
        execution_result = WorkExecutionResult(
            config.name,
            plan.execution_id,
            plan.scientific_fingerprint,
            status,
            execution_dir,
            tuple(outcomes),
            named_outputs,
            self._summary(config, outcomes),
        )
        atomic_json(existing, execution_result.to_dict())
        return execution_result

    @staticmethod
    def _summary(config: WorkConfig, outcomes: Sequence[WorkResult]) -> Mapping[str, Any]:
        objectives = [
            (definition.name, definition.objective)
            for level in config.levels
            for definition in level.runs
            if definition.objective is not None
        ]
        summary: dict[str, Any] = {
            "planned_runs": config.planned_runs,
            "completed_runs": sum(result.ok for result in outcomes),
            "failed_runs": sum(not result.ok for result in outcomes),
        }
        if not objectives:
            return summary
        objective_name, objective = objectives[0]
        assert objective is not None
        metric, mode = objective["metric"], objective["mode"]
        objective_runs = [result for result in outcomes if result.name == objective_name]
        successful = [result for result in objective_runs if result.ok]
        if any(metric not in result.metrics for result in successful):
            missing = [result.run_id for result in successful if metric not in result.metrics]
            summary["objective"] = dict(objective)
            summary["objective_error"] = f"Successful Runs missing metric {metric!r}: {missing}."
            return summary
        grouped: dict[int, list[WorkResult]] = {}
        for result in successful:
            trial = result.trial or {"index": 0, "parameters": {}}
            grouped.setdefault(int(trial["index"]), []).append(result)
        candidates: list[dict[str, Any]] = [
            {
                "trial": trial_index,
                "parameters": dict(results[0].trial["parameters"]) if results[0].trial else {},
                "value": sum(float(result.metrics[metric]) for result in results) / len(results),
                "runs": [result.run_id for result in results],
                "seeds": [result.seed for result in results],
            }
            for trial_index, results in sorted(grouped.items())
        ]
        selected = (
            (max if mode == "max" else min)(candidates, key=lambda item: item["value"])
            if candidates
            else None
        )
        summary["objective"] = dict(objective)
        summary["candidates"] = candidates
        summary["best"] = selected
        return summary

    @staticmethod
    def _resolve_references(value: Any, outputs: Mapping[str, Mapping[str, Any]]) -> Any:
        if isinstance(value, Mapping):
            if set(value) == {"from"}:
                producer, _, output = str(value["from"]).partition(".")
                selected = outputs[producer]
                if "__multiple_runs__" in selected:
                    raise ValueError(
                        f"Output reference {value['from']!r} is ambiguous across "
                        f"{selected['__multiple_runs__']} Runs."
                    )
                if output not in selected:
                    raise KeyError(f"Work {producer!r} did not register output {output!r}.")
                return selected[output]
            return {
                str(key): WorkRunner._resolve_references(item, outputs)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [WorkRunner._resolve_references(item, outputs) for item in value]
        return value

    @staticmethod
    def _expanded(definition: RunDefinition) -> Sequence[tuple[int, Mapping[str, Any], int | None]]:
        return tuple(
            (trial_index, variant, seed)
            for trial_index, variant in enumerate(definition.variants, 1)
            for seed in definition.seeds
        )

    @staticmethod
    def _study_identity(config: WorkConfig) -> str:
        source = WorkRunner._source(config)
        source_dir = source.parent
        payload = {
            "identity_version": 1,
            "name": config.name,
            "source": _code_identity(WorkRunner._project_root(source.parent)),
            "levels": [
                [
                    {
                        "name": definition.name,
                        "work_class": definition.work_class,
                        "parameters": _identity_values(definition.parameters, source_dir),
                        "seeds": definition.seeds,
                        "variants": [dict(variant) for variant in definition.variants],
                        "objective": dict(definition.objective or {}),
                    }
                    for definition in level.runs
                ]
                for level in config.levels
            ],
        }
        return ScientificIdentity.from_payload(payload).digest

    @staticmethod
    def _source(config: WorkConfig) -> Path:
        if config.source is None:
            raise ValueError("Executing Work requires a source YAML path.")
        return config.source

    @staticmethod
    def _execution_root(source: Path, name: str) -> Path:
        configured = os.environ.get("LAMBDAFORGE_RUN_ROOT")
        project = WorkRunner._project_root(source.parent)
        root = Path(configured).expanduser() if configured else project / ".lambdaforge" / "runs"
        slug = "".join(
            character if character.isalnum() or character in "-_." else "-" for character in name
        ).strip(".-")
        return root.resolve() / (slug or "work")

    @staticmethod
    def _project_root(start: Path) -> Path:
        return next(
            (
                candidate
                for candidate in (start, *start.parents)
                if (candidate / "pyproject.toml").is_file()
            ),
            start,
        )

    @staticmethod
    def _execution_id(identity: str, *, rerun: bool) -> str:
        base = identity.removeprefix("sha256:")[:16]
        return f"execution-{base}-{uuid4().hex[:8]}" if rerun else f"execution-{base}"

    @staticmethod
    def _read_execution(path: Path) -> WorkExecutionResult:
        value = json.loads(path.read_text(encoding="utf-8"))
        runs = tuple(_work_result_from_mapping(item) for item in value.get("runs", ()))
        return WorkExecutionResult(
            str(value["name"]),
            str(value["execution_id"]),
            str(value["scientific_fingerprint"]),
            str(value["status"]),
            Path(str(value["execution_dir"])),
            runs,
            value.get("outputs", {}),
            value.get("summary", {}),
        )


def _execute_group(specifications: Sequence[Mapping[str, Any]]) -> tuple[WorkResult, ...]:
    """Execute one Work definition's seed/trial Runs serially inside its allocation."""
    outcomes: list[WorkResult] = []
    for specification in specifications:
        result = _execute_run(specification)
        outcomes.append(result)
        if not result.ok:
            break
    return tuple(outcomes)


def _execute_run(specification: Mapping[str, Any]) -> WorkResult:
    definition_value = specification["definition"]
    definition = RunDefinition(
        str(definition_value["name"]),
        str(definition_value["work_class"]),
        {},
        ResourceRequest.from_mapping(definition_value["resources"]),
        variants=tuple(definition_value["variants"]),
        objective=definition_value.get("objective") or None,
    )
    source = Path(specification["source"])
    execution_dir = Path(specification["execution_dir"])
    project_root = WorkRunner._project_root(source.parent)
    parameters, inputs, identity_parameters = _resolve_inputs(
        specification["parameters"], source.parent
    )
    seed = specification["seed"]
    trial_index = int(specification["trial_index"])
    code_identity = _code_identity(project_root)
    identity = ScientificIdentity.from_payload(
        {
            "identity_version": 1,
            "work_class": definition.work_class,
            "code": code_identity,
            "parameters": identity_parameters,
            "inputs": [value.identity_dict() for value in inputs.values()],
            "seed": seed,
            "trial_parameters": dict(specification.get("trial_parameters", {})),
        }
    ).digest
    run_id = f"run-{identity.removeprefix('sha256:')[:20]}"
    run_root = execution_dir / "runs" / run_id
    checkpoint_root = run_root / "checkpoints"
    if specification["restart"] and checkpoint_root.exists():
        if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
            raise RuntimeError(f"Unsafe checkpoint root: {checkpoint_root}")
        shutil.rmtree(checkpoint_root)
    attempts = run_root / "attempts"
    attempt_number = 1 + len(tuple(attempts.glob("attempt-*"))) if attempts.is_dir() else 1
    attempt_id = f"attempt-{attempt_number:04d}"
    run_dir = attempts / attempt_id
    run_dir.mkdir(parents=True, exist_ok=False)
    temp_dir = run_dir / "tmp"
    temp_dir.mkdir()
    checkpoints = CheckpointCollection(checkpoint_root)
    resuming = any(path.is_file() for path in checkpoint_root.rglob("*"))
    resources = _work_resources(definition.resources.to_dict())
    config_view = WorkConfiguration(
        definition.name,
        definition.work_class,
        parameters,
        resources,
    )
    trial_parameters = dict(specification.get("trial_parameters", {}))
    trial = WorkTrial(trial_index, trial_parameters) if definition.variants != ({},) else None
    runtime = WorkRuntime(
        config_view,
        inputs,
        resources,
        run_dir,
        temp_dir,
        project_root,
        checkpoints,
        WorkCache(
            Path(os.environ.get("LAMBDAFORGE_CACHE_ROOT", project_root / ".lambdaforge" / "cache"))
            / "work"
            / identity.removeprefix("sha256:"),
            hold_gc_lease=True,
        ),
        seed,
        trial,
        resuming,
        identity,
        str(specification["execution_id"]),
        run_id,
        attempt_id,
    )
    created = datetime.now(timezone.utc)
    started = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    status = "succeeded"
    primary: Any = None
    failure: Mapping[str, Any] | None = None
    try:
        environment = EnvironmentManifest.capture(project_root)
        environment.write(run_dir / "environment.json")
        target = import_work_class(definition.work_class)
        instance = target()
        instance._bind(runtime)
        _seed(seed)
        with (run_dir / "work.log").open("a", encoding="utf-8", buffering=1) as log:
            with redirect_stdout(_Tee(log, sys.stdout)), redirect_stderr(_Tee(log, sys.stderr)):
                primary = instance.run(**parameters)
        json.dumps(primary, allow_nan=False)
        runtime.outputs.finalize()
    except BaseException as error:
        status = "failed"
        # A rejected return value must not leak into the durable result envelope. This
        # matters for values such as JSON's non-standard NaN and Infinity extensions.
        primary = None
        failure = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        from lambdaforge.diagnostics import work_failure_diagnostic

        failure = {
            **failure,
            "diagnostic": work_failure_diagnostic(
                name=definition.name,
                source=source,
                error=failure,
                run_dir=run_dir,
            ).to_dict(),
        }
    finally:
        # The initial manifest guarantees a record even if Work construction fails. Re-capturing
        # here adds only external tools that the Work actually required or executed.
        try:
            EnvironmentManifest.capture(
                project_root,
                external_tools=runtime.tools.provenance,
            ).write(run_dir / "environment.json")
        except Exception:
            pass
        if temp_dir.exists() and temp_dir.is_dir() and not temp_dir.is_symlink():
            shutil.rmtree(temp_dir)
        runtime.cache.close()
    finished = datetime.now(timezone.utc)
    result = WorkResult(
        definition.name,
        definition.work_class,
        str(specification["execution_id"]),
        run_id,
        attempt_id,
        attempt_number,
        identity,
        status,
        run_dir,
        created.isoformat(),
        started.isoformat(),
        finished.isoformat(),
        time.perf_counter() - started_clock,
        seed,
        {"index": trial.index, "parameters": dict(trial.parameters)} if trial else None,
        identity_parameters,
        tuple(inputs.values()),
        resources,
        primary,
        runtime.outputs.values,
        runtime.metrics.latest,
        runtime.metrics.count,
        runtime.outputs.artifacts,
        runtime.outputs.datasets,
        failure=failure,
        resumed_from_checkpoint=resuming,
        job_id=os.environ.get("LAMBDAFORGE_JOB_ID"),
    )
    result.write(run_dir / "result.json")
    atomic_json(run_root / "result.json", result.to_dict())
    return result


def _resolve_inputs(
    value: Mapping[str, Any], source_dir: Path
) -> tuple[dict[str, Any], dict[str, WorkInput], dict[str, Any]]:
    inputs: dict[str, WorkInput] = {}

    def resolve(item: Any, name: str) -> tuple[Any, Any]:
        if isinstance(item, Mapping):
            if set(item) == {"file"}:
                configured = str(item["file"])
                path = Path(configured)
                path = path.resolve() if path.is_absolute() else (source_dir / path).resolve()
                digest, size = _path_fingerprint(path)
                record = WorkInput(name, "file", configured, path, digest, size_bytes=size)
                inputs[name] = record
                return path, {"file": {"sha256": digest, "size_bytes": size}}
            if set(item) == {"dataset"}:
                configured = str(item["dataset"])
                reference = (
                    configured if configured.startswith("dataset:") else f"dataset:{configured}"
                )
                registry_path = os.environ.get(
                    "LAMBDAFORGE_DATASET_REGISTRY"
                ) or DatasetRegistry.project_path(source_dir)
                cluster = os.environ.get("LAMBDAFORGE_CLUSTER", "local")
                resolution = DatasetResolver(
                    DatasetRegistry(registry_path),
                    environment=cluster,
                    managed_environment=cluster,
                    source_dir=source_dir,
                ).resolve(DatasetReference.parse(reference))
                path = Path(resolution.location.uri)
                content_id = (
                    str(
                        resolution.identity.get("content_id")
                        or resolution.identity.get("dataset_id")
                        or ""
                    )
                    or None
                )
                record = WorkInput(
                    name, "dataset", resolution.exact_reference, path, content_id=content_id
                )
                inputs[name] = record
                return path, {"dataset": resolution.exact_reference, "content_id": content_id}
            resolved: dict[str, Any] = {}
            identities: dict[str, Any] = {}
            for key, nested in item.items():
                resolved[str(key)], identities[str(key)] = resolve(nested, f"{name}.{key}")
            return resolved, identities
        if isinstance(item, list):
            pairs = [resolve(nested, f"{name}[{index}]") for index, nested in enumerate(item)]
            return [pair[0] for pair in pairs], [pair[1] for pair in pairs]
        return item, item

    parameters: dict[str, Any] = {}
    identities: dict[str, Any] = {}
    for key, item in value.items():
        parameters[str(key)], identities[str(key)] = resolve(item, str(key))
    return parameters, inputs, identities


def _identity_values(value: Any, source_dir: Path) -> Any:
    """Resolve typed input identity while retaining output references as logical values."""
    if isinstance(value, Mapping):
        if set(value) == {"file"}:
            configured = str(value["file"])
            path = Path(configured)
            path = path.resolve() if path.is_absolute() else (source_dir / path).resolve()
            digest, size = _path_fingerprint(path)
            return {"file": {"sha256": digest, "size_bytes": size}}
        if set(value) == {"dataset"}:
            configured = str(value["dataset"])
            reference = configured if configured.startswith("dataset:") else f"dataset:{configured}"
            registry_path = os.environ.get(
                "LAMBDAFORGE_DATASET_REGISTRY", DatasetRegistry.project_path(source_dir)
            )
            parsed = DatasetReference.parse(reference)
            record = DatasetRegistry(registry_path).get(parsed.selector)
            return {
                "dataset": record.key,
                "content_id": record.dataset_id,
            }
        return {str(key): _identity_values(item, source_dir) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_identity_values(item, source_dir) for item in value]
    return value


def _path_fingerprint(path: Path) -> tuple[str, int]:
    if not path.exists() or path.is_symlink():
        raise FileNotFoundError(f"File input is missing or symbolic: {path}")
    digest, size = fingerprint(path)
    return f"sha256:{digest}", size


def _code_identity(project_root: Path) -> dict[str, Any]:
    """Use the submitting project's identity embedded in a remote bundle when present."""
    embedded = project_root / "code-identity.json"
    if (
        os.environ.get("LAMBDAFORGE_BUNDLE") == "1"
        and embedded.is_file()
        and not embedded.is_symlink()
    ):
        value = json.loads(embedded.read_text(encoding="utf-8"))
        if isinstance(value, dict) and {"provider", "revision"}.issubset(value):
            return value
        raise ValueError(f"Invalid embedded code identity: {embedded}")
    return CodeIdentity.capture(project_root).to_dict()


def _consumer_package(project_root: Path) -> dict[str, str | None]:
    """Capture the installable consumer name/version without storing its repository."""
    pyproject = project_root / "pyproject.toml"
    if not pyproject.is_file() or pyproject.is_symlink():
        return {"name": None, "version": None}
    try:
        value = tomli.loads(pyproject.read_text(encoding="utf-8"))
        project = value.get("project", {})
        name = str(project.get("name") or "").strip() or None
        declared = str(project.get("version") or "").strip() or None
        installed = metadata.version(name) if name is not None else None
        return {"name": name, "version": installed or declared}
    except metadata.PackageNotFoundError:
        return {"name": name, "version": declared}
    except (OSError, TypeError, ValueError):
        return {"name": None, "version": None}


def _work_resources(value: Mapping[str, Any]) -> WorkResources:
    return WorkResources(
        int(value["cpu_cores"]),
        int(value["ram_bytes"]),
        int(value["gpu_count"]),
        int(value["gpu_memory_bytes"]),
        float(value["runtime_seconds"]) if value["runtime_seconds"] is not None else None,
        int(value["storage_bytes"]),
        int(value["processes"]),
    )


def _seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _work_result_from_mapping(value: Mapping[str, Any]) -> WorkResult:
    from lambdaforge.work.models import WorkArtifact

    resources = value.get("resources", {}).get("requested", {})
    return WorkResult(
        str(value["name"]),
        str(value["work_class"]),
        str(value["execution_id"]),
        str(value["run_id"]),
        str(value["attempt_id"]),
        int(value["attempt_number"]),
        str(value["scientific_fingerprint"]),
        str(value["status"]),
        Path(str(value["run_dir"])),
        str(value["created_at_utc"]),
        str(value["started_at_utc"]),
        str(value["finished_at_utc"]),
        float(value["duration_seconds"]),
        int(value["seed"]) if value.get("seed") is not None else None,
        value.get("trial"),
        value.get("parameters", {}),
        tuple(
            WorkInput(**{**item, "path": Path(item["path"])}) for item in value.get("inputs", ())
        ),
        _work_resources(
            {
                "cpu_cores": resources.get("cpu", 1),
                "ram_bytes": resources.get("memory", 0),
                "gpu_count": resources.get("gpu", 0),
                "gpu_memory_bytes": resources.get("gpu_memory", 0),
                "runtime_seconds": resources.get("time"),
                "storage_bytes": resources.get("storage", 0),
                "processes": resources.get("processes", 1),
            }
        ),
        value.get("result"),
        value.get("outputs", {}),
        value.get("metrics", {}),
        int(value.get("metric_observations", 0)),
        tuple(WorkArtifact(**item) for item in value.get("artifacts", ())),
        value.get("datasets", {}),
        str(value.get("environment_manifest", "environment.json")),
        str(value.get("logs", "work.log")),
        value.get("failure"),
        bool(value.get("resumed_from_checkpoint", False)),
        str(value["job_id"]) if value.get("job_id") is not None else None,
    )


class _Tee:
    def __init__(self, first: Any, second: Any) -> None:
        self.first = first
        self.second = second

    def write(self, value: str) -> int:
        self.first.write(value)
        self.first.flush()
        self.second.write(value)
        self.second.flush()
        return len(value)

    def flush(self) -> None:
        self.first.flush()
        self.second.flush()
