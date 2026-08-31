"""Plan and execute the single Work runtime without historical execution adapters."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import multiprocessing
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from contextlib import contextmanager, redirect_stderr, redirect_stdout
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
from lambdaforge.hpo.AdaptiveSampler import AdaptiveSampler, CandidateObservation
from lambdaforge.hpo.AdaptiveSearch import AdaptiveSearchPolicy
from lambdaforge.hpo.AdaptiveStatistics import AdaptiveSeedRacer
from lambdaforge.hpo.BayesianSampler import BayesianSampler
from lambdaforge.reproducibility.CodeIdentity import CodeIdentity
from lambdaforge.reproducibility.ScientificIdentity import ScientificIdentity
from lambdaforge.work.cache import WorkCache
from lambdaforge.work.checkpoints import CheckpointCollection
from lambdaforge.work.config import RunDefinition, WorkConfig, import_work_class
from lambdaforge.work.managed import fingerprint
from lambdaforge.work.models import (
    WorkConfiguration,
    WorkFidelity,
    WorkInput,
    WorkResources,
    WorkResult,
    WorkTrial,
    atomic_json,
)
from lambdaforge.work.paths import WorkPathContext
from lambdaforge.work.retention import compact_attempt
from lambdaforge.work.runtime import WorkRuntime
from lambdaforge.work.study import StudyTelemetry

_GPU_ADMISSION_POLL_SECONDS = 1.0
_GPU_LAUNCH_STAGGER_SECONDS = 5.0
_GPU_WAIT_LOG_SECONDS = 30.0


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
        self._verify_shared_bundle_inputs(source)
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
                            "search": (
                                definition.search_policy.to_dict()
                                if definition.search_policy is not None
                                else {"strategy": "exhaustive"}
                            ),
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

    @staticmethod
    def _verify_shared_bundle_inputs(source: Path) -> None:
        """Recheck controller-declared shared bytes immediately before Work planning."""
        marker = source.parent / ".lambdaforge-shared-inputs.json"
        if not marker.exists():
            return
        if marker.is_symlink() or not marker.is_file() or marker.stat().st_size > 1024 * 1024:
            raise ValueError(f"Invalid shared-input identity file: {marker}")
        try:
            values = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid shared-input identity file: {marker}") from error
        if not isinstance(values, list):
            raise ValueError(f"Invalid shared-input identity file: {marker}")
        for value in values:
            if not isinstance(value, dict):
                raise ValueError(f"Invalid shared-input identity file: {marker}")
            path = Path(str(value.get("remote_path", "")))
            if not path.is_absolute():
                raise ValueError(f"Invalid shared-input identity file: {marker}")
            try:
                digest, size = fingerprint(path)
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"Shared project input {value.get('configured')!r} is not usable at {path}: "
                    f"{error}"
                ) from error
            kind = "file" if path.is_file() else "directory"
            observed = {"kind": kind, "sha256": digest, "size_bytes": size}
            differences = [
                f"{field} expected={value.get(field)!r} observed={observed[field]!r}"
                for field in observed
                if value.get(field) != observed[field]
            ]
            if differences:
                raise ValueError(
                    f"Shared project input {value.get('configured')!r} changed before execution "
                    f"at {path}: {'; '.join(differences)}. Synchronize the mirror and retry."
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
                self._publish_job_result(prior_execution)
                self._compact_outcomes(prior_execution.runs)
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
                                "search_policy": (
                                    definition.search_policy.to_dict()
                                    if definition.search_policy is not None
                                    else None
                                ),
                                "study_expected": definition.study_expected,
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
        self._publish_job_result(execution_result)
        self._compact_outcomes(outcomes)
        return execution_result

    @staticmethod
    def _compact_outcomes(outcomes: Sequence[WorkResult]) -> None:
        """Compact only content proven redundant after its result is durable."""
        for outcome in outcomes:
            try:
                compact_attempt(outcome)
            except Exception as error:
                print(
                    f"[retention] Could not compact {outcome.attempt_id}: "
                    f"{type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )

    @staticmethod
    def _publish_job_result(result: WorkExecutionResult) -> None:
        """Publish one exact Job-level copy for provider-independent diagnostics."""
        configured = os.environ.get("LAMBDAFORGE_JOB_RESULT_PATH")
        if configured:
            atomic_json(Path(configured).expanduser().resolve(), result.to_dict())

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
            "pruned_runs": sum(result.pruned for result in outcomes),
        }
        adaptive_definitions = [
            definition
            for level in config.levels
            for definition in level.runs
            if definition.search_policy is not None
        ]
        if adaptive_definitions:
            summary["adaptive_controller"] = {
                "strategy": "adaptive",
                "decisions": "hpo-control/decisions.jsonl",
                "state": "hpo-control/state.json",
                "policies": [
                    definition.search_policy.to_dict()
                    for definition in adaptive_definitions
                    if definition.search_policy is not None
                ],
            }
        if not objectives:
            return summary
        objective_name, objective = objectives[0]
        assert objective is not None
        metric, mode = objective["metric"], objective["mode"]
        objective_runs = [result for result in outcomes if result.name == objective_name]
        successful_all = [result for result in objective_runs if result.ok and not result.pruned]
        latest: dict[tuple[int, int | None, str | None], WorkResult] = {}
        for result in successful_all:
            trial = result.trial or {"index": 0}
            key = (int(trial["index"]), result.seed, result.study_phase)
            target = int((result.fidelity or {}).get("target", 2**31 - 1))
            previous = latest.get(key)
            previous_target = (
                int((previous.fidelity or {}).get("target", 2**31 - 1))
                if previous is not None
                else -1
            )
            if previous is None or (target, result.attempt_number) >= (
                previous_target,
                previous.attempt_number,
            ):
                latest[key] = result
        successful = list(latest.values())
        if any(_result_objective(result, metric) is None for result in successful):
            missing = [
                result.run_id for result in successful if _result_objective(result, metric) is None
            ]
            summary["objective"] = dict(objective)
            summary["objective_error"] = f"Successful Runs missing metric {metric!r}: {missing}."
            return summary
        grouped: dict[int, list[WorkResult]] = {}
        for result in successful:
            trial = result.trial or {"index": 0, "parameters": {}}
            grouped.setdefault(int(trial["index"]), []).append(result)
        candidates: list[dict[str, Any]] = []
        selection_scores: dict[int, float] = {}
        has_confirmation = any(result.study_phase == "confirmation" for result in successful)
        for trial_index, results in sorted(grouped.items()):
            search_results = [result for result in results if result.study_phase != "confirmation"]
            confirmation_results = [
                result for result in results if result.study_phase == "confirmation"
            ]
            evidence = confirmation_results or search_results
            feasibility = _candidate_constraints(evidence, objective)
            value = statistics.fmean(
                value
                for result in evidence
                for value in [_result_objective(result, metric)]
                if value is not None
            )
            candidate = {
                "trial": trial_index,
                "parameters": dict(results[0].trial["parameters"]) if results[0].trial else {},
                "value": value,
                "runs": [result.run_id for result in results],
                "seeds": sorted(
                    (result.seed for result in search_results),
                    key=lambda seed: -1 if seed is None else seed,
                ),
                "search_value": (
                    statistics.fmean(
                        value
                        for result in search_results
                        for value in [_result_objective(result, metric)]
                        if value is not None
                    )
                    if search_results
                    else None
                ),
                "confirmation_value": (
                    statistics.fmean(
                        value
                        for result in confirmation_results
                        for value in [_result_objective(result, metric)]
                        if value is not None
                    )
                    if confirmation_results
                    else None
                ),
                "confirmation_seeds": sorted(
                    (result.seed for result in confirmation_results),
                    key=lambda seed: -1 if seed is None else seed,
                ),
                "search_uncertainty": _sample_uncertainty(search_results, metric),
                "confirmation_uncertainty": _sample_uncertainty(confirmation_results, metric),
                "feasibility": feasibility,
                "fidelity": sorted(
                    {
                        int(result.fidelity["target"])
                        for result in results
                        if result.fidelity is not None
                    }
                ),
            }
            candidates.append(candidate)
            if confirmation_results and feasibility["feasible"]:
                selection_scores[trial_index] = value
            elif not has_confirmation and feasibility["feasible"]:
                score = _candidate_score(search_results, metric, mode, 1.0)
                if score is not None:
                    selection_scores[trial_index] = score
        selected = (
            (max if mode == "max" else min)(
                (candidate for candidate in candidates if candidate["trial"] in selection_scores),
                key=lambda item: selection_scores[int(item["trial"])],
            )
            if selection_scores
            else None
        )
        if selected is not None:
            selected = {
                **selected,
                "selection_score": selection_scores[int(selected["trial"])],
                "selection_basis": (
                    "fresh-confirmation-mean"
                    if selected["confirmation_seeds"]
                    else "conservative-search-bound"
                ),
            }
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
                        "search_policy": (
                            definition.search_policy.to_dict()
                            if definition.search_policy is not None
                            else None
                        ),
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
    """Execute one Work definition serially or through its adaptive controller."""
    if specifications:
        raw = specifications[0]["definition"].get("search_policy")
        if isinstance(raw, Mapping):
            return _execute_adaptive_group(specifications, AdaptiveSearchPolicy.from_search(raw))
    definition = specifications[0]["definition"] if specifications else {}
    telemetry = (
        StudyTelemetry.from_environment() if definition.get("study_expected") is True else None
    )
    if telemetry is not None and specifications:
        telemetry.initialize(
            name=str(definition["name"]),
            execution_id=str(specifications[0]["execution_id"]),
            strategy="exhaustive",
            objective=definition.get("objective"),
            specifications=specifications,
        )
    outcomes: list[WorkResult] = []
    for specification in specifications:
        if telemetry is not None:
            telemetry.schedule((specification,))
        result = _execute_run(specification)
        outcomes.append(result)
        if telemetry is not None:
            telemetry.refresh()
        if not result.ok:
            break
    if telemetry is not None:
        telemetry.candidate_states(
            active=tuple(int(value["trial_index"]) for value in specifications[: len(outcomes)]),
            ranked=tuple(int(value["trial_index"]) for value in specifications),
            finished=True,
        )
    return tuple(outcomes)


def _execute_run(specification: Mapping[str, Any]) -> WorkResult:
    """Apply process-local HPO/GPU context and execute one isolated Run."""
    environment: dict[str, str] = {}
    gpu_slot = specification.get("gpu_slot")
    if gpu_slot is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_slot)
    metrics_path = specification.get("hpo_metrics_path")
    stop_path = specification.get("hpo_stop_path")
    objective = specification.get("hpo_objective")
    if metrics_path is not None:
        environment["LAMBDAFORGE_HPO_METRICS_PATH"] = str(metrics_path)
    if stop_path is not None:
        environment["LAMBDAFORGE_STOP_REQUEST_PATH"] = str(stop_path)
    if objective is not None:
        environment["LAMBDAFORGE_HPO_OBJECTIVE"] = str(objective)
    semaphore = specification.get("gpu_semaphore")
    if semaphore is not None:
        semaphore.acquire()
    try:
        with _scoped_environment(environment):
            try:
                return _execute_run_inner(specification)
            except BaseException as error:
                telemetry = (
                    StudyTelemetry.from_environment()
                    if specification["definition"].get("study_expected") is True
                    else None
                )
                if telemetry is not None:
                    telemetry.run_crashed(specification, error)
                raise
    finally:
        if gpu_slot is not None:
            _release_unused_cuda_cache()
        if semaphore is not None:
            semaphore.release()


def _release_unused_cuda_cache() -> None:
    """Return one finished packed Run's unused allocator cache to its GPU."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except Exception:
        # Cleanup must never replace the scientific Run outcome.  The next
        # admission/launch still fails normally if the CUDA runtime is unhealthy.
        return


def _execute_run_inner(specification: Mapping[str, Any]) -> WorkResult:
    definition_value = specification["definition"]
    raw_policy = definition_value.get("search_policy")
    definition = RunDefinition(
        str(definition_value["name"]),
        str(definition_value["work_class"]),
        {},
        ResourceRequest.from_mapping(definition_value["resources"]),
        variants=tuple(definition_value["variants"]),
        objective=definition_value.get("objective") or None,
        search_policy=(
            AdaptiveSearchPolicy.from_search(raw_policy)
            if isinstance(raw_policy, Mapping)
            else None
        ),
        study_expected=bool(definition_value.get("study_expected", False)),
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
    raw_fidelity = specification.get("hpo_fidelity")
    fidelity = (
        WorkFidelity(
            int(raw_fidelity["current"]),
            int(raw_fidelity["target"]),
            int(raw_fidelity["maximum"]),
        )
        if isinstance(raw_fidelity, Mapping)
        else None
    )
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
        fidelity,
        resuming,
        identity,
        str(specification["execution_id"]),
        run_id,
        attempt_id,
        WorkPathContext.load(source),
    )
    training_metrics_path = run_dir / "training-metrics.jsonl"
    telemetry = StudyTelemetry.from_environment() if definition.study_expected else None
    if telemetry is not None:
        telemetry.run_started(
            specification,
            run_dir=run_dir,
            metrics_path=run_dir / "metrics.jsonl",
            training_metrics_path=training_metrics_path,
        )
    created = datetime.now(timezone.utc)
    started = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    status = "succeeded"
    primary: Any = None
    failure: Mapping[str, Any] | None = None
    try:
        run_environment = {
            "LAMBDAFORGE_TRAINING_METRICS_PATH": str(training_metrics_path),
            "LAMBDAFORGE_PROGRESS_PATH": str(run_dir / "progress.json"),
        }
        if fidelity is not None:
            run_environment.update(
                {
                    "LAMBDAFORGE_FIDELITY_CURRENT": str(fidelity.current),
                    "LAMBDAFORGE_FIDELITY_TARGET": str(fidelity.target),
                    "LAMBDAFORGE_FIDELITY_MAX": str(fidelity.maximum),
                    "LAMBDAFORGE_HPO_CHECKPOINT_DIR": str(checkpoint_root / "lightning"),
                }
            )
        with _scoped_environment(run_environment):
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
        _adopt_training_metrics(runtime, training_metrics_path)
        _adopt_hpo_objective(runtime)
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
    objective_observation = _objective_observation(
        definition.objective,
        paths=(run_dir / "metrics.jsonl", training_metrics_path),
        fallback=runtime.metrics.latest,
    )
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
        pruned=bool(
            specification.get("hpo_stop_path")
            and Path(str(specification["hpo_stop_path"])).is_file()
        ),
        prune_reason=(
            _prune_reason(Path(str(specification["hpo_stop_path"])))
            if specification.get("hpo_stop_path")
            and Path(str(specification["hpo_stop_path"])).is_file()
            else None
        ),
        gpu_index=(
            int(specification["gpu_index"]) if specification.get("gpu_index") is not None else None
        ),
        gpu_token=(
            str(specification["gpu_slot"]) if specification.get("gpu_slot") is not None else None
        ),
        study_phase=str(specification.get("hpo_phase", "search")),
        fidelity=(
            {
                "current": fidelity.current,
                "target": fidelity.target,
                "maximum": fidelity.maximum,
            }
            if fidelity is not None
            else None
        ),
        objective_observation=objective_observation,
    )
    result.write(run_dir / "result.json")
    atomic_json(run_root / "result.json", result.to_dict())
    if telemetry is not None:
        telemetry.run_finished(specification, result)
    return result


def _execute_adaptive_group(
    specifications: Sequence[Mapping[str, Any]], policy: AdaptiveSearchPolicy
) -> tuple[WorkResult, ...]:
    """Interleave new candidates and probability-driven shared-seed evidence."""
    if not specifications:
        return ()
    definition = specifications[0]["definition"]
    resources = ResourceRequest.from_mapping(definition["resources"])
    objective = definition.get("objective") or {}
    metric = str(objective["metric"])
    mode = str(objective["mode"])
    parallelism = _adaptive_parallelism(resources, policy)
    by_trial: dict[int, list[Mapping[str, Any]]] = {}
    for specification in specifications:
        by_trial.setdefault(int(specification["trial_index"]), []).append(specification)
    all_trials = sorted(by_trial)
    control_root = Path(str(specifications[0]["execution_dir"])) / "hpo-control"
    control_root.mkdir(parents=True, exist_ok=True)
    decisions_path = control_root / "decisions.jsonl"
    state_path = control_root / "state.json"
    decision_number = 0
    if decisions_path.is_file():
        with decisions_path.open(encoding="utf-8") as existing_decisions:
            decision_number = sum(1 for _ in existing_decisions)

    def record_decision(action: str, **evidence: Any) -> None:
        nonlocal decision_number
        decision_number += 1
        event = {
            "event_version": 1,
            "decision": decision_number,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "action": action,
            **evidence,
        }
        with decisions_path.open("a", encoding="utf-8", buffering=1) as stream:
            stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if telemetry is not None:
            telemetry.controller_decision(event)

    seed_count = max(len(values) for values in by_trial.values())
    completed: dict[int, list[WorkResult]] = {trial: [] for trial in all_trials}
    outcomes: list[WorkResult] = []
    telemetry = StudyTelemetry.from_environment()
    if telemetry is not None:
        telemetry.initialize(
            name=str(definition["name"]),
            execution_id=str(specifications[0]["execution_id"]),
            strategy="adaptive",
            objective=objective,
            specifications=(),
            planned_runs=len(specifications)
            + min(policy.confirmation_top_k, len(all_trials)) * len(policy.confirmation_seeds),
            planned_candidates=len(all_trials),
        )
    print(
        f"[hpo] action-adaptive search: candidate_budget={len(all_trials)} "
        f"startup={min(policy.startup_trials, len(all_trials))} seeds={seed_count} "
        f"parallel={parallelism} runs_per_gpu={policy.runs_per_gpu}",
        flush=True,
    )
    record_decision(
        "INITIALIZE",
        candidate_budget=len(all_trials),
        search_seed_budget=seed_count,
        policy=policy.to_dict(),
    )
    candidate_parameters = {
        trial: dict(by_trial[trial][0].get("trial_parameters", {})) for trial in all_trials
    }
    selector = AdaptiveSampler(candidate_parameters, mode=mode)
    bayesian = BayesianSampler(candidate_parameters, mode=mode)
    racer = AdaptiveSeedRacer(
        mode=mode,
        margin=policy.equivalence_margin,
        probability_threshold=policy.seed_probability_threshold,
    )
    proposed: list[int] = []
    proposal_numbers: dict[int, int] = {}
    pool_trials_by_proposal: dict[int, int] = {}
    inflight_by_trial: dict[int, int] = {}

    def public_trial(pool_trial: int) -> int:
        public_trial = proposal_numbers.get(pool_trial)
        if public_trial is None:
            public_trial = len(proposal_numbers) + 1
            proposal_numbers[pool_trial] = public_trial
            pool_trials_by_proposal[public_trial] = pool_trial
        return public_trial

    def prepared(
        pool_trial: int, raw: Mapping[str, Any], *, phase: str = "search"
    ) -> dict[str, Any]:
        value = dict(raw)
        value["candidate_pool_index"] = pool_trial
        value["trial_index"] = public_trial(pool_trial)
        value["hpo_phase"] = phase
        if policy.fidelity is not None:
            value["hpo_fidelity"] = {
                "current": 0,
                "target": policy.fidelity.minimum,
                "maximum": policy.fidelity.maximum,
            }
        return value

    def initial_specifications(pool_trial: int) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        for raw in by_trial[pool_trial][: policy.min_seeds]:
            value = dict(raw)
            value["candidate_pool_index"] = pool_trial
            value["trial_index"] = public_trial(pool_trial)
            value["hpo_phase"] = "search"
            if policy.fidelity is not None:
                value["hpo_fidelity"] = {
                    "current": 0,
                    "target": policy.fidelity.minimum,
                    "maximum": policy.fidelity.maximum,
                }
            values.append(value)
        return values

    def next_seed_specification(pool_trial: int) -> dict[str, Any] | None:
        attempted = {result.seed for result in completed[pool_trial]}
        for raw in by_trial[pool_trial]:
            if raw.get("seed") not in attempted:
                return prepared(pool_trial, raw)
        return None

    def resume_specification(result: WorkResult) -> dict[str, Any] | None:
        if policy.fidelity is None or result.fidelity is None or not result.ok or result.pruned:
            return None
        current = int(result.fidelity["target"])
        if current >= policy.fidelity.maximum:
            return None
        pool_trial = pool_trial_for(result)
        raw = next(
            (value for value in by_trial[pool_trial] if value.get("seed") == result.seed),
            None,
        )
        if raw is None:
            return None
        value = prepared(pool_trial, raw)
        value["hpo_fidelity"] = {
            "current": current,
            "target": min(
                policy.fidelity.maximum,
                max(current + 1, current * policy.fidelity.reduction_factor),
            ),
            "maximum": policy.fidelity.maximum,
        }
        return value

    def pool_trial_for(result: WorkResult) -> int:
        public_trial = int((result.trial or {"index": 0})["index"])
        return pool_trials_by_proposal[public_trial]

    def values_by_trial(*, final_only: bool = True) -> dict[int, dict[int | None, float]]:
        values: dict[int, dict[int | None, tuple[int, float, WorkResult]]] = {}
        for trial, results in completed.items():
            for result in results:
                objective_value = _result_objective(result, metric)
                if not result.ok or result.pruned or objective_value is None:
                    continue
                target = (
                    int(result.fidelity["target"]) if result.fidelity is not None else 2**31 - 1
                )
                maximum = int(result.fidelity["maximum"]) if result.fidelity is not None else target
                if final_only and target < maximum:
                    continue
                previous = values.setdefault(trial, {}).get(result.seed)
                if previous is None or target >= previous[0]:
                    values[trial][result.seed] = (target, objective_value, result)
        return {
            trial: {seed: value for seed, (_target, value, _result) in by_seed.items()}
            for trial, by_seed in values.items()
            if _candidate_constraints(
                [result for _target, _value, result in by_seed.values()], objective
            )["feasible"]
        }

    def observations() -> list[CandidateObservation]:
        estimates = racer.estimates(values_by_trial(final_only=False))
        return [
            CandidateObservation(
                trial,
                estimates[trial].mean,
                estimates[trial].standard_error,
            )
            for trial in proposed
            if trial in estimates
        ]

    def propose(count: int) -> tuple[int, ...]:
        evidence = observations()
        observed_trials = {value.trial for value in evidence}
        censored_trials = sorted(
            trial
            for trial in proposed
            if trial not in observed_trials and any(result.pruned for result in completed[trial])
        )
        pending_trials = sorted(
            trial for trial, active_count in inflight_by_trial.items() if active_count > 0
        )
        use_bayesian = policy.sampler == "botorch" or (
            policy.sampler == "auto" and BayesianSampler.available()
        )
        if use_bayesian:
            try:
                selected = bayesian.propose(
                    evidence,
                    selected=proposed,
                    pending=pending_trials,
                    censored=censored_trials,
                    count=count,
                )
                if selected:
                    print("[hpo] START_NEW selected by BoTorch qLogNEI", flush=True)
                    record_decision(
                        "PROPOSE",
                        backend="botorch-qlognei",
                        requested=count,
                        selected=list(selected),
                        observed=len(evidence),
                        censored_trials=censored_trials,
                    )
                    return selected
            except Exception as error:
                print(
                    f"[hpo] Bayesian proposal unavailable ({type(error).__name__}: {error}); "
                    "using deterministic k-NN fallback",
                    flush=True,
                )
                record_decision(
                    "SURROGATE_FALLBACK",
                    requested=policy.sampler,
                    fallback="mixed-knn",
                    error_type=type(error).__name__,
                    error=str(error),
                )
        selected = selector.propose(
            evidence,
            selected=proposed,
            pending=pending_trials,
            censored=censored_trials,
            count=count,
        )
        if selected:
            record_decision(
                "PROPOSE",
                backend="mixed-knn",
                requested=count,
                selected=list(selected),
                observed=len(evidence),
                censored_trials=censored_trials,
            )
        return selected

    started = time.monotonic()

    def allowance() -> int:
        if policy.max_runs is None:
            return 2**31 - 1
        return max(0, policy.max_runs - len(outcomes))

    def within_time() -> bool:
        return (
            policy.max_time_seconds is None or time.monotonic() - started < policy.max_time_seconds
        )

    def persist_state() -> None:
        atomic_json(
            state_path,
            {
                "state_version": 1,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "proposed_pool_trials": list(proposed),
                "public_trial_map": {
                    str(pool_trial): public for pool_trial, public in proposal_numbers.items()
                },
                "completed_runs": len(outcomes),
                "runs": [
                    {
                        "trial": int((result.trial or {"index": 0})["index"]),
                        "seed": result.seed,
                        "phase": result.study_phase,
                        "status": result.status,
                        "run_id": result.run_id,
                        "attempt_id": result.attempt_id,
                        "objective": _result_objective(result, metric),
                        "fidelity": dict(result.fidelity or {}),
                    }
                    for result in outcomes
                ],
            },
        )

    def execute(scheduled: Sequence[dict[str, Any]], observed_trials: Sequence[int]) -> None:
        limited = list(scheduled[: allowance()])
        if not limited:
            return
        if telemetry is not None:
            telemetry.schedule(limited)
        for value in limited:
            trial = int(value.get("candidate_pool_index", value["trial_index"]))
            inflight_by_trial[trial] = inflight_by_trial.get(trial, 0) + 1
        # One or two speculative candidates are enough to hide a straggler without allowing
        # stale model decisions to build an unbounded queue. They are proposed only after a real
        # terminal observation and only when the dispatcher has no already-planned Run waiting.
        lookahead_candidates = max(1, min(2, parallelism // 3))
        speculative_candidates = 0

        def observed(
            result: WorkResult, queued: int, pending_runs: int
        ) -> Sequence[dict[str, Any]]:
            nonlocal speculative_candidates
            trial = pool_trial_for(result)
            inflight_by_trial[trial] = max(0, inflight_by_trial.get(trial, 1) - 1)
            completed[trial].append(result)
            outcomes.append(result)
            persist_state()
            if queued or speculative_candidates >= lookahead_candidates or not within_time():
                return ()
            remaining_capacity = allowance() - pending_runs
            if remaining_capacity <= 0 or len(proposed) >= len(all_trials):
                return ()
            proposal = propose(1)
            if not proposal:
                return ()
            specifications = initial_specifications(proposal[0])
            if not specifications or len(specifications) > remaining_capacity:
                return ()
            proposed.extend(proposal)
            speculative_candidates += 1
            record_decision(
                "START_NEW",
                reason="bounded-async-lookahead",
                pool_trials=list(proposal),
                public_trials=[public_trial(value) for value in proposal],
                completed_runs=len(outcomes),
                running_runs=pending_runs,
                lookahead_candidate=speculative_candidates,
                lookahead_limit=lookahead_candidates,
            )
            print(
                f"[hpo] ASYNC_START_NEW trial={public_trial(proposal[0])}: "
                f"one slot became free while {pending_runs} Run(s) remain; "
                f"lookahead={speculative_candidates}/{lookahead_candidates}",
                flush=True,
            )
            if telemetry is not None:
                telemetry.schedule(specifications)
            for value in specifications:
                pool_trial = int(value.get("candidate_pool_index", value["trial_index"]))
                inflight_by_trial[pool_trial] = inflight_by_trial.get(pool_trial, 0) + 1
            return specifications

        _execute_adaptive_round(
            limited,
            resources=resources,
            policy=policy,
            objective_metric=metric,
            objective_mode=mode,
            parallelism=parallelism,
            telemetry=telemetry,
            on_result=observed,
        )
        if telemetry is not None and observed_trials:
            telemetry.candidates_observed(observed_trials)

    startup = selector.initial(min(policy.startup_trials, len(all_trials)))
    proposed.extend(startup)
    record_decision(
        "START_NEW",
        reason="space-filling-startup",
        pool_trials=list(startup),
        public_trials=[public_trial(trial) for trial in startup],
    )
    startup_specs = [value for trial in startup for value in initial_specifications(trial)]
    execute(startup_specs, startup)

    best_value: float | None = None
    stale_rounds = 0
    search_converged = False
    while within_time() and allowance() > 0:
        remaining = len(all_trials) - len(proposed)
        eligible = [trial for trial in proposed if next_seed_specification(trial) is not None]
        decisions = list(racer.decisions(values_by_trial(), eligible=eligible))
        provisional = racer.decisions(values_by_trial(final_only=False), eligible=proposed)
        competitive = {decision.trial for decision in provisional}
        latest_by_seed: dict[tuple[int, int | None], WorkResult] = {}
        for trial in proposed:
            for result in completed[trial]:
                target = int((result.fidelity or {}).get("target", 2**31 - 1))
                key = (trial, result.seed)
                previous = latest_by_seed.get(key)
                previous_target = (
                    int((previous.fidelity or {}).get("target", 2**31 - 1))
                    if previous is not None
                    else -1
                )
                if target >= previous_target:
                    latest_by_seed[key] = result
        resume_actions = [
            value
            for (trial, _seed), result in latest_by_seed.items()
            if trial in competitive
            for value in [resume_specification(result)]
            if value is not None
        ]
        resume_actions.sort(
            key=lambda value: (
                int(value["hpo_fidelity"]["target"])
                / max(1, int(value["hpo_fidelity"]["current"])),
                -int(value["trial_index"]),
            ),
            reverse=True,
        )
        # Prefer evidence with more uncertainty reduction per observed wall-clock cost.
        durations = [
            result.duration_seconds
            for result in outcomes
            if result.ok and result.duration_seconds > 0
        ]
        default_cost = statistics.fmean(durations) if durations else 1.0
        decisions.sort(
            key=lambda decision: (
                -decision.expected_uncertainty_reduction
                / max(
                    statistics.fmean(
                        [
                            result.duration_seconds
                            for result in completed[decision.trial]
                            if result.ok and result.duration_seconds > 0
                        ]
                    )
                    if any(
                        result.ok and result.duration_seconds > 0
                        for result in completed[decision.trial]
                    )
                    else default_cost,
                    1e-9,
                )
            )
        )
        slots = min(parallelism, allowance())
        evidence_actions = len(resume_actions) + len(decisions)
        seed_slots = min(
            evidence_actions, slots // 2 if remaining and not search_converged else slots
        )
        new_count = 0
        if remaining and not search_converged:
            new_count = min(remaining, max(1, slots - seed_slots))
        proposal_batch = propose(new_count)
        proposed.extend(proposal_batch)
        if proposal_batch:
            record_decision(
                "START_NEW",
                reason="adaptive-acquisition",
                pool_trials=list(proposal_batch),
                public_trials=[public_trial(trial) for trial in proposal_batch],
            )
        scheduled = [value for trial in proposal_batch for value in initial_specifications(trial)]
        available = max(0, slots - len(scheduled))
        selected_resumes = resume_actions[:available]
        scheduled.extend(selected_resumes)
        for value in selected_resumes:
            fidelity = value["hpo_fidelity"]
            print(
                f"[hpo] RESUME trial={value['trial_index']} seed={value.get('seed')} "
                f"budget={fidelity['current']}->{fidelity['target']}",
                flush=True,
            )
            record_decision(
                "RESUME",
                trial=int(value["trial_index"]),
                seed=value.get("seed"),
                current=int(fidelity["current"]),
                target=int(fidelity["target"]),
                maximum=int(fidelity["maximum"]),
            )
        available = max(0, slots - len(scheduled))
        for decision in decisions[:available]:
            next_value = next_seed_specification(decision.trial)
            if next_value is not None:
                scheduled.append(next_value)
                print(
                    f"[hpo] ADD_SEED trial={public_trial(decision.trial)} "
                    f"p_competitive={decision.probability_competitive:.3f} "
                    f"completed={decision.completed_seeds}",
                    flush=True,
                )
                record_decision(
                    "ADD_SEED",
                    trial=public_trial(decision.trial),
                    seed=next_value.get("seed"),
                    probability_competitive=decision.probability_competitive,
                    expected_uncertainty_reduction=(decision.expected_uncertainty_reduction),
                    completed_seeds=decision.completed_seeds,
                )
        if not scheduled:
            break
        execute(scheduled, proposal_batch)
        estimates = racer.estimates(values_by_trial())
        if estimates:
            current = (
                max(value.mean for value in estimates.values())
                if mode == "max"
                else min(value.mean for value in estimates.values())
            )
            improvement = (
                math.inf
                if best_value is None
                else current - best_value
                if mode == "max"
                else best_value - current
            )
            if improvement > policy.min_improvement:
                best_value, stale_rounds = current, 0
            else:
                stale_rounds += 1
            if policy.convergence_patience and stale_rounds >= policy.convergence_patience:
                search_converged = True
                print(
                    f"[hpo] new-candidate search converged after {stale_rounds} "
                    "round(s) without material improvement; resolving seed uncertainty",
                    flush=True,
                )
                record_decision(
                    "STOP_PROPOSING",
                    reason="convergence",
                    stale_rounds=stale_rounds,
                    best_objective=best_value,
                    minimum_improvement=policy.min_improvement,
                )

    estimates = racer.estimates(values_by_trial())
    ranked = sorted(
        estimates,
        key=lambda trial: estimates[trial].mean,
        reverse=mode == "max",
    )
    active = ranked[: max(1, min(len(ranked), policy.confirmation_top_k))]
    if policy.confirmation_seeds and active and within_time() and allowance() > 0:
        confirmation: list[dict[str, Any]] = []
        for trial in active:
            for seed in policy.confirmation_seeds:
                raw = dict(by_trial[trial][0])
                raw["seed"] = seed
                value = prepared(trial, raw, phase="confirmation")
                if policy.fidelity is not None:
                    value["hpo_fidelity"] = {
                        "current": 0,
                        "target": policy.fidelity.maximum,
                        "maximum": policy.fidelity.maximum,
                    }
                confirmation.append(value)
        print(
            f"[hpo] CONFIRM top={len(active)} with {len(policy.confirmation_seeds)} fresh seed(s)",
            flush=True,
        )
        record_decision(
            "CONFIRM",
            pool_trials=list(active),
            public_trials=[public_trial(trial) for trial in active],
            seeds=list(policy.confirmation_seeds),
            full_fidelity=(policy.fidelity.maximum if policy.fidelity is not None else None),
        )
        execute(confirmation, active)

    if telemetry is not None:
        telemetry.candidate_states(
            active=tuple(proposal_numbers[trial] for trial in active),
            ranked=tuple(proposal_numbers[trial] for trial in ranked),
            finished=True,
        )
    record_decision(
        "FINISH",
        completed_runs=len(outcomes),
        ranked_public_trials=[public_trial(trial) for trial in ranked],
        confirmed_public_trials=[public_trial(trial) for trial in active],
        budget_exhausted=allowance() <= 0,
        time_exhausted=not within_time(),
    )
    return tuple(outcomes)


def _execute_adaptive_round(
    specifications: Sequence[dict[str, Any]],
    *,
    resources: ResourceRequest,
    policy: AdaptiveSearchPolicy,
    objective_metric: str,
    objective_mode: str,
    parallelism: int,
    telemetry: StudyTelemetry | None = None,
    on_result: Callable[[WorkResult, int, int], Sequence[dict[str, Any]]] | None = None,
) -> tuple[WorkResult, ...]:
    if not specifications:
        return ()
    control_root = Path(specifications[0]["execution_dir"]) / "hpo-control"
    prepared: list[dict[str, Any]] = []
    per_run = _adaptive_run_resources(resources, parallelism)
    visible_gpus = _visible_gpu_tokens(resources.gpu_count)

    def prepare_specification(specification: Mapping[str, Any]) -> dict[str, Any]:
        trial = int(specification["trial_index"])
        seed = specification.get("seed")
        token = f"trial-{trial:05d}-seed-{seed if seed is not None else 'none'}"
        metrics = control_root / f"{token}.metrics.jsonl"
        stop = control_root / f"{token}.stop"
        prune_evidence = stop.with_name(stop.name + ".evidence.json")
        raw_fidelity = specification.get("hpo_fidelity")
        continuing = isinstance(raw_fidelity, Mapping) and int(raw_fidelity.get("current", 0)) > 0
        if not continuing:
            metrics.unlink(missing_ok=True)
        stop.unlink(missing_ok=True)
        prune_evidence.unlink(missing_ok=True)
        value = dict(specification)
        definition = dict(value["definition"])
        definition["resources"] = per_run.to_dict()
        value.update(
            {
                "definition": definition,
                "gpu_slot": None,
                "hpo_metrics_path": metrics,
                "hpo_stop_path": stop,
                "hpo_objective": objective_metric,
            }
        )
        return value

    prepared.extend(prepare_specification(value) for value in specifications)

    def dispatch_refill(result: WorkResult, queued: int, pending: int) -> Sequence[dict[str, Any]]:
        if on_result is None:
            return ()
        return tuple(prepare_specification(value) for value in on_result(result, queued, pending))

    results: list[WorkResult] = []
    executors: list[ProcessPoolExecutor] = []
    try:
        if resources.gpu_count:
            _execute_gpu_admitted_runs(
                prepared,
                resources=resources,
                policy=policy,
                parallelism=parallelism,
                visible_gpus=visible_gpus,
                objective_metric=objective_metric,
                objective_mode=objective_mode,
                telemetry=telemetry,
                results=results,
                executors=executors,
                on_result=dispatch_refill if on_result is not None else None,
            )
        else:
            _execute_cpu_isolated_runs(
                prepared,
                policy=policy,
                parallelism=parallelism,
                objective_metric=objective_metric,
                objective_mode=objective_mode,
                telemetry=telemetry,
                results=results,
                executors=executors,
                on_result=dispatch_refill if on_result is not None else None,
            )
    finally:
        for pool in executors:
            pool.shutdown(wait=True, cancel_futures=True)
        if telemetry is not None:
            telemetry.refresh()
    return tuple(results)


def _execute_cpu_isolated_runs(
    prepared: Sequence[dict[str, Any]],
    *,
    policy: AdaptiveSearchPolicy,
    parallelism: int,
    objective_metric: str,
    objective_mode: str,
    telemetry: StudyTelemetry | None,
    results: list[WorkResult],
    executors: list[ProcessPoolExecutor],
    on_result: Callable[[WorkResult, int, int], Sequence[dict[str, Any]]] | None = None,
) -> None:
    """Keep CPU Runs isolated so one killed worker cannot break unrelated candidates."""
    queued = deque(dict(value) for value in prepared)
    pending: dict[Any, tuple[dict[str, Any], ProcessPoolExecutor]] = {}
    while queued or pending:
        while queued and len(pending) < parallelism:
            value = queued.popleft()
            pool = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
            )
            executors.append(pool)
            try:
                future = pool.submit(_execute_run, value)
            except BaseException:
                pool.shutdown(wait=True, cancel_futures=True)
                executors.remove(pool)
                raise
            pending[future] = (value, pool)
        done, _ = wait(tuple(pending), timeout=0.5, return_when=FIRST_COMPLETED)
        for future in done:
            value, pool = pending.pop(future)
            try:
                result = future.result()
            except BaseException as error:
                retry = _retry_specification(
                    value,
                    reason=f"{type(error).__name__}: {error}",
                    policy=policy,
                    telemetry=telemetry,
                    controller_error=error,
                )
                if retry is not None:
                    queued.append(retry)
                else:
                    failure = _controller_failure_result(value, error)
                    results.append(failure)
                    if telemetry is not None:
                        telemetry.run_finished(value, failure)
                    if on_result is not None:
                        queued.extend(on_result(failure, len(queued), len(pending)))
            else:
                retry = _retry_failed_result(
                    value,
                    result,
                    policy=policy,
                    telemetry=telemetry,
                )
                if retry is not None:
                    queued.append(retry)
                else:
                    results.append(result)
                    if on_result is not None:
                        queued.extend(on_result(result, len(queued), len(pending)))
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
                executors.remove(pool)
        if policy.early_stopping and pending:
            _request_early_stops(
                tuple(value for value, _pool in pending.values()),
                metric=objective_metric,
                mode=objective_mode,
                min_step=policy.early_stopping_min_step,
                confirmations=policy.early_stopping_confirmations,
                probability_threshold=policy.seed_probability_threshold,
                margin=policy.equivalence_margin,
            )
        if telemetry is not None:
            telemetry.refresh()


def _retry_specification(
    specification: Mapping[str, Any],
    *,
    reason: str,
    policy: AdaptiveSearchPolicy,
    telemetry: StudyTelemetry | None,
    controller_error: BaseException,
) -> dict[str, Any] | None:
    """Retry only evidence of a lost worker, never arbitrary consumer exceptions."""
    name = type(controller_error).__name__
    message = str(controller_error).lower()
    retryable = name in {
        "BrokenProcessPool",
        "ChildProcessError",
        "ConnectionResetError",
        "EOFError",
    } or any(
        marker in message
        for marker in ("terminated abruptly", "worker process", "killed by signal", "lost process")
    )
    return (
        _new_retry(specification, reason=reason, policy=policy, telemetry=telemetry)
        if retryable
        else None
    )


def _retry_failed_result(
    specification: Mapping[str, Any],
    result: WorkResult,
    *,
    policy: AdaptiveSearchPolicy,
    telemetry: StudyTelemetry | None,
) -> dict[str, Any] | None:
    """Retry one likely transient resource failure; deterministic science fails once."""
    if not isinstance(result, WorkResult):
        return None
    if result.ok or result.pruned or not isinstance(result.failure, Mapping):
        return None
    kind = str(result.failure.get("type", ""))
    message = str(result.failure.get("message", "")).lower()
    resource_failure = kind in {"OutOfMemoryError", "CUDAOutOfMemoryError"} or any(
        marker in message
        for marker in (
            "cuda out of memory",
            "cublas_status_alloc_failed",
            "hip out of memory",
        )
    )
    if not resource_failure:
        return None
    return _new_retry(
        specification,
        reason=f"{kind or 'resource failure'}: {result.failure.get('message', '')}",
        policy=policy,
        telemetry=telemetry,
    )


def _new_retry(
    specification: Mapping[str, Any],
    *,
    reason: str,
    policy: AdaptiveSearchPolicy,
    telemetry: StudyTelemetry | None,
) -> dict[str, Any] | None:
    attempted = int(specification.get("controller_retry", 0))
    if attempted >= policy.failure_retries:
        return None
    retry = dict(specification)
    retry["controller_retry"] = attempted + 1
    # A retry is a new Attempt and may be admitted to a different granted GPU.
    retry.pop("gpu_index", None)
    retry["gpu_slot"] = None
    if telemetry is not None:
        telemetry.run_retrying(specification, reason=reason, retry=attempted + 1)
    print(
        f"[hpo] retrying trial={specification['trial_index']} seed={specification.get('seed')} "
        f"attempt={attempted + 1}/{policy.failure_retries}: {reason}",
        flush=True,
    )
    return retry


def _controller_failure_result(
    specification: Mapping[str, Any], error: BaseException
) -> WorkResult:
    """Persist one terminal failed Run after an isolated worker cannot return evidence."""
    definition = specification["definition"]
    encoded = json.dumps(
        {
            "work_class": definition["work_class"],
            "parameters": specification.get("parameters", {}),
            "trial_parameters": specification.get("trial_parameters", {}),
            "seed": specification.get("seed"),
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    identity = "sha256:" + hashlib.sha256(encoded).hexdigest()
    run_id = f"run-{identity.removeprefix('sha256:')[:20]}"
    run_root = Path(specification["execution_dir"]) / "runs" / run_id
    attempts = run_root / "attempts"
    attempt_number = 1 + len(tuple(attempts.glob("attempt-*"))) if attempts.is_dir() else 1
    attempt_id = f"attempt-{attempt_number:04d}"
    run_dir = attempts / attempt_id
    run_dir.mkdir(parents=True, exist_ok=False)
    message = f"{type(error).__name__}: {error}"
    (run_dir / "work.log").write_text(
        f"LambdaForge worker process failed before returning a complete WorkResult.\n{message}\n",
        encoding="utf-8",
    )
    now = datetime.now(timezone.utc).isoformat()
    failure = {
        "type": type(error).__name__,
        "message": str(error),
        "phase": "worker-process",
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
    }
    resources = _work_resources(ResourceRequest.from_mapping(definition["resources"]).to_dict())
    result = WorkResult(
        name=str(definition["name"]),
        work_class=str(definition["work_class"]),
        execution_id=str(specification["execution_id"]),
        run_id=run_id,
        attempt_id=attempt_id,
        attempt_number=attempt_number,
        scientific_fingerprint=identity,
        status="failed",
        run_dir=run_dir,
        created_at_utc=now,
        started_at_utc=now,
        finished_at_utc=now,
        duration_seconds=0.0,
        seed=(int(specification["seed"]) if specification.get("seed") is not None else None),
        trial={
            "index": int(specification["trial_index"]),
            "parameters": dict(specification.get("trial_parameters", {})),
        },
        parameters=dict(specification.get("parameters", {})),
        inputs=(),
        requested_resources=resources,
        failure=failure,
        job_id=os.environ.get("LAMBDAFORGE_JOB_ID"),
        gpu_index=(
            int(specification["gpu_index"]) if specification.get("gpu_index") is not None else None
        ),
        gpu_token=(
            str(specification["gpu_slot"]) if specification.get("gpu_slot") is not None else None
        ),
        study_phase=str(specification.get("hpo_phase", "search")),
        fidelity=(
            {
                "current": int(specification["hpo_fidelity"]["current"]),
                "target": int(specification["hpo_fidelity"]["target"]),
                "maximum": int(specification["hpo_fidelity"]["maximum"]),
            }
            if isinstance(specification.get("hpo_fidelity"), Mapping)
            else None
        ),
    )
    result.write(run_dir / "result.json")
    atomic_json(run_root / "result.json", result.to_dict())
    return result


def _execute_gpu_admitted_runs(
    prepared: Sequence[dict[str, Any]],
    *,
    resources: ResourceRequest,
    policy: AdaptiveSearchPolicy,
    parallelism: int,
    visible_gpus: Sequence[str],
    objective_metric: str,
    objective_mode: str,
    telemetry: StudyTelemetry | None,
    results: list[WorkResult],
    executors: list[ProcessPoolExecutor],
    on_result: Callable[[WorkResult, int, int], Sequence[dict[str, Any]]] | None = None,
) -> None:
    """Launch only Runs that currently fit, waiting through temporary VRAM pressure."""
    required = resources.gpu_memory_bytes
    memory = (
        _gpu_memory_inventory(resources.gpu_count)
        if required > 0
        else tuple((0, 0) for _ in range(resources.gpu_count))
    )
    _validate_gpu_memory_capacity(memory, required)
    usable = tuple(index for index, (_, total) in enumerate(memory) if required <= total)
    queued = deque(dict(value) for value in prepared)
    pending: dict[Any, tuple[dict[str, Any], int, ProcessPoolExecutor]] = {}
    active = [0] * resources.gpu_count
    last_launch = [float("-inf")] * resources.gpu_count
    admission_budget = [0] * resources.gpu_count
    next_wait_log = 0.0
    while queued or pending:
        done: set[Any] = set()
        if pending:
            done, _ = wait(tuple(pending), timeout=0.5, return_when=FIRST_COMPLETED)
        for future in done:
            value, slot, pool = pending.pop(future)
            try:
                result = future.result()
            except BaseException as error:
                retry = _retry_specification(
                    value,
                    reason=f"{type(error).__name__}: {error}",
                    policy=policy,
                    telemetry=telemetry,
                    controller_error=error,
                )
                if retry is not None:
                    queued.append(retry)
                else:
                    failure = _controller_failure_result(value, error)
                    results.append(failure)
                    if telemetry is not None:
                        telemetry.run_finished(value, failure)
                    if on_result is not None:
                        queued.extend(on_result(failure, len(queued), len(pending)))
            else:
                retry = _retry_failed_result(
                    value,
                    result,
                    policy=policy,
                    telemetry=telemetry,
                )
                if retry is not None:
                    queued.append(retry)
                else:
                    results.append(result)
                    if on_result is not None:
                        queued.extend(on_result(result, len(queued), len(pending)))
                    print(
                        f"[hpo] finished trial={value['trial_index']} "
                        f"seed={value.get('seed')} on GPU {visible_gpus[slot]}; "
                        "worker exited and released its CUDA context",
                        flush=True,
                    )
            finally:
                # A long-lived CUDA worker retains its device context after the Run has
                # returned.  That idle context can keep enough VRAM reserved to prevent
                # the admission loop from ever launching the queued Runs.  One executor
                # per Run makes process exit the resource-release boundary.
                pool.shutdown(wait=True, cancel_futures=True)
                executors.remove(pool)
                active[slot] -= 1

        if policy.early_stopping and pending:
            _request_early_stops(
                tuple(value for value, _slot, _pool in pending.values()),
                metric=objective_metric,
                mode=objective_mode,
                min_step=policy.early_stopping_min_step,
                confirmations=policy.early_stopping_confirmations,
                probability_threshold=policy.seed_probability_threshold,
                margin=policy.equivalence_margin,
            )
        if telemetry is not None:
            telemetry.refresh()

        if not queued:
            continue

        now = time.monotonic()
        memory = (
            _gpu_memory_inventory(resources.gpu_count)
            if required > 0
            else tuple((0, 0) for _ in range(resources.gpu_count))
        )
        for index in usable:
            if active[index] == 0:
                admission_budget[index] = memory[index][0]
        slots = _admissible_gpu_slots(
            memory,
            active=active,
            last_launch=last_launch,
            admission_budget=admission_budget,
            required_bytes=required,
            runs_per_gpu=policy.runs_per_gpu,
            now=now,
            launch_stagger_seconds=_GPU_LAUNCH_STAGGER_SECONDS,
            usable=usable,
        )
        launched = False
        for slot in slots:
            if not queued or len(pending) >= parallelism:
                break
            value = queued.popleft()
            value["gpu_slot"] = visible_gpus[slot]
            value["gpu_index"] = slot
            pool = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_gpu_worker,
                initargs=(visible_gpus[slot],),
            )
            executors.append(pool)
            try:
                future = pool.submit(_execute_run, value)
            except BaseException:
                pool.shutdown(wait=True, cancel_futures=True)
                executors.remove(pool)
                raise
            pending[future] = (value, slot, pool)
            active[slot] += 1
            last_launch[slot] = now
            free, _total = memory[slot]
            memory_state = (
                f"free={_memory_text(free)} threshold={_memory_text(required)} " if required else ""
            )
            print(
                f"[hpo] admitted trial={value['trial_index']} seed={value.get('seed')} "
                f"on GPU {visible_gpus[slot]}: {memory_state}"
                f"active={active[slot]}/"
                f"{policy.runs_per_gpu}",
                flush=True,
            )
            launched = True

        if queued and not launched and now >= next_wait_log:
            states = ", ".join(
                f"GPU {visible_gpus[index]} "
                f"{'free=' + _memory_text(memory[index][0]) + ' ' if required else ''}"
                f"active={active[index]}/{policy.runs_per_gpu}"
                for index in usable
            )
            print(
                f"[hpo] waiting for safe GPU admission: {len(queued)} Run(s) pending; "
                f"{'need ' + _memory_text(required) + ' free per new Run; ' if required else ''}"
                f"{states}",
                flush=True,
            )
            next_wait_log = now + _GPU_WAIT_LOG_SECONDS
        if queued and not pending and not launched:
            time.sleep(_GPU_ADMISSION_POLL_SECONDS)


def _gpu_memory_inventory(gpu_count: int) -> tuple[tuple[int, int], ...]:
    """Read VRAM in a short-lived probe so the controller retains no CUDA contexts.

    Calling ``torch.cuda.mem_get_info`` in the long-lived HPO controller creates one visible CUDA
    process per device.  Besides confusing researchers, those otherwise idle contexts consume
    memory that can prevent the last packed Run from satisfying its admission threshold.  The
    child inherits the exact site grant and exits immediately after emitting bounded JSON.
    """
    script = (
        "import json, torch; "
        f"expected={gpu_count}; "
        "count=torch.cuda.device_count(); "
        "assert count >= expected, f'only {count} CUDA device(s) visible'; "
        "print(json.dumps([list(map(int, torch.cuda.mem_get_info(i))) "
        "for i in range(expected)]))"
    )
    try:
        completed = subprocess.run(  # noqa: S603
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        decoded = json.loads(completed.stdout)
        if not isinstance(decoded, list) or len(decoded) != gpu_count:
            raise ValueError("CUDA probe returned an unexpected device count.")
        observed = tuple(
            (int(item[0]), int(item[1]))
            for item in decoded
            if isinstance(item, list) and len(item) == 2
        )
        if len(observed) != gpu_count:
            raise ValueError("CUDA probe returned malformed memory records.")
        return observed
    except Exception as error:
        raise RuntimeError(
            "CUDA GPU admission could not read current device memory safely."
        ) from error


def _validate_gpu_memory_capacity(memory: Sequence[tuple[int, int]], required_bytes: int) -> None:
    """Reject only a bound that cannot ever fit any allocated physical device."""
    if required_bytes <= 0:
        return
    if not memory or all(required_bytes > total for _free, total in memory):
        totals = ", ".join(_memory_text(total) for _free, total in memory) or "none"
        raise ValueError(
            f"resources.gpu_memory={_memory_text(required_bytes)} exceeds total memory on every "
            f"allocated GPU ({totals}); no Run can ever be admitted."
        )


def _admissible_gpu_slots(
    memory: Sequence[tuple[int, int]],
    *,
    active: Sequence[int],
    last_launch: Sequence[float],
    admission_budget: Sequence[int],
    required_bytes: int,
    runs_per_gpu: int,
    now: float,
    launch_stagger_seconds: float,
    usable: Sequence[int],
) -> tuple[int, ...]:
    """Return devices that can accept one Run now without treating pressure as failure."""
    return tuple(
        index
        for index in usable
        if active[index] < runs_per_gpu
        and now - last_launch[index] >= launch_stagger_seconds
        and (required_bytes <= 0 or memory[index][0] >= required_bytes)
        and (required_bytes <= 0 or (active[index] + 1) * required_bytes <= admission_budget[index])
    )


def _memory_text(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f}{unit}" if unit not in {"B", "KiB"} else f"{amount:.0f}{unit}"
        amount /= 1024
    return f"{value}B"


def _adaptive_parallelism(resources: ResourceRequest, policy: AdaptiveSearchPolicy) -> int:
    derived = (
        resources.gpu_count * policy.runs_per_gpu
        if resources.gpu_count
        else policy.max_parallel or resources.cpu_cores
    )
    maximum = min(int(derived), policy.max_parallel) if policy.max_parallel else int(derived)
    return max(1, maximum)


def _visible_gpu_tokens(gpu_count: int) -> tuple[str, ...]:
    """Return only GPUs inherited from an external allocation, failing closed when required."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    mode = os.environ.get("LAMBDAFORGE_GPU_ACCESS_MODE", "")
    inherited = tuple(value.strip() for value in (raw or "").split(",") if value.strip())
    disabled = {"-1", "none", "nodevfiles", "void"}
    if inherited and any(value.lower() in disabled for value in inherited):
        inherited = ()
    if inherited:
        if len(set(inherited)) != len(inherited):
            raise RuntimeError("CUDA_VISIBLE_DEVICES contains duplicate allocated GPU tokens.")
        if len(inherited) < gpu_count:
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICES grants {len(inherited)} GPU(s), but the Work requests "
                f"{gpu_count}; refusing to invent or broaden the allocation."
            )
        return inherited[:gpu_count]
    if gpu_count > 0 and mode in {"command", "scheduler"}:
        raise RuntimeError(
            f"GPU access mode {mode!r} did not provide CUDA_VISIBLE_DEVICES; refusing to use "
            "unallocated physical GPU indices. Correct the site wrapper/scheduler environment."
        )
    return tuple(str(index) for index in range(gpu_count))


def _initialize_gpu_worker(slot: str) -> None:
    """Narrow one worker to a token already present in the parent allocation."""
    os.environ["CUDA_VISIBLE_DEVICES"] = slot


def _adaptive_run_resources(resources: ResourceRequest, parallelism: int) -> ResourceRequest:
    cpu = max(1, resources.cpu_cores // parallelism)
    return ResourceRequest(
        cpu_cores=cpu,
        ram_bytes=resources.ram_bytes // parallelism,
        gpu_count=1 if resources.gpu_count else 0,
        gpu_memory_bytes=resources.gpu_memory_bytes,
        runtime_seconds=resources.runtime_seconds,
        storage_bytes=resources.storage_bytes // parallelism,
        processes=min(cpu, resources.processes),
    )


def _candidate_score(
    results: Sequence[WorkResult], metric: str, mode: str, confidence: float
) -> float | None:
    values = [
        value
        for result in results
        if result.ok and not result.pruned
        for value in [_result_objective(result, metric)]
        if value is not None
    ]
    if not values:
        return None
    mean = statistics.fmean(values)
    if len(values) < 2 or confidence == 0:
        return mean
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    return (
        mean - confidence * standard_error if mode == "max" else mean + confidence * standard_error
    )


def _sample_uncertainty(
    results: Sequence[WorkResult], metric: str
) -> Mapping[str, int | float | None] | None:
    """Return compact empirical uncertainty without inventing certainty for one seed."""
    values = [
        value
        for result in results
        if result.ok and not result.pruned
        for value in [_result_objective(result, metric)]
        if value is not None
    ]
    if not values:
        return None
    return {
        "samples": len(values),
        "mean": statistics.fmean(values),
        "standard_error": (
            statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
        ),
    }


def _request_early_stops(
    specifications: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    mode: str,
    min_step: int,
    confirmations: int = 2,
    probability_threshold: float = 0.1,
    margin: float = 0.0,
) -> None:
    """Request pruning only when a curve is probably not practically competitive."""
    if confirmations < 1:
        raise ValueError("early-stopping confirmations must be positive.")
    histories: list[tuple[Mapping[str, Any], tuple[tuple[int, float], ...]]] = []
    for specification in specifications:
        history = _objective_history(Path(str(specification["hpo_metrics_path"])), metric)
        if history and history[-1][0] >= min_step:
            histories.append((specification, history))
    if len(histories) < 2:
        return
    common_step = min(history[-1][0] for _, history in histories)
    latest_values = [
        [value for step, value in history if step <= common_step][-1]
        for _, history in histories
        if any(step <= common_step for step, _ in history)
    ]
    pooled_deviation = statistics.stdev(latest_values) if len(latest_values) > 1 else 1.0
    target_step = common_step + max(1, min_step)
    observed: list[tuple[Mapping[str, Any], float, float]] = []
    for specification, history in histories:
        comparable = [(step, value) for step, value in history if step <= common_step]
        if comparable:
            mean, deviation = _curve_prediction(
                comparable,
                target_step=target_step,
                prior_deviation=max(pooled_deviation, 1e-6),
            )
            observed.append((specification, mean, deviation))
    if len(observed) < 2:
        return
    incumbent = (max if mode == "max" else min)(observed, key=lambda item: item[1])
    sign = 1.0 if mode == "max" else -1.0
    for specification, mean, deviation in observed:
        if specification is incumbent[0]:
            continue
        difference = sign * (mean - incumbent[1])
        difference_error = math.sqrt(deviation**2 + incumbent[2] ** 2)
        probability = (
            1.0
            if difference_error <= 0 and difference >= -margin
            else 0.0
            if difference_error <= 0
            else 1.0 - statistics.NormalDist(difference, difference_error).cdf(-margin)
        )
        stop = Path(str(specification["hpo_stop_path"]))
        evidence_path = stop.with_name(stop.name + ".evidence.json")
        if probability >= probability_threshold:
            evidence_path.unlink(missing_ok=True)
            continue
        prior: Mapping[str, Any] = {}
        if evidence_path.is_file() and not evidence_path.is_symlink():
            try:
                decoded = json.loads(evidence_path.read_text(encoding="utf-8"))
                prior = decoded if isinstance(decoded, Mapping) else {}
            except (OSError, json.JSONDecodeError):
                prior = {}
        prior_step = prior.get("common_step")
        prior_count = prior.get("confirmations")
        count = (
            int(prior_count) + 1
            if isinstance(prior_step, int)
            and prior_step < common_step
            and isinstance(prior_count, int)
            else int(prior_count)
            if prior_step == common_step and isinstance(prior_count, int)
            else 1
        )
        atomic_json(
            evidence_path,
            {
                "common_step": common_step,
                "confirmations": count,
                "required_confirmations": confirmations,
                "probability_competitive": probability,
                "threshold": probability_threshold,
            },
        )
        if count >= confirmations and not stop.exists():
            stop.parent.mkdir(parents=True, exist_ok=True)
            stop.write_text(
                "probabilistic adaptive pruning requested: "
                f"p_competitive={probability:.6f} threshold={probability_threshold:.6f} "
                f"common_step={common_step} confirmations={count}/{confirmations}\n",
                encoding="utf-8",
            )


def _curve_prediction(
    history: Sequence[tuple[int, float]],
    *,
    target_step: int,
    prior_deviation: float,
) -> tuple[float, float]:
    """Return a conservative local-linear predictive distribution for one learning curve."""
    recent = tuple(history[-5:])
    if len(recent) < 2:
        return float(recent[-1][1]), float(prior_deviation)
    steps = [float(item[0]) for item in recent]
    values = [float(item[1]) for item in recent]
    step_mean = statistics.fmean(steps)
    value_mean = statistics.fmean(values)
    denominator = sum((step - step_mean) ** 2 for step in steps)
    slope = (
        sum(
            (step - step_mean) * (value - value_mean)
            for step, value in zip(steps, values, strict=True)
        )
        / denominator
        if denominator > 0
        else 0.0
    )
    intercept = value_mean - slope * step_mean
    prediction = intercept + slope * target_step
    residuals = [
        value - (intercept + slope * step) for step, value in zip(steps, values, strict=True)
    ]
    residual_deviation = statistics.stdev(residuals) if len(residuals) > 2 else prior_deviation
    horizon = max(0.0, target_step - steps[-1]) / max(1.0, steps[-1] - steps[0])
    deviation = max(residual_deviation, prior_deviation * 0.1) * (1.0 + 0.25 * horizon)
    return prediction, deviation


def _objective_history(path: Path, metric: str) -> tuple[tuple[int, float], ...]:
    if not path.is_file() or path.is_symlink():
        return ()
    history: list[tuple[int, float]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            key = f"{value['split']}_{value['name']}" if value.get("split") else str(value["name"])
            step = value.get("step")
            if key == metric and isinstance(step, int):
                history.append((step, float(value["value"])))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return ()
    return tuple(sorted(history))


def _prune_reason(path: Path) -> str:
    """Return a compact persisted pruning explanation without making it resumable state."""
    try:
        detail = path.read_text(encoding="utf-8").strip()
    except OSError:
        detail = ""
    return f"adaptive-early-stopping: {detail}" if detail else "adaptive-early-stopping"


def _objective_observation(
    objective: Mapping[str, Any] | None,
    *,
    paths: Sequence[Path],
    fallback: Mapping[str, int | float],
) -> Mapping[str, Any] | None:
    """Separate current evidence from the best checkpoint used by HPO.

    Curve pruning deliberately consumes the comparable current curve elsewhere. Completed
    candidates, however, are ranked by the best observed validation checkpoint so a later
    overfitting epoch cannot silently make an otherwise strong Run look worse.
    """
    if not objective:
        return None
    metric, mode = str(objective["metric"]), str(objective["mode"])
    requested_constraints = objective.get("constraints", {})
    requested_constraints = (
        requested_constraints if isinstance(requested_constraints, Mapping) else {}
    )
    wanted = {metric, *(str(name) for name in requested_constraints)}
    by_metric: dict[str, list[tuple[int | None, int, float]]] = {name: [] for name in wanted}
    sequence = 0
    for path in paths:
        if not path.is_file() or path.is_symlink():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                value = json.loads(line)
                key = (
                    f"{value['split']}_{value['name']}"
                    if value.get("split")
                    else str(value["name"])
                )
                raw = value.get("value")
                if (
                    key not in wanted
                    or isinstance(raw, bool)
                    or not isinstance(raw, int | float)
                    or not math.isfinite(float(raw))
                ):
                    continue
                step = value.get("step")
                sequence += 1
                by_metric[key].append(
                    (
                        int(step) if isinstance(step, int) and not isinstance(step, bool) else None,
                        sequence,
                        float(raw),
                    )
                )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
    observations = by_metric[metric]
    if not observations:
        raw = fallback.get(metric)
        if (
            isinstance(raw, bool)
            or not isinstance(raw, int | float)
            or not math.isfinite(float(raw))
        ):
            return None
        observations.append((None, sequence + 1, float(raw)))
    stepped = [(step, order, value) for step, order, value in observations if step is not None]
    current_step, _current_order, current = (
        max(stepped, key=lambda item: (item[0], item[1])) if stepped else observations[-1]
    )
    best_step, _best_order, best = (min if mode == "min" else max)(
        observations,
        key=lambda item: item[2],
    )
    constraint_observations: dict[str, Any] = {}
    for name, raw_rule in requested_constraints.items():
        rule = raw_rule if isinstance(raw_rule, Mapping) else {}
        candidates = by_metric.get(str(name), [])
        matched = (
            [item for item in candidates if item[0] == best_step]
            if best_step is not None
            else [item for item in candidates if item[0] is None]
        )
        if not matched and best_step is None:
            fallback_value = fallback.get(str(name))
            if (
                isinstance(fallback_value, int | float)
                and not isinstance(fallback_value, bool)
                and math.isfinite(float(fallback_value))
            ):
                matched = [(None, sequence + 1, float(fallback_value))]
        selected = max(matched, key=lambda item: item[1]) if matched else None
        constraint_value = selected[2] if selected is not None else None
        constraint_observations[str(name)] = {
            "value": constraint_value,
            "step": selected[0] if selected is not None else best_step,
            "min": rule.get("min"),
            "max": rule.get("max"),
            "satisfied": _constraint_satisfied(constraint_value, rule),
        }
    output = {
        "metric": metric,
        "mode": mode,
        "current": current,
        "current_step": current_step,
        "best": best,
        "best_step": best_step,
        "selection": "best-observed-checkpoint",
    }
    if constraint_observations:
        output["constraints"] = constraint_observations
        output["feasible"] = all(
            bool(value["satisfied"]) for value in constraint_observations.values()
        )
    return output


def _constraint_satisfied(value: Any, rule: Mapping[str, Any]) -> bool:
    """Evaluate one explicit outcome bound; missing evidence fails closed."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        return False
    numeric = float(value)
    minimum = rule.get("min")
    maximum = rule.get("max")
    return not (
        isinstance(minimum, int | float)
        and not isinstance(minimum, bool)
        and numeric < float(minimum)
    ) and not (
        isinstance(maximum, int | float)
        and not isinstance(maximum, bool)
        and numeric > float(maximum)
    )


def _result_objective(result: WorkResult, metric: str) -> float | None:
    """Return the canonical best-observed value, with old-result compatibility."""
    observation = result.objective_observation
    if isinstance(observation, Mapping) and observation.get("metric") == metric:
        value = observation.get("best")
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    value = result.metrics.get(metric)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _candidate_constraints(
    results: Sequence[WorkResult], objective: Mapping[str, Any]
) -> dict[str, Any]:
    """Aggregate explicit outcome constraints across seeds at objective-best epochs.

    Constraints are candidate-level evidence: a lucky feasible seed cannot hide other completed
    seeds. Missing same-epoch evidence fails closed. The primary objective remains scalar, avoiding
    an implicit and irreproducible weighting of unrelated metrics.
    """
    raw_rules = objective.get("constraints", {})
    rules = raw_rules if isinstance(raw_rules, Mapping) else {}
    if not rules:
        return {"feasible": True, "constraints": {}}
    summary: dict[str, Any] = {}
    for name, raw_rule in rules.items():
        rule = raw_rule if isinstance(raw_rule, Mapping) else {}
        values: list[float] = []
        missing = 0
        for result in results:
            observation = result.objective_observation
            observation = observation if isinstance(observation, Mapping) else {}
            observed = observation.get("constraints", {})
            observed = observed if isinstance(observed, Mapping) else {}
            detail = observed.get(str(name), {})
            detail = detail if isinstance(detail, Mapping) else {}
            value = detail.get("value")
            if (
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ):
                values.append(float(value))
            else:
                missing += 1
        mean = statistics.fmean(values) if values and not missing else None
        satisfied = not missing and bool(values) and _constraint_satisfied(mean, rule)
        summary[str(name)] = {
            "mean": mean,
            "samples": len(values),
            "missing": missing,
            "min": rule.get("min"),
            "max": rule.get("max"),
            "satisfied": satisfied,
        }
    return {
        "feasible": all(bool(value["satisfied"]) for value in summary.values()),
        "constraints": summary,
    }


def _adopt_hpo_objective(runtime: WorkRuntime) -> None:
    """Bring a Lightning-bridged objective into the canonical Work metric result."""
    metric = os.environ.get("LAMBDAFORGE_HPO_OBJECTIVE")
    path = os.environ.get("LAMBDAFORGE_HPO_METRICS_PATH")
    if not metric or not path or metric in runtime.metrics.latest:
        return
    history = _objective_history(Path(path), metric)
    if history:
        step, value = history[-1]
        runtime.metrics.log(metric, value, step=step)


def _adopt_training_metrics(runtime: WorkRuntime, path: Path) -> None:
    """Copy only final training scalars into the canonical result summary."""
    if not path.is_file() or path.is_symlink():
        return
    latest: dict[str, float] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            metric = value.get("value")
            if not isinstance(metric, int | float) or isinstance(metric, bool):
                continue
            name = str(value["name"])
            key = f"{value['split']}_{name}" if value.get("split") else name
            latest[key] = float(metric)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return
    for name, value in latest.items():
        if name not in runtime.metrics.latest:
            runtime.metrics.log(name, value)


@contextmanager
def _scoped_environment(values: Mapping[str, str]) -> Any:
    prior = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in prior.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


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
        bool(value.get("pruned", False)),
        str(value["prune_reason"]) if value.get("prune_reason") is not None else None,
        int(value["gpu_index"]) if value.get("gpu_index") is not None else None,
        str(value["gpu_token"]) if value.get("gpu_token") is not None else None,
        str(value["study_phase"]) if value.get("study_phase") is not None else None,
        value.get("fidelity"),
        value.get("objective_observation"),
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
