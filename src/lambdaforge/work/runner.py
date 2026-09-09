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
from dataclasses import dataclass, replace
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
from lambdaforge.hpo.CurveEvidence import CompletedCurve, audit_pruner, predict_curve
from lambdaforge.hpo.ObjectiveUtility import (
    ObjectiveUtility,
    aggregate_constraint,
    constraint_satisfied,
    pareto_front,
)
from lambdaforge.hpo.ScientificDesign import (
    ExperimentalDesignPolicy,
    ScientificQuestionAnalyzer,
    SeedNoiseModel,
)
from lambdaforge.hpo.SurvivalModel import (
    SurvivalAcquisitionPolicy,
    SurvivalModel,
    SurvivalObservation,
)
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
_GPU_PROBE_FAILURE_LIMIT = 12
_PRUNER_AUDIT_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


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
                # Keep the candidate pool exactly once in ``RunDefinition``.  Embedding a fresh
                # copy of every candidate in every seed/candidate specification made adaptive
                # planning quadratic in the proposal-pool size (4096 candidates x 10 seeds used
                # tens of GiB before the first Run could start).  Child Runs need only know
                # whether this specification belongs to a parameterized trial.
                definition_payload = {
                    "name": definition.name,
                    "work_class": definition.work_class,
                    "resources": definition.resources.to_dict(),
                    "has_variants": definition.variants != ({},),
                    "objective": dict(definition.objective or {}),
                    "search_policy": (
                        definition.search_policy.to_dict()
                        if definition.search_policy is not None
                        else None
                    ),
                    "study_expected": definition.study_expected,
                }
                specifications: list[dict[str, Any]] = []
                for trial_index, variant, seed in self._expanded(definition):
                    specifications.append(
                        {
                            "definition": definition_payload,
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
        study_definitions = [
            definition
            for level in config.levels
            for definition in level.runs
            if definition.study_expected and definition.objective is not None
        ]
        if study_definitions:
            try:
                from lambdaforge.analysis.StudyAnalysis import StudyAnalysis

                StudyAnalysis.persist(
                    execution_result.to_dict(),
                    execution_dir / "analysis.json",
                    objective=study_definitions[0].objective,
                    authored_space=StudyAnalysis.authored_space(config.raw),
                    status="final",
                )
            except Exception as error:
                print(
                    f"[analysis] Final Study Analysis unavailable: {type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
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
        successful_all = [
            result
            for result in objective_runs
            if result.ok and not result.pruned and result.termination_type == "completed"
        ]
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
        censored_search_trials = {
            int((result.trial or {"index": 0})["index"])
            for result in objective_runs
            if result.study_phase != "confirmation"
            and result.termination_type == "performance_pruned"
        }
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
        confirmation_policies = [
            definition.search_policy
            for definition in adaptive_definitions
            if definition.name == objective_name and definition.search_policy is not None
        ]
        expected_confirmation_seeds = {
            seed for policy in confirmation_policies for seed in policy.confirmation_seeds
        }
        confirmation_attempts = [
            result for result in objective_runs if result.study_phase == "confirmation"
        ]
        confirmation_trial_ids = {
            int((result.trial or {"index": -1})["index"]) for result in confirmation_attempts
        }
        has_confirmation = bool(expected_confirmation_seeds or confirmation_attempts)
        for trial_index, results in sorted(grouped.items()):
            search_results = [result for result in results if result.study_phase != "confirmation"]
            confirmation_results = [
                result for result in results if result.study_phase == "confirmation"
            ]
            failed_confirmation = [
                result
                for result in confirmation_attempts
                if int((result.trial or {"index": -1})["index"]) == trial_index and not result.ok
            ]
            completed_confirmation_seeds = {result.seed for result in confirmation_results}
            confirmation_expected = trial_index in confirmation_trial_ids
            confirmation_complete = (
                confirmation_expected
                and bool(expected_confirmation_seeds)
                and (
                    completed_confirmation_seeds == expected_confirmation_seeds
                    and not failed_confirmation
                )
            )
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
                "confirmation_expected": confirmation_expected,
                "confirmation_complete": confirmation_complete,
                "confirmation_incomplete": confirmation_expected and not confirmation_complete,
                "confirmation_failures": [result.run_id for result in failed_confirmation],
                "partially_censored": trial_index in censored_search_trials,
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
            component_groups: dict[str, list[float]] = {}
            for result in evidence:
                observation = result.objective_observation or {}
                components = observation.get("components", {})
                if not isinstance(components, Mapping):
                    continue
                for name, raw_detail in components.items():
                    detail = raw_detail if isinstance(raw_detail, Mapping) else {}
                    raw = detail.get("raw")
                    if isinstance(raw, int | float) and not isinstance(raw, bool):
                        component_groups.setdefault(str(name), []).append(float(raw))
            if component_groups:
                candidate["raw_metrics"] = {
                    name: statistics.fmean(values) for name, values in component_groups.items()
                }
            candidates.append(candidate)
            if confirmation_complete and feasibility["feasible"]:
                selection_scores[trial_index] = value
            elif (
                not has_confirmation
                and feasibility["feasible"]
                and trial_index not in censored_search_trials
            ):
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
        pareto_trials = set(pareto_front(candidates, objective))
        for candidate in candidates:
            raw_trial = candidate["trial"]
            candidate["pareto_optimal"] = isinstance(raw_trial, int) and raw_trial in pareto_trials
        summary["candidates"] = candidates
        summary["best"] = selected
        if has_confirmation:
            complete_candidates = sum(
                bool(candidate.get("confirmation_complete")) for candidate in candidates
            )
            expected_candidates = min(
                max(
                    (policy.confirmation_top_k for policy in confirmation_policies),
                    default=0,
                ),
                len(candidates),
            )
            confirmation_incomplete = complete_candidates < expected_candidates
            summary["confirmation"] = {
                "status": "incomplete" if confirmation_incomplete else "complete",
                "confirmation_incomplete": confirmation_incomplete,
                "expected_seeds": sorted(expected_confirmation_seeds),
                "expected_candidates": expected_candidates,
                "attempted_runs": len(confirmation_attempts),
                "completed_runs": sum(result.ok for result in confirmation_attempts),
                "failed_runs": sum(not result.ok for result in confirmation_attempts),
                "complete_candidates": complete_candidates,
                "selection_available": selected is not None,
            }
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
    objective_config = specification.get("hpo_objective_config")
    if metrics_path is not None:
        environment["LAMBDAFORGE_HPO_METRICS_PATH"] = str(metrics_path)
    if stop_path is not None:
        environment["LAMBDAFORGE_STOP_REQUEST_PATH"] = str(stop_path)
    if objective is not None:
        environment["LAMBDAFORGE_HPO_OBJECTIVE"] = str(objective)
    if isinstance(objective_config, Mapping):
        environment["LAMBDAFORGE_HPO_OBJECTIVE_CONFIG"] = json.dumps(
            objective_config, sort_keys=True, separators=(",", ":")
        )
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
    raw_checkpoint_manifest = specification.get("hpo_checkpoint_manifest_path")
    if raw_checkpoint_manifest is not None:
        atomic_json(
            Path(str(raw_checkpoint_manifest)),
            {"checkpoint_root": str(checkpoint_root.resolve()), "run_id": run_id},
        )
    resuming = any(path.is_file() for path in checkpoint_root.rglob("*"))
    resources = _work_resources(definition.resources.to_dict())
    config_view = WorkConfiguration(
        definition.name,
        definition.work_class,
        parameters,
        resources,
    )
    trial_parameters = dict(specification.get("trial_parameters", {}))
    trial = (
        WorkTrial(trial_index, trial_parameters)
        if bool(definition_value.get("has_variants", False))
        else None
    )
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
    stop_path = (
        Path(str(specification["hpo_stop_path"])) if specification.get("hpo_stop_path") else None
    )
    termination_type, termination = _run_termination(
        status=status,
        failure=failure,
        stop_path=stop_path,
        completed_step=(
            max(
                (
                    step
                    for step, _value in _objective_history(
                        Path(str(specification["hpo_metrics_path"])),
                        str(specification.get("hpo_objective", "")),
                    )
                ),
                default=None,
            )
            if specification.get("hpo_metrics_path") and specification.get("hpo_objective")
            else None
        ),
        target_step=fidelity.target if fidelity is not None else None,
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
        pruned=termination_type == "performance_pruned",
        prune_reason=(
            _prune_reason(Path(str(specification["hpo_stop_path"])))
            if termination_type == "performance_pruned"
            and specification.get("hpo_stop_path")
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
        termination_type=termination_type,
        termination=termination,
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
    objective_evaluator = ObjectiveUtility(objective)
    parallelism = _adaptive_parallelism(resources, policy)
    by_trial: dict[int, list[Mapping[str, Any]]] = {}
    for specification in specifications:
        by_trial.setdefault(int(specification["trial_index"]), []).append(specification)
    all_trials = sorted(by_trial)
    candidate_budget = min(policy.candidate_budget, len(all_trials))
    startup_trial_count = _adaptive_startup_trial_count(
        policy,
        parallelism=parallelism,
        candidate_budget=candidate_budget,
    )
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
            "event_version": 2,
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
            planned_runs=candidate_budget * seed_count
            + min(policy.confirmation_top_k, candidate_budget) * len(policy.confirmation_seeds),
            planned_candidates=candidate_budget,
        )
    print(
        f"[hpo] action-adaptive search: candidate_budget={candidate_budget} "
        f"proposal_pool={len(all_trials)} startup={startup_trial_count} "
        f"seeds={seed_count} "
        f"parallel={parallelism} runs_per_gpu={policy.runs_per_gpu}",
        flush=True,
    )
    record_decision(
        "INITIALIZE",
        candidate_budget=candidate_budget,
        proposal_pool_size=len(all_trials),
        startup_trials_configured=policy.startup_trials,
        startup_trials_effective=startup_trial_count,
        search_seed_budget=seed_count,
        slots_total=parallelism,
        policy=policy.to_dict(),
        controller_value_policy={
            "meaning": "automatic-performance-and-scientific-value-per-observed-cost",
            "manual_weights": False,
        },
        survival_acquisition=SurvivalAcquisitionPolicy().to_dict(),
    )
    candidate_parameters = {
        trial: dict(by_trial[trial][0].get("trial_parameters", {})) for trial in all_trials
    }
    selector = AdaptiveSampler(candidate_parameters, mode=mode)
    bayesian = BayesianSampler(candidate_parameters, mode=mode)
    survival_model = SurvivalModel(candidate_parameters)
    racer = AdaptiveSeedRacer(
        mode=mode,
        margin=policy.equivalence_margin,
        probability_threshold=policy.seed_probability_threshold,
    )
    proposed: list[int] = []
    proposal_numbers: dict[int, int] = {}
    pool_trials_by_proposal: dict[int, int] = {}
    inflight_by_trial: dict[int, int] = {}
    pending_fidelity_by_trial: dict[int, float] = {}
    pending_observations: list[tuple[int, float]] = []
    scheduled_run_keys: set[tuple[int, int | None, str, int]] = set()

    def scheduling_key(specification: Mapping[str, Any]) -> tuple[int, int | None, str, int]:
        raw_fidelity = specification.get("hpo_fidelity")
        target = int(raw_fidelity.get("target", 0)) if isinstance(raw_fidelity, Mapping) else 0
        return (
            int(specification.get("candidate_pool_index", specification["trial_index"])),
            int(specification["seed"]) if specification.get("seed") is not None else None,
            str(specification.get("hpo_phase", "search")),
            target,
        )

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

    def next_seed_specification(
        pool_trial: int, *, preferred_seed: int | None = None
    ) -> dict[str, Any] | None:
        attempted = {result.seed for result in completed[pool_trial]}
        available = list(by_trial[pool_trial])
        if preferred_seed is not None:
            available.sort(key=lambda value: value.get("seed") != preferred_seed)
        for raw in available:
            value = prepared(pool_trial, raw)
            if raw.get("seed") not in attempted and scheduling_key(value) not in scheduled_run_keys:
                return value
        return None

    def resume_specification(result: WorkResult) -> dict[str, Any] | None:
        if policy.fidelity is None or result.fidelity is None or not result.ok or result.pruned:
            return None
        preempted = result.termination_type == "scheduler_preempted"
        raw_observed = result.termination.get("observed_step")
        current = (
            int(raw_observed)
            if preempted and isinstance(raw_observed, int) and not isinstance(raw_observed, bool)
            else int(result.fidelity["target"])
        )
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
        target = (
            int(result.fidelity["target"])
            if preempted
            else min(
                policy.fidelity.maximum,
                max(current + 1, current * policy.fidelity.reduction_factor),
            )
        )
        if target <= current:
            return None
        value["hpo_fidelity"] = {
            "current": current,
            "target": target,
            "maximum": policy.fidelity.maximum,
        }
        if preempted:
            value["hpo_resume_preempted"] = True
        if scheduling_key(value) in scheduled_run_keys:
            return None
        return value

    def pool_trial_for(result: WorkResult) -> int:
        public_trial = int((result.trial or {"index": 0})["index"])
        return pool_trials_by_proposal[public_trial]

    def values_by_trial(*, final_only: bool = True) -> dict[int, dict[int | None, float]]:
        """Return one exact comparable rung for seed racing or final selection."""
        grouped = _candidate_values_by_rung(completed, metric=metric, objective=objective)
        eligible_rungs = [rung for rung in grouped if not final_only or rung[0] >= rung[1]]
        if not eligible_rungs:
            return {}
        # A new seed always starts at the lowest authored rung, so ADD_SEED comparisons use that
        # exact rung. Final ranking uses the full-budget rung. No heterogeneous values are merged.
        selected_rung = (max if final_only else min)(
            eligible_rungs, key=lambda rung: (rung[0] / max(1, rung[1]), rung)
        )
        return grouped[selected_rung]

    def observations() -> list[CandidateObservation]:
        return list(
            _candidate_observations(
                completed,
                metric=metric,
                objective=objective,
                racer=racer,
            )
        )

    def scientific_snapshot() -> dict[str, Any]:
        candidates: list[dict[str, Any]] = []
        for pool_trial in proposed:
            runs = []
            for result in completed[pool_trial]:
                runs.append(
                    {
                        "seed": result.seed,
                        "phase": result.study_phase,
                        "state": (
                            "pruned"
                            if result.termination_type == "performance_pruned"
                            else result.status
                        ),
                        "final_objective": (
                            _result_objective(result, metric)
                            if result.termination_type == "completed"
                            else None
                        ),
                        "best_observed_objective": _result_objective(result, metric),
                        "fidelity": dict(result.fidelity or {}),
                        "censored": result.termination_type == "performance_pruned",
                    }
                )
            candidates.append(
                {
                    "trial": pool_trial,
                    "parameters": candidate_parameters[pool_trial],
                    "runs": runs,
                    "state": (
                        "pruned"
                        if runs and all(value["state"] == "pruned" for value in runs)
                        else "observed"
                    ),
                }
            )
        return ScientificQuestionAnalyzer.analyze(
            candidates,
            objective,
            practical_margin=policy.scientific_margin,
            fingerprint=str(specifications[0]["execution_id"]),
            candidate_pool=candidate_parameters,
        )

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
        survival_observations: list[SurvivalObservation] = []
        for trial in proposed:
            scientific = [
                result
                for result in completed[trial]
                if result.termination_type in {"completed", "performance_pruned"}
            ]
            if not scientific:
                continue
            candidate_weight = 1.0 / len(scientific)
            for result in scientific:
                fidelity = result.fidelity or {}
                target = int(fidelity.get("target", 1))
                maximum = max(1, int(fidelity.get("maximum", target)))
                evidence_weight = 1.0
                if result.termination_type == "performance_pruned":
                    termination = result.termination
                    probability = termination.get("probability_competitive")
                    threshold = termination.get("threshold")
                    observed_confirmations = termination.get("confirmations")
                    required_confirmations = termination.get("required_confirmations")
                    probability_strength = (
                        min(
                            1.0,
                            max(0.1, 1.0 - float(probability) / max(float(threshold), 1e-9)),
                        )
                        if isinstance(probability, int | float)
                        and not isinstance(probability, bool)
                        and isinstance(threshold, int | float)
                        and not isinstance(threshold, bool)
                        else 0.5
                    )
                    confirmation_strength = (
                        min(1.0, float(observed_confirmations) / float(required_confirmations))
                        if isinstance(observed_confirmations, int | float)
                        and not isinstance(observed_confirmations, bool)
                        and isinstance(required_confirmations, int | float)
                        and not isinstance(required_confirmations, bool)
                        and float(required_confirmations) > 0
                        else 0.5
                    )
                    evidence_weight = probability_strength * confirmation_strength
                survival_observations.append(
                    SurvivalObservation(
                        trial,
                        result.termination_type == "completed",
                        fidelity=target / maximum,
                        weight=candidate_weight * evidence_weight,
                    )
                )
        survival = survival_model.predict_all(survival_observations)
        use_bayesian = policy.sampler == "botorch" or (
            policy.sampler == "auto" and BayesianSampler.available()
        )
        if use_bayesian:
            try:
                selected = bayesian.propose(
                    evidence,
                    selected=proposed,
                    pending=pending_trials,
                    pending_fidelity=pending_fidelity_by_trial,
                    pending_observations=pending_observations,
                    censored=censored_trials,
                    survival=survival,
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
                        survival_evidence=len(survival_observations),
                        surrogate_belief=dict(bayesian.last_diagnostics),
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
            pending_fidelity=pending_fidelity_by_trial,
            pending_observations=pending_observations,
            censored=censored_trials,
            survival=survival,
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
                survival_evidence=len(survival_observations),
                surrogate_belief=dict(selector.last_diagnostics),
            )
        return selected

    started = time.monotonic()
    best_value: float | None = None
    stale_events = 0
    search_converged = False

    def allowance() -> int:
        if policy.max_runs is None:
            return 2**31 - 1
        return max(0, policy.max_runs - len(outcomes))

    def within_time() -> bool:
        return (
            policy.max_time_seconds is None or time.monotonic() - started < policy.max_time_seconds
        )

    def persist_state() -> None:
        scientific = scientific_snapshot()
        atomic_json(
            state_path,
            {
                "state_version": 2,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "proposed_pool_trials": list(proposed),
                "public_trial_map": {
                    str(pool_trial): public for pool_trial, public in proposal_numbers.items()
                },
                "completed_runs": len(outcomes),
                "pruner_calibration": (
                    _pruner_calibration(
                        outcomes,
                        objective_evaluator,
                        min_step=policy.early_stopping_min_step,
                        confirmations=policy.early_stopping_confirmations,
                        probability_threshold=policy.early_stopping_probability_threshold,
                        margin=policy.early_stopping_equivalence_margin,
                    )
                    if policy.early_stopping
                    else None
                ),
                "scientific_understanding": scientific,
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

    def next_event_action(capacity: int) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        """Choose one action for a newly free slot from all currently useful action types."""
        if capacity < 1 or not within_time():
            return "WAIT", [], {"reason": "budget-or-time-exhausted"}
        durations = [
            result.duration_seconds
            for result in outcomes
            if result.duration_seconds > 0 and result.termination_type != "scheduler_preempted"
        ]
        default_cost = statistics.fmean(durations) if durations else 1.0
        options: list[tuple[float, int, str, dict[str, Any], dict[str, Any]]] = []
        scientific = scientific_snapshot()
        optimization_weight = float(scientific["optimization_weight"])
        information_weight = float(scientific["information_weight"])

        provisional_values = values_by_trial(final_only=False)
        eligible = [trial for trial in proposed if next_seed_specification(trial) is not None]
        # Do not spend an extra seed while a peer at the comparison rung is already in flight.
        # This is an event-driven wait, not a round barrier: unrelated new candidates may still
        # fill a free slot below, and the next completion immediately replans every queued action.
        seed_decisions = (
            racer.decisions(
                provisional_values,
                eligible=eligible,
                available_seeds={
                    trial: [
                        int(value["seed"])
                        for value in by_trial[trial]
                        if isinstance(value.get("seed"), int)
                        and not isinstance(value.get("seed"), bool)
                    ]
                    for trial in eligible
                },
            )
            if len(provisional_values) >= 2 or not pending_observations
            else ()
        )
        seed_noise = SeedNoiseModel.fit(provisional_values)
        calibration_inflight = not seed_noise.calibrated and any(
            inflight_by_trial.get(trial, 0) > 0 and bool(values)
            for trial, values in provisional_values.items()
        )
        for decision in seed_decisions:
            # One repeated candidate is sufficient to identify the initial within-candidate noise
            # scale.  If that calibration is already running, do not spend another slot repeating
            # an arbitrary (possibly poor) candidate before its evidence arrives.
            if decision.purpose == "CALIBRATE_SEED_NOISE" and calibration_inflight:
                continue
            specification = next_seed_specification(
                decision.trial, preferred_seed=decision.recommended_seed
            )
            if specification is None:
                continue
            candidate_costs = [
                result.duration_seconds
                for result in completed[decision.trial]
                if result.duration_seconds > 0
            ]
            cost = statistics.fmean(candidate_costs) if candidate_costs else default_cost
            information_value = min(1.0, max(0.0, decision.information_value))
            optimization_value = information_value * max(
                decision.probability_competitive,
                1.0 - decision.probability_competitive,
            )
            cost_ratio = cost / max(default_cost, 1e-9)
            value_proxy = (
                optimization_weight * optimization_value + information_weight * information_value
            )
            options.append(
                (
                    value_proxy / max(cost_ratio, 1e-9),
                    1,
                    decision.purpose,
                    specification,
                    {
                        "purpose": decision.purpose,
                        "probability_competitive": decision.probability_competitive,
                        "current_standard_error": (
                            decision.current_standard_error
                            if math.isfinite(decision.current_standard_error)
                            else None
                        ),
                        "expected_uncertainty_reduction": (decision.expected_uncertainty_reduction),
                        "optimization_value": optimization_value,
                        "information_value": information_value,
                        "controller_value": value_proxy,
                        "value_basis": ("incumbent-challenger-comparison-uncertainty-reduction"),
                        "expected_cost_seconds": cost,
                        "cost_ratio": cost_ratio,
                        "completed_seeds": decision.completed_seeds,
                        "recommended_shared_seed": decision.recommended_seed,
                        "comparison_trials": list(decision.comparison_trials),
                        "target_questions": [
                            f"Which of Trial {decision.trial} and its challengers is "
                            "practically competitive?"
                        ],
                        "reason": (
                            "calibrate within-candidate seed noise without using "
                            "between-candidate spread"
                            if decision.purpose == "CALIBRATE_SEED_NOISE"
                            else "reduce uncertainty in incumbent/challenger comparisons "
                            "using paired evidence"
                            if decision.purpose == "ADD_SHARED_SEED"
                            else "reduce uncertainty in the incumbent before declaring it stable"
                            if decision.purpose == "REPLICATE_INCUMBENT"
                            else "reduce uncertainty in a practically relevant candidate comparison"
                        ),
                    },
                )
            )

        comparable_rungs = _candidate_values_by_rung(completed, metric=metric, objective=objective)
        pending_fidelities = {fidelity for _trial, fidelity in pending_observations}
        provisional = tuple(
            decision
            for (target, maximum), rung_values in comparable_rungs.items()
            if len(rung_values) >= 2 or target / max(1, maximum) not in pending_fidelities
            for decision in racer.decisions(rung_values, eligible=proposed)
        )
        competitive = {decision.trial for decision in provisional}
        if seed_count == 1 and comparable_rungs:
            # With one explicitly authored seed, that shared seed is the complete estimand rather
            # than an uncalibrated population mean. Promote only the best practical set at a rung;
            # when more seeds exist, the conservative probabilistic racer remains authoritative.
            latest_rung = max(
                comparable_rungs,
                key=lambda rung: (rung[0] / max(1, rung[1]), rung),
            )
            rung_values = comparable_rungs[latest_rung]
            rung_fraction = latest_rung[0] / max(1, latest_rung[1])
            # Do not promote the only completed candidate while a peer at the same rung is still
            # running.  This is a local comparison wait, not a global round barrier: unrelated
            # candidate and seed actions may still fill other slots.
            rung_means = (
                {
                    trial: statistics.fmean(values.values())
                    for trial, values in rung_values.items()
                    if values
                }
                if len(rung_values) >= 2 or rung_fraction not in pending_fidelities
                else {}
            )
            if rung_means:
                best_rung_value = (max if mode == "max" else min)(rung_means.values())
                margin = policy.scientific_margin or 0.0
                competitive = {
                    trial
                    for trial, value in rung_means.items()
                    if (best_rung_value - value if mode == "max" else value - best_rung_value)
                    <= margin
                }
        latest_by_seed: dict[tuple[int, int | None], WorkResult] = {}
        for trial in proposed:
            for result in completed[trial]:
                key = (trial, result.seed)
                target = int((result.fidelity or {}).get("target", 2**31 - 1))
                previous = latest_by_seed.get(key)
                if previous is None or target >= int((previous.fidelity or {}).get("target", -1)):
                    latest_by_seed[key] = result
                if result.termination_type == "scheduler_preempted":
                    competitive.add(trial)
        for (trial, _seed), result in latest_by_seed.items():
            if trial not in competitive:
                continue
            specification = resume_specification(result)
            if specification is None:
                continue
            fidelity = specification["hpo_fidelity"]
            current, target = int(fidelity["current"]), int(fidelity["target"])
            incremental_fraction = max(1, target - current) / max(1, current)
            base_cost = result.duration_seconds if result.duration_seconds > 0 else default_cost
            cost = max(1e-9, base_cost * incremental_fraction)
            maximum = max(1, int(fidelity["maximum"]))
            fidelity_gain = max(0.0, min(1.0, (target - current) / maximum))
            information_value = min(
                1.0,
                float(scientific["scientific_uncertainty"]) * math.sqrt(fidelity_gain),
            )
            optimization_value = min(1.0, math.sqrt(fidelity_gain))
            combined_value = (
                optimization_weight * optimization_value + information_weight * information_value
            )
            value_proxy = combined_value
            cost_ratio = cost / max(default_cost, 1e-9)
            options.append(
                (
                    combined_value / max(cost_ratio, 1e-9),
                    2,
                    (
                        "RESUME_PREEMPTED"
                        if specification.get("hpo_resume_preempted")
                        else "PROMOTE_FIDELITY"
                    ),
                    specification,
                    {
                        "controller_value": value_proxy,
                        "optimization_value": optimization_value,
                        "information_value": information_value,
                        "value_basis": "fidelity-evidence-value-under-current-study-balance",
                        "expected_cost_seconds": cost,
                        "cost_ratio": cost_ratio,
                        "current": current,
                        "target": target,
                        "purpose": "PROMOTE_FIDELITY",
                        "target_questions": [],
                        "reason": (
                            "higher-fidelity evidence remains competitive per incremental cost"
                        ),
                    },
                )
            )

        proposal: tuple[int, ...] = ()
        if len(proposed) < candidate_budget and not search_converged:
            proposal = propose(1)
        if len(proposed) < candidate_budget and provisional_values and not search_converged:
            costs_by_trial = {
                trial: statistics.fmean(
                    result.duration_seconds
                    for result in completed[trial]
                    if result.duration_seconds > 0
                )
                for trial in proposed
                if any(result.duration_seconds > 0 for result in completed[trial])
            }
            designed = ExperimentalDesignPolicy(
                candidate_parameters,
                mode=mode,
                practical_margin=policy.scientific_margin,
            ).rank(
                provisional_values,
                selected=proposed,
                scientific_state=scientific,
                costs=costs_by_trial,
            )
            candidates_to_value = list(designed[:1])
            if proposal:
                optimization_candidate = next(
                    (value for value in designed if value.trial == proposal[0]), None
                )
                if optimization_candidate is not None and all(
                    value.trial != optimization_candidate.trial for value in candidates_to_value
                ):
                    candidates_to_value.append(optimization_candidate)
            for design in candidates_to_value:
                specification = initial_specifications(design.trial)[0]
                specification["hpo_probe_purpose"] = design.purpose
                specification["hpo_target_questions"] = list(design.target_questions)
                evidence = design.to_dict()
                evidence["pool_trial"] = design.trial
                options.append(
                    (
                        design.score,
                        0,
                        str(evidence["action"]),
                        specification,
                        evidence,
                    )
                )
        elif proposal:
            # No comparable outcome exists yet; preserve deterministic space-filling startup.
            specification = initial_specifications(proposal[0])[0]
            options.append(
                (
                    1.0,
                    0,
                    "START_NEW",
                    specification,
                    {
                        "purpose": "EXPLORE_COVERAGE",
                        "optimization_value": None,
                        "information_value": None,
                        "expected_cost_seconds": default_cost,
                        "cost_ratio": 1.0,
                        "pool_trial": proposal[0],
                        "target_questions": [],
                        "reason": "collect the first comparable objective evidence",
                    },
                )
            )
        options = [value for value in options if value[0] > 1e-12]
        if not options:
            return "WAIT", [], {"reason": "no-scientifically-useful-action"}
        score, _tie, action, specification, evidence = max(
            options, key=lambda value: (value[0], -value[1], -int(value[3]["trial_index"]))
        )
        if action in {"START_NEW", "DESIGNED_PROBE"}:
            pool_trial = int(evidence["pool_trial"])
            proposed.append(pool_trial)
            specifications = initial_specifications(pool_trial)[:capacity]
            if action == "DESIGNED_PROBE":
                for value in specifications:
                    value["hpo_probe_purpose"] = evidence.get("purpose")
                    value["hpo_target_questions"] = list(evidence.get("target_questions", ()))
        else:
            specifications = [specification]
        alternatives = [
            {
                "action": value[2],
                "trial": int(value[3]["trial_index"]),
                "score": value[0],
                "controller_value": value[4].get("controller_value"),
                "optimization_value": value[4].get("optimization_value"),
                "information_value": value[4].get("information_value"),
                "expected_cost_seconds": value[4].get("expected_cost_seconds"),
            }
            for value in sorted(options, key=lambda item: item[0], reverse=True)[:6]
        ]
        return (
            action,
            specifications,
            {
                **evidence,
                "score": score,
                "alternatives": alternatives,
                "optimization_opportunity": scientific["optimization_opportunity"],
                "scientific_uncertainty": scientific["scientific_uncertainty"],
                "optimization_weight": optimization_weight,
                "information_weight": information_weight,
                "study_phase": scientific["phase"],
                "reason": evidence.get(
                    "reason", "highest combined performance-and-information value per cost"
                ),
                "pending_context": [
                    {"trial": trial, "target_fidelity": fidelity}
                    for trial, fidelity in sorted(pending_fidelity_by_trial.items())[:12]
                ],
            },
        )

    def execute(
        scheduled: Sequence[dict[str, Any]],
        observed_trials: Sequence[int],
        *,
        deferred: Sequence[dict[str, Any]] = (),
    ) -> None:
        limited = list(scheduled[: allowance()])
        if not limited:
            return
        deferred_queue = deque(dict(value) for value in deferred)

        def register(values: Sequence[dict[str, Any]]) -> None:
            if telemetry is not None:
                telemetry.schedule(values)
            for value in values:
                scheduled_run_keys.add(scheduling_key(value))
                trial = int(value.get("candidate_pool_index", value["trial_index"]))
                inflight_by_trial[trial] = inflight_by_trial.get(trial, 0) + 1

        register(limited)

        def observed(
            result: WorkResult,
            queued_specifications: Sequence[Mapping[str, Any]],
            pending_specifications: Sequence[Mapping[str, Any]],
        ) -> Sequence[dict[str, Any]]:
            nonlocal best_value, search_converged, stale_events
            pending_runs = len(pending_specifications)
            pending_fidelity_by_trial.clear()
            pending_observations.clear()
            for pending in pending_specifications:
                pending_trial = int(pending.get("candidate_pool_index", pending["trial_index"]))
                raw_fidelity = pending.get("hpo_fidelity")
                pending_fraction = (
                    int(raw_fidelity.get("target", 1)) / max(1, int(raw_fidelity.get("maximum", 1)))
                    if isinstance(raw_fidelity, Mapping)
                    else 1.0
                )
                pending_fidelity_by_trial[pending_trial] = max(
                    pending_fraction, pending_fidelity_by_trial.get(pending_trial, 0.0)
                )
                pending_observations.append((pending_trial, pending_fraction))
            trial = pool_trial_for(result)
            if result.termination_type == "scheduler_preempted":
                scheduled_run_keys.discard(
                    (
                        trial,
                        result.seed,
                        str(result.study_phase or "search"),
                        int((result.fidelity or {}).get("target", 0)),
                    )
                )
            inflight_by_trial[trial] = max(0, inflight_by_trial.get(trial, 1) - 1)
            completed[trial].append(result)
            outcomes.append(result)
            result_fidelity = result.fidelity or {}
            reached_final_fidelity = int(result_fidelity.get("target", 1)) >= int(
                result_fidelity.get("maximum", 1)
            )
            if (
                result.termination_type == "completed"
                and reached_final_fidelity
                and result.study_phase != "confirmation"
            ):
                final_estimates = racer.estimates(values_by_trial())
                if final_estimates:
                    current = (
                        max(value.mean for value in final_estimates.values())
                        if mode == "max"
                        else min(value.mean for value in final_estimates.values())
                    )
                    improvement = (
                        math.inf
                        if best_value is None
                        else current - best_value
                        if mode == "max"
                        else best_value - current
                    )
                    if improvement > policy.min_improvement:
                        best_value, stale_events = current, 0
                    else:
                        stale_events += 1
                    if (
                        policy.convergence_patience
                        and stale_events >= policy.convergence_patience
                        and not search_converged
                    ):
                        search_converged = True
                        record_decision(
                            "STOP_PROPOSING",
                            reason="full-fidelity-evidence-convergence",
                            stale_evidence_events=stale_events,
                            best_objective=best_value,
                            minimum_improvement=policy.min_improvement,
                        )
            persist_state()
            if result.termination_type == "scheduler_preempted":
                record_decision(
                    "PAUSE",
                    trial=int((result.trial or {"index": 0})["index"]),
                    seed=result.seed,
                    observed_step=result.termination.get("observed_step"),
                    reason=result.termination.get("reason"),
                    checkpoint_required=bool(result.termination.get("checkpoint_required")),
                )
            elif result.termination_type == "performance_pruned":
                prune_event = dict(result.termination)
                prune_event.setdefault("trial", int((result.trial or {"index": 0})["index"]))
                prune_event.setdefault("seed", result.seed)
                record_decision(
                    "PERFORMANCE_PRUNE",
                    **prune_event,
                )
            if result.study_phase == "confirmation":
                return [dict(value) for value in queued_specifications]
            stale_queue = [dict(value) for value in queued_specifications]
            for stale in stale_queue:
                scheduled_run_keys.discard(scheduling_key(stale))
                stale_trial = int(stale.get("candidate_pool_index", stale["trial_index"]))
                inflight_by_trial[stale_trial] = max(0, inflight_by_trial.get(stale_trial, 1) - 1)
            if result.termination_type == "performance_pruned" and deferred_queue:
                retained: deque[dict[str, Any]] = deque()
                while deferred_queue:
                    deferred_value = deferred_queue.popleft()
                    deferred_trial = int(
                        deferred_value.get("candidate_pool_index", deferred_value["trial_index"])
                    )
                    if deferred_trial != trial:
                        retained.append(deferred_value)
                        continue
                    if telemetry is not None:
                        telemetry.queued_action_cancelled(
                            deferred_value,
                            reason=(
                                "candidate received a candidate-level performance-prune decision"
                            ),
                        )
                    record_decision(
                        "CANCEL_QUEUED_ACTION",
                        reason="candidate-level-performance-prune",
                        trial=int(deferred_value["trial_index"]),
                        seed=deferred_value.get("seed"),
                        started=False,
                        compute_seconds=0.0,
                    )
                deferred_queue.extend(retained)
            deferred_keys = {scheduling_key(value) for value in deferred_queue}
            for stale in stale_queue:
                stale_trial = int(stale.get("candidate_pool_index", stale["trial_index"]))
                key = scheduling_key(stale)
                if (
                    not completed[stale_trial]
                    and stale.get("hpo_phase", "search") == "search"
                    and key not in deferred_keys
                ):
                    deferred_queue.append(stale)
                    deferred_keys.add(key)
            if not within_time():
                return ()
            remaining_capacity = allowance() - pending_runs
            if remaining_capacity <= 0:
                return ()
            # Startup is space-filling evidence, not a barrier. Alternate deferred startup work
            # with model-driven actions as soon as at least two outcomes exist.
            use_deferred = bool(deferred_queue) and (len(outcomes) < 2 or len(outcomes) % 2 == 1)
            evidence: dict[str, Any]
            if use_deferred:
                specifications = [deferred_queue.popleft()]
                action = "START_NEW"
                evidence = {
                    "reason": "asynchronous-space-filling-startup",
                    "controller_value": None,
                    "value_basis": "required-space-filling-coverage",
                    "expected_cost_seconds": None,
                    "score": None,
                    "alternatives": [],
                }
            else:
                action, specifications, evidence = next_event_action(remaining_capacity)
                if not specifications and deferred_queue:
                    specifications = [deferred_queue.popleft()]
                    action = "START_NEW"
                    evidence = {
                        "reason": "remaining-space-filling-startup",
                        "alternatives": [],
                    }
            specifications = list(specifications[:remaining_capacity])
            if not specifications:
                for stale in stale_queue:
                    if telemetry is not None:
                        telemetry.queued_action_cancelled(
                            stale, reason="no longer useful after new HPO evidence"
                        )
                    record_decision(
                        "CANCEL_QUEUED_ACTION",
                        reason="superseded-by-new-evidence",
                        trial=int(stale["trial_index"]),
                        seed=stale.get("seed"),
                        phase=stale.get("hpo_phase", "search"),
                        started=False,
                        compute_seconds=0.0,
                        old_priority=stale.get("hpo_scheduler_priority"),
                        new_best_action="WAIT",
                        new_priority=None,
                        new_evidence={"termination_type": result.termination_type},
                    )
                return ()
            selected_keys = {scheduling_key(value) for value in specifications}
            retained_deferred: deque[dict[str, Any]] = deque()
            while deferred_queue:
                deferred_value = deferred_queue.popleft()
                if scheduling_key(deferred_value) not in selected_keys:
                    retained_deferred.append(deferred_value)
            deferred_queue.extend(retained_deferred)
            action_priority = evidence.get("score")
            for specification in specifications:
                specification["hpo_scheduler_action"] = action
                if isinstance(action_priority, int | float) and not isinstance(
                    action_priority, bool
                ):
                    specification["hpo_scheduler_priority"] = float(action_priority)
            for stale in stale_queue:
                if scheduling_key(stale) in selected_keys:
                    continue
                cancellation = {
                    "reason": "superseded-by-new-evidence",
                    "trial": int(stale["trial_index"]),
                    "seed": stale.get("seed"),
                    "phase": stale.get("hpo_phase", "search"),
                    "target_fidelity": int((stale.get("hpo_fidelity") or {}).get("target", 0)),
                    "started": False,
                    "compute_seconds": 0.0,
                    "old_priority": stale.get("hpo_scheduler_priority"),
                    "new_best_action": action,
                    "new_priority": evidence.get("score"),
                    "new_evidence": {
                        "termination_type": result.termination_type,
                        "completed_trial": int((result.trial or {"index": 0})["index"]),
                        "completed_seed": result.seed,
                    },
                }
                if telemetry is not None:
                    telemetry.queued_action_cancelled(
                        stale, reason="superseded by new HPO evidence before dispatch"
                    )
                record_decision("CANCEL_QUEUED_ACTION", **cancellation)
            preemption = _request_scheduler_preemption(
                pending_specifications,
                alternatives=(
                    evidence.get("alternatives", ())
                    if isinstance(evidence.get("alternatives"), Sequence)
                    else ()
                ),
                selected_action=action,
                selected_trial=int(specifications[0]["trial_index"]),
                metric=metric,
                min_step=policy.early_stopping_min_step,
            )
            if preemption is not None:
                record_decision("PREEMPT", **preemption)
            elif pending_specifications:
                record_decision(
                    "CONTINUE",
                    running_runs=pending_runs,
                    reason="preemption benefit did not exceed checkpoint and hysteresis policy",
                    competing_action=action,
                    competing_priority=(
                        float(action_priority)
                        if isinstance(action_priority, int | float)
                        and not isinstance(action_priority, bool)
                        else None
                    ),
                )
            decision_evidence = {
                key: value
                for key, value in evidence.items()
                if key not in {"action", "trial", "seed"}
            }
            record_decision(
                action,
                **decision_evidence,
                trial=int(specifications[0]["trial_index"]),
                seed=specifications[0].get("seed"),
                completed_runs=len(outcomes),
                running_runs=pending_runs,
                queued_startup_runs=len(deferred_queue),
            )
            print(
                f"[hpo] {action} trial={specifications[0]['trial_index']} "
                f"seed={specifications[0].get('seed')} because {evidence.get('reason')}; "
                f"{pending_runs} Run(s) remain active",
                flush=True,
            )
            register(specifications)
            return specifications

        _execute_adaptive_dispatch(
            limited,
            resources=resources,
            policy=policy,
            objective_metric=metric,
            objective_mode=mode,
            objective=objective,
            historical_results=outcomes,
            parallelism=parallelism,
            telemetry=telemetry,
            on_result=observed,
            on_queued_cancel=lambda value: record_decision(
                "CANCEL_QUEUED_ACTION",
                reason="candidate-level-performance-prune",
                trial=int(value["trial_index"]),
                seed=value.get("seed"),
                started=False,
                compute_seconds=0.0,
            ),
        )
        # The dispatcher returns only after both its process set and its mutable queue are empty;
        # queued actions invalidated by candidate pruning have no WorkResult callback to decrement
        # this controller-side pending counter.
        for trial in inflight_by_trial:
            inflight_by_trial[trial] = 0
        if telemetry is not None and observed_trials:
            telemetry.candidates_observed(observed_trials)

    startup = selector.initial(startup_trial_count)
    proposed.extend(startup)
    record_decision(
        "START_NEW",
        reason="space-filling-startup",
        pool_trials=list(startup),
        public_trials=[public_trial(trial) for trial in startup],
    )
    # Fill the first wave with distinct space-filling candidates before scheduling a second seed
    # for any candidate.  This gives the sampler broad evidence and avoids leaving expensive
    # parallel hardware idle during startup.
    startup_by_trial = [initial_specifications(trial) for trial in startup]
    startup_specs = [
        values[seed_index]
        for seed_index in range(max((len(values) for values in startup_by_trial), default=0))
        for values in startup_by_trial
        if seed_index < len(values)
    ]
    for value in startup_specs:
        value["hpo_probe_purpose"] = "EXPLORE_COVERAGE"
        value["hpo_target_questions"] = ["Establish broad initial search-space evidence"]
    # Deferred startup Runs remain a provisional planning pool. Only specifications handed to the
    # dispatcher enter ``scheduled_run_keys``; this lets every result re-rank the undispatched tail.
    startup_width = max(1, min(parallelism, allowance(), len(startup_specs)))
    execute(
        startup_specs[:startup_width],
        startup,
        deferred=startup_specs[startup_width:],
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

    final_pruning_audit = (
        _pruner_calibration(
            outcomes,
            objective_evaluator,
            min_step=policy.early_stopping_min_step,
            confirmations=policy.early_stopping_confirmations,
            probability_threshold=policy.early_stopping_probability_threshold,
            margin=policy.early_stopping_equivalence_margin,
        )
        if policy.early_stopping
        else {"status": "disabled"}
    )
    budget_exhausted = allowance() <= 0
    time_exhausted = not within_time()
    candidate_budget_reached = len(proposed) >= candidate_budget
    candidate_pool_exhausted = len(proposed) >= len(all_trials)
    finish_reason = (
        "run-budget-exhausted"
        if budget_exhausted
        else "time-budget-exhausted"
        if time_exhausted
        else "candidate-budget-reached"
        if candidate_budget_reached
        else "candidate-pool-exhausted"
        if candidate_pool_exhausted
        else "explicit-record-convergence"
        if search_converged
        else "no-scientifically-useful-action"
    )
    # Persist FINISH before producing the terminal Study snapshot.  Consumers must be able to
    # distinguish budget exhaustion from an explicitly requested early convergence policy.
    record_decision(
        "FINISH",
        reason=finish_reason,
        completed_runs=len(outcomes),
        proposed_candidates=len(proposed),
        candidate_budget=candidate_budget,
        ranked_public_trials=[public_trial(trial) for trial in ranked],
        confirmed_public_trials=[public_trial(trial) for trial in active],
        candidate_budget_reached=candidate_budget_reached,
        candidate_pool_exhausted=candidate_pool_exhausted,
        budget_exhausted=budget_exhausted,
        time_exhausted=time_exhausted,
        convergence_enabled=policy.convergence_patience > 0,
        pruning_audit=final_pruning_audit,
    )
    if telemetry is not None:
        telemetry.candidate_states(
            active=tuple(proposal_numbers[trial] for trial in active),
            ranked=tuple(proposal_numbers[trial] for trial in ranked),
            finished=True,
        )
    return tuple(outcomes)


def _execute_adaptive_dispatch(
    specifications: Sequence[dict[str, Any]],
    *,
    resources: ResourceRequest,
    policy: AdaptiveSearchPolicy,
    objective_metric: str,
    objective_mode: str,
    objective: Mapping[str, Any],
    historical_results: Sequence[WorkResult],
    parallelism: int,
    telemetry: StudyTelemetry | None = None,
    on_result: Callable[
        [WorkResult, Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]],
        Sequence[dict[str, Any]],
    ]
    | None = None,
    on_queued_cancel: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[WorkResult, ...]:
    if not specifications:
        return ()
    control_root = Path(specifications[0]["execution_dir"]) / "hpo-control"
    visible_gpus = _visible_gpu_tokens(resources.gpu_count)
    effective_resources = resources
    effective_parallelism = parallelism
    if resources.gpu_count and len(visible_gpus) < resources.gpu_count:
        effective_resources = replace(resources, gpu_count=len(visible_gpus))
        effective_parallelism = min(
            parallelism,
            len(visible_gpus) * policy.runs_per_gpu,
        )
        print(
            f"[hpo] site granted {len(visible_gpus)} of the requested "
            f"{resources.gpu_count} GPU(s); continuing within the exact inherited allocation "
            f"with at most {effective_parallelism} concurrent Run(s).",
            flush=True,
        )
    prepared: list[dict[str, Any]] = []
    per_run = _adaptive_run_resources(effective_resources, effective_parallelism)

    def prepare_specification(specification: Mapping[str, Any]) -> dict[str, Any]:
        trial = int(specification["trial_index"])
        seed = specification.get("seed")
        token = f"trial-{trial:05d}-seed-{seed if seed is not None else 'none'}"
        metrics = control_root / f"{token}.metrics.jsonl"
        stop = control_root / f"{token}.stop"
        prune_evidence = stop.with_name(stop.name + ".evidence.json")
        checkpoint_manifest = control_root / f"{token}.checkpoint.json"
        raw_fidelity = specification.get("hpo_fidelity")
        continuing = isinstance(raw_fidelity, Mapping) and int(raw_fidelity.get("current", 0)) > 0
        if not continuing:
            metrics.unlink(missing_ok=True)
            checkpoint_manifest.unlink(missing_ok=True)
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
                "hpo_checkpoint_manifest_path": checkpoint_manifest,
                "hpo_objective": objective_metric,
                "hpo_objective_config": dict(specification["definition"].get("objective", {})),
            }
        )
        return value

    prepared.extend(prepare_specification(value) for value in specifications)

    def dispatch_refill(
        result: WorkResult,
        queued: Sequence[Mapping[str, Any]],
        pending: Sequence[Mapping[str, Any]],
    ) -> Sequence[dict[str, Any]]:
        if on_result is None:
            return ()
        return tuple(prepare_specification(value) for value in on_result(result, queued, pending))

    results: list[WorkResult] = []
    executors: list[ProcessPoolExecutor] = []
    try:
        if resources.gpu_count:
            _execute_gpu_admitted_runs(
                prepared,
                resources=effective_resources,
                policy=policy,
                parallelism=effective_parallelism,
                visible_gpus=visible_gpus,
                objective_metric=objective_metric,
                objective_mode=objective_mode,
                objective=objective,
                historical_results=historical_results,
                telemetry=telemetry,
                results=results,
                executors=executors,
                on_result=dispatch_refill if on_result is not None else None,
                on_queued_cancel=on_queued_cancel,
            )
        else:
            _execute_cpu_isolated_runs(
                prepared,
                resources=resources,
                policy=policy,
                parallelism=parallelism,
                objective_metric=objective_metric,
                objective_mode=objective_mode,
                objective=objective,
                historical_results=historical_results,
                telemetry=telemetry,
                results=results,
                executors=executors,
                on_result=dispatch_refill if on_result is not None else None,
                on_queued_cancel=on_queued_cancel,
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
    resources: ResourceRequest | None = None,
    policy: AdaptiveSearchPolicy,
    parallelism: int,
    objective_metric: str,
    objective_mode: str,
    objective: Mapping[str, Any] | None = None,
    historical_results: Sequence[WorkResult] = (),
    telemetry: StudyTelemetry | None,
    results: list[WorkResult],
    executors: list[ProcessPoolExecutor],
    on_result: Callable[
        [WorkResult, Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]],
        Sequence[dict[str, Any]],
    ]
    | None = None,
    on_queued_cancel: Callable[[Mapping[str, Any]], None] | None = None,
) -> None:
    """Keep CPU Runs isolated so one killed worker cannot break unrelated candidates."""
    resources = resources or ResourceRequest()
    queued = deque(dict(value) for value in prepared)
    pending: dict[Any, tuple[dict[str, Any], ProcessPoolExecutor]] = {}
    while queued or pending:
        if telemetry is not None:
            telemetry.admission_state(
                {
                    "summary": "waiting_for_resources"
                    if queued and len(pending) >= parallelism
                    else "admissible",
                    "pending_runs": len(queued),
                    "max_parallel": policy.max_parallel,
                    "cpu_slots": parallelism,
                    "cpu_capacity": resources.cpu_cores,
                    "ram_capacity_bytes": resources.ram_bytes,
                    "active_runs": len(pending),
                    "resource": "max_parallel" if queued and len(pending) >= parallelism else "cpu",
                    "reason": "max_parallel"
                    if queued and len(pending) >= parallelism
                    else "admissible",
                    "retryable": True,
                }
            )
        while queued and len(pending) < parallelism:
            value = queued.popleft()
            value["hpo_dispatched_monotonic"] = time.monotonic()
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
                        replacement = on_result(
                            failure,
                            tuple(queued),
                            tuple(item for item, _pool in pending.values()),
                        )
                        queued.clear()
                        queued.extend(replacement)
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
                        replacement = on_result(
                            result,
                            tuple(queued),
                            tuple(item for item, _pool in pending.values()),
                        )
                        queued.clear()
                        queued.extend(replacement)
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
                probability_threshold=policy.early_stopping_probability_threshold,
                margin=policy.early_stopping_equivalence_margin,
                historical_results=historical_results,
                objective=objective,
            )
            _cancel_queued_pruned_candidates(
                queued,
                tuple(value for value, _pool in pending.values()),
                telemetry=telemetry,
                on_cancel=on_queued_cancel,
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
    if not _is_gpu_memory_failure(result):
        return None
    kind = str(result.failure.get("type", ""))
    return _new_retry(
        specification,
        reason=f"{kind or 'resource failure'}: {result.failure.get('message', '')}",
        policy=policy,
        telemetry=telemetry,
    )


def _is_gpu_memory_failure(result: WorkResult) -> bool:
    """Recognize allocation failures that can benefit from safer GPU packing."""
    if not isinstance(result, WorkResult):
        return False
    if result.ok or result.pruned or not isinstance(result.failure, Mapping):
        return False
    kind = str(result.failure.get("type", ""))
    message = str(result.failure.get("message", "")).lower()
    return kind in {"OutOfMemoryError", "CUDAOutOfMemoryError"} or any(
        marker in message
        for marker in (
            "cuda out of memory",
            "cublas_status_alloc_failed",
            "hip out of memory",
        )
    )


def _safer_gpu_concurrency(current_limit: int, failed_active_runs: int) -> int:
    """Learn a lower packing ceiling after an OOM at the observed concurrency."""
    return min(current_limit, max(1, failed_active_runs - 1))


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
        termination_type="resource_failed",
        termination={
            "type": "resource_failed",
            "failure_type": type(error).__name__,
            "reason": str(error),
        },
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
    objective: Mapping[str, Any] | None = None,
    historical_results: Sequence[WorkResult] = (),
    telemetry: StudyTelemetry | None,
    results: list[WorkResult],
    executors: list[ProcessPoolExecutor],
    on_result: Callable[
        [WorkResult, Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]],
        Sequence[dict[str, Any]],
    ]
    | None = None,
    on_queued_cancel: Callable[[Mapping[str, Any]], None] | None = None,
) -> None:
    """Launch only Runs that currently fit, waiting through temporary VRAM pressure."""
    required = resources.gpu_memory_bytes
    memory = (
        _initial_gpu_memory_inventory(resources.gpu_count)
        if required > 0
        else tuple((0, 0) for _ in range(resources.gpu_count))
    )
    _validate_gpu_memory_capacity(memory, required)
    usable = tuple(index for index, (_, total) in enumerate(memory) if required <= total)
    queued = deque(dict(value) for value in prepared)
    pending: dict[Any, tuple[dict[str, Any], int, ProcessPoolExecutor]] = {}
    active = [0] * resources.gpu_count
    last_launch = [float("-inf")] * resources.gpu_count
    # OOM evidence belongs to the device on which it was observed.  A single externally loaded or
    # unusually constrained GPU must not throttle healthy siblings for the remainder of a Study.
    effective_runs_per_gpu = [policy.runs_per_gpu] * resources.gpu_count
    stable_runs_after_backoff = [0] * resources.gpu_count
    next_wait_log = 0.0
    consecutive_probe_failures = 0
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
                        replacement = on_result(
                            failure,
                            tuple(queued),
                            tuple(item for item, _slot, _pool in pending.values()),
                        )
                        queued.clear()
                        queued.extend(replacement)
            else:
                memory_failure = _is_gpu_memory_failure(result)
                retry = _retry_failed_result(
                    value,
                    result,
                    policy=policy,
                    telemetry=telemetry,
                )
                if memory_failure:
                    previous_limit = effective_runs_per_gpu[slot]
                    effective_runs_per_gpu[slot] = _safer_gpu_concurrency(
                        effective_runs_per_gpu[slot],
                        active[slot],
                    )
                    stable_runs_after_backoff[slot] = 0
                    if effective_runs_per_gpu[slot] < previous_limit:
                        print(
                            f"[hpo] CUDA OOM observed with {active[slot]} active Run(s) on "
                            f"GPU {visible_gpus[slot]}; reducing the study packing ceiling "
                            f"for that device from {previous_limit} to "
                            f"{effective_runs_per_gpu[slot]} Run(s)",
                            flush=True,
                        )
                elif (
                    isinstance(result, WorkResult)
                    and (result.ok or result.pruned)
                    and (effective_runs_per_gpu[slot] < policy.runs_per_gpu)
                ):
                    # One full turnover at the learned lower ceiling is direct evidence that the
                    # device is stable there. Probe exactly one additional slot next; live free
                    # VRAM still has to satisfy the normal per-launch threshold.
                    stable_runs_after_backoff[slot] += 1
                    if stable_runs_after_backoff[slot] >= effective_runs_per_gpu[slot]:
                        previous_limit = effective_runs_per_gpu[slot]
                        effective_runs_per_gpu[slot] += 1
                        stable_runs_after_backoff[slot] = 0
                        print(
                            f"[hpo] GPU {visible_gpus[slot]} completed one stable turnover at "
                            f"the reduced packing ceiling; cautiously restoring it from "
                            f"{previous_limit} to {effective_runs_per_gpu[slot]} Run(s)",
                            flush=True,
                        )
                if retry is not None:
                    queued.append(retry)
                else:
                    results.append(result)
                    if on_result is not None:
                        replacement = on_result(
                            result,
                            tuple(queued),
                            tuple(item for item, _slot, _pool in pending.values()),
                        )
                        queued.clear()
                        queued.extend(replacement)
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
                probability_threshold=policy.early_stopping_probability_threshold,
                margin=policy.early_stopping_equivalence_margin,
                historical_results=historical_results,
                objective=objective,
            )
            _cancel_queued_pruned_candidates(
                queued,
                tuple(value for value, _slot, _pool in pending.values()),
                telemetry=telemetry,
                on_cancel=on_queued_cancel,
            )
        if telemetry is not None:
            telemetry.refresh()

        if not queued:
            continue

        now = time.monotonic()
        if required > 0:
            try:
                memory = _gpu_memory_inventory(resources.gpu_count)
            except RuntimeError as error:
                # Observation is not execution.  A transient NVML/driver/CUDA probe
                # failure must never tear down healthy Runs that are already using the
                # allocation.  Fail closed for admission, preserve those children and
                # retry.  Once no child remains, a bounded failure makes a persistent
                # loss of the GPU grant visible instead of waiting forever.
                consecutive_probe_failures += 1
                if now >= next_wait_log:
                    print(
                        "[hpo] GPU admission probe unavailable; no new Run will start "
                        f"until observation recovers ({consecutive_probe_failures}/"
                        f"{_GPU_PROBE_FAILURE_LIMIT}): {error}",
                        flush=True,
                    )
                    next_wait_log = now + _GPU_WAIT_LOG_SECONDS
                if telemetry is not None:
                    telemetry.admission_state(
                        {
                            "status": "probe-unavailable",
                            "safe_to_launch": False,
                            "consecutive_failures": consecutive_probe_failures,
                            "reason": str(error),
                            "active_runs": len(pending),
                            "queued_runs": len(queued),
                        }
                    )
                if not pending and consecutive_probe_failures >= _GPU_PROBE_FAILURE_LIMIT:
                    raise RuntimeError(
                        "CUDA GPU admission remained unavailable for "
                        f"{consecutive_probe_failures} consecutive probes while Runs were "
                        f"waiting. Last cause: {error}"
                    ) from error
                time.sleep(_GPU_ADMISSION_POLL_SECONDS)
                continue
            else:
                if consecutive_probe_failures:
                    print(
                        "[hpo] GPU admission probe recovered; queued Runs may start again.",
                        flush=True,
                    )
                consecutive_probe_failures = 0
        else:
            memory = tuple((0, 0) for _ in range(resources.gpu_count))
        slots = _admissible_gpu_slots(
            memory,
            active=active,
            last_launch=last_launch,
            required_bytes=required,
            runs_per_gpu=effective_runs_per_gpu,
            now=now,
            launch_stagger_seconds=_GPU_LAUNCH_STAGGER_SECONDS,
            usable=usable,
        )
        if telemetry is not None:
            telemetry.admission_state(
                _gpu_admission_diagnostics(
                    memory,
                    active=active,
                    required_bytes=required,
                    runs_per_gpu=effective_runs_per_gpu,
                    configured_runs_per_gpu=policy.runs_per_gpu,
                    max_parallel=policy.max_parallel,
                    pending=len(queued),
                    visible_gpus=visible_gpus,
                    usable=usable,
                    admissible=slots,
                )
            )
        launched = False
        for slot in slots:
            if not queued or len(pending) >= parallelism:
                break
            value = queued.popleft()
            value["hpo_dispatched_monotonic"] = time.monotonic()
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
                f"{effective_runs_per_gpu[slot]}",
                flush=True,
            )
            launched = True

        if queued and not launched and now >= next_wait_log:
            states = ", ".join(
                f"GPU {visible_gpus[index]} "
                f"{'free=' + _memory_text(memory[index][0]) + ' ' if required else ''}"
                f"active={active[index]}/{effective_runs_per_gpu[index]}"
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
        detail = ""
        if isinstance(error, subprocess.CalledProcessError):
            output = (error.stderr or error.stdout or "").strip()
            if output:
                detail = f" Probe output: {output[-1200:]}"
        raise RuntimeError(
            f"CUDA GPU admission could not read current device memory safely.{detail}"
        ) from error


def _initial_gpu_memory_inventory(gpu_count: int) -> tuple[tuple[int, int], ...]:
    """Require a safe baseline, tolerating a short transient CUDA observation failure."""
    last_error: RuntimeError | None = None
    for attempt in range(1, 4):
        try:
            return _gpu_memory_inventory(gpu_count)
        except RuntimeError as error:
            last_error = error
            if attempt < 3:
                time.sleep(_GPU_ADMISSION_POLL_SECONDS)
    assert last_error is not None
    raise RuntimeError(
        "CUDA GPU admission could not establish a safe initial memory baseline after "
        f"3 probes. Last cause: {last_error}"
    ) from last_error


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
    required_bytes: int,
    runs_per_gpu: int | Sequence[int],
    now: float,
    launch_stagger_seconds: float,
    usable: Sequence[int],
) -> tuple[int, ...]:
    """Return devices that can accept one Run now, least-loaded and longest-idle first."""
    candidates = (
        index
        for index in usable
        if active[index] < _gpu_run_limit(runs_per_gpu, index)
        and now - last_launch[index] >= launch_stagger_seconds
        and (required_bytes <= 0 or memory[index][0] >= required_bytes)
    )
    # Stable index order used to starve a higher-index GPU whenever only one global slot became
    # free.  Prefer lower load, then the device that has waited longest, with index only as the
    # deterministic final tie-breaker.
    return tuple(sorted(candidates, key=lambda index: (active[index], last_launch[index], index)))


def _gpu_admission_diagnostics(
    memory: Sequence[tuple[int, int]],
    *,
    active: Sequence[int],
    required_bytes: int,
    runs_per_gpu: int | Sequence[int],
    configured_runs_per_gpu: int | None = None,
    max_parallel: int | None,
    pending: int,
    visible_gpus: Sequence[str],
    usable: Sequence[int],
    admissible: Sequence[int],
) -> dict[str, Any]:
    """Describe why queued Runs can or cannot consume each allocated GPU."""
    allowed = set(usable)
    ready = set(admissible)
    configured_limit = configured_runs_per_gpu or (
        runs_per_gpu if isinstance(runs_per_gpu, int) else max(runs_per_gpu, default=1)
    )
    devices: list[dict[str, Any]] = []
    for index, (free, total) in enumerate(memory):
        device_limit = _gpu_run_limit(runs_per_gpu, index)
        if index not in allowed:
            reason = "gpu_unavailable"
        elif active[index] >= device_limit:
            reason = "oom_backoff" if device_limit < configured_limit else "runs_per_gpu"
        elif required_bytes > 0 and free < required_bytes:
            reason = "insufficient_free_vram"
        elif index not in ready:
            reason = "launch_stagger_or_lease"
        else:
            reason = "admissible"
        devices.append(
            {
                "gpu": index,
                "token": visible_gpus[index] if index < len(visible_gpus) else str(index),
                "admitted": index in ready,
                "resource": "gpu_memory" if reason == "insufficient_free_vram" else "gpu",
                "reason": reason,
                "required": required_bytes,
                "available": free,
                "total": total,
                "active_runs": active[index],
                "runs_per_gpu": device_limit,
                "configured_runs_per_gpu": configured_limit,
                "retryable": reason not in {"gpu_unavailable"},
            }
        )
    return {
        "pending_runs": pending,
        "max_parallel": max_parallel,
        "runs_per_gpu": configured_limit,
        "effective_runs_per_gpu": min(
            (_gpu_run_limit(runs_per_gpu, index) for index in usable),
            default=configured_limit,
        ),
        "effective_runs_per_gpu_by_device": {
            visible_gpus[index] if index < len(visible_gpus) else str(index): _gpu_run_limit(
                runs_per_gpu, index
            )
            for index in range(len(memory))
        },
        "gpu_memory_semantics": "live-free-vram-threshold-per-new-run-not-a-reservation",
        "devices": devices,
        "summary": ("admissible" if ready else "waiting_for_resources" if pending else "idle"),
    }


def _gpu_run_limit(value: int | Sequence[int], index: int) -> int:
    """Return one device's current packing ceiling from scalar or per-device state."""
    return value if isinstance(value, int) else int(value[index])


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


def _adaptive_startup_trial_count(
    policy: AdaptiveSearchPolicy,
    *,
    parallelism: int,
    candidate_budget: int,
) -> int:
    """Resolve the distinct space-filling startup width.

    An explicit ``startup_trials`` remains authoritative.  Automatic startup retains the
    historical ten-candidate statistical floor but grows to the number of executable slots, so
    increasing safe parallelism cannot accidentally leave hardware idle before evidence exists.
    """
    requested = policy.startup_trials if policy.startup_trials is not None else max(10, parallelism)
    return min(candidate_budget, requested)


def _visible_gpu_tokens(gpu_count: int) -> tuple[str, ...]:
    """Return only GPUs inherited from an external allocation, failing closed when absent.

    A command launcher may grant fewer devices than the outer adaptive-study ceiling (for example
    when an opportunistic second GPU is unavailable).  In that mode the controller safely scales
    down to the observed tokens.  Scheduler allocations remain exact contracts.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    mode = os.environ.get("LAMBDAFORGE_GPU_ACCESS_MODE", "")
    inherited = tuple(value.strip() for value in (raw or "").split(",") if value.strip())
    disabled = {"-1", "none", "nodevfiles", "void"}
    if inherited and any(value.lower() in disabled for value in inherited):
        inherited = ()
    if inherited:
        if len(set(inherited)) != len(inherited):
            raise RuntimeError("CUDA_VISIBLE_DEVICES contains duplicate allocated GPU tokens.")
        if len(inherited) < gpu_count and mode != "command":
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
    if any(result.pruned or result.termination_type != "completed" for result in results):
        return None
    values = [
        value
        for result in results
        if not result.pruned and result.termination_type == "completed"
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
    if any(result.pruned or result.termination_type != "completed" for result in results):
        return None
    values = [
        value
        for result in results
        if not result.pruned and result.termination_type == "completed"
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


def _request_scheduler_preemption(
    specifications: Sequence[Mapping[str, Any]],
    *,
    alternatives: Sequence[Any],
    selected_action: str,
    selected_trial: int,
    metric: str,
    min_step: int,
    hysteresis_ratio: float = 0.5,
    minimum_runtime_seconds: float = 30.0,
) -> Mapping[str, Any] | None:
    """Cooperatively pause one resumable Run only for a materially better alternative."""
    ranked_alternatives = [
        value
        for value in alternatives
        if isinstance(value, Mapping)
        and isinstance(value.get("score"), int | float)
        and not isinstance(value.get("score"), bool)
        and not (
            str(value.get("action")) == selected_action
            and int(value.get("trial", -1)) == selected_trial
        )
    ]
    if not ranked_alternatives:
        return None
    alternative = max(ranked_alternatives, key=lambda value: float(value["score"]))
    alternative_score = float(alternative["score"])
    now = time.monotonic()
    candidates: list[tuple[float, Mapping[str, Any], int, float, float]] = []
    for specification in specifications:
        fidelity = specification.get("hpo_fidelity")
        if (
            specification.get("hpo_phase") == "confirmation"
            or not isinstance(fidelity, Mapping)
            or specification.get("hpo_resume_preempted")
        ):
            continue
        raw_stop = specification.get("hpo_stop_path")
        if raw_stop is None:
            continue
        stop = Path(str(raw_stop))
        if stop.is_file():
            continue
        started = specification.get("hpo_dispatched_monotonic")
        if not isinstance(started, int | float) or isinstance(started, bool):
            continue
        elapsed = max(0.0, now - float(started))
        if elapsed < minimum_runtime_seconds:
            continue
        raw_old_priority = specification.get("hpo_scheduler_priority")
        if (
            not isinstance(raw_old_priority, int | float)
            or isinstance(raw_old_priority, bool)
            or float(raw_old_priority) <= 0
        ):
            # Startup coverage has no comparable acquisition score. Do not interrupt it using a
            # synthetic priority: preemption is allowed only between auditable planned actions.
            continue
        if not _active_checkpoint_available(specification):
            continue
        history = _objective_history(Path(str(specification["hpo_metrics_path"])), metric)
        if not history or history[-1][0] < min_step:
            continue
        observed_step = history[-1][0]
        current = int(fidelity.get("current", 0))
        target = int(fidelity.get("target", 0))
        if observed_step <= current or observed_step >= target:
            continue
        completed_fraction = (observed_step - current) / max(1, target - current)
        remaining_cost = elapsed * (1.0 - completed_fraction) / max(completed_fraction, 1e-6)
        continue_information = 0.5 / math.sqrt(1 + observed_step)
        continue_priority = continue_information / max(remaining_cost, 1e-6)
        old_priority = max(float(raw_old_priority), continue_priority)
        improvement = alternative_score - old_priority * (1.0 + hysteresis_ratio)
        if improvement > 0:
            candidates.append((improvement, specification, observed_step, old_priority, elapsed))
    if not candidates:
        return None
    _improvement, selected, observed_step, old_priority, elapsed = max(
        candidates,
        key=lambda value: (
            value[0],
            -int(value[1].get("trial_index", 0)),
        ),
    )
    stop = Path(str(selected["hpo_stop_path"]))
    evidence = {
        "termination_type": "scheduler_preempted",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": int(selected["trial_index"]),
        "seed": selected.get("seed"),
        "observed_step": observed_step,
        "fidelity": dict(selected["hpo_fidelity"]),
        "old_action": str(selected.get("hpo_scheduler_action", "CONTINUE")),
        "new_action": str(alternative.get("action")),
        "old_priority": old_priority,
        "new_priority": alternative_score,
        "hysteresis_ratio": hysteresis_ratio,
        "minimum_runtime_seconds": minimum_runtime_seconds,
        "elapsed_seconds": elapsed,
        "checkpoint_required": True,
        "reason": ("checkpoint-safe alternative exceeded continuation priority plus hysteresis"),
        "evidence_revision": "new terminal Run changed information-per-cost ordering",
    }
    evidence_path = stop.with_name(stop.name + ".evidence.json")
    atomic_json(evidence_path, evidence)
    stop.parent.mkdir(parents=True, exist_ok=True)
    stop.write_text(
        "scheduler requested a cooperative checkpoint-safe pause; "
        f"alternative={evidence['new_action']} old_priority={old_priority:.8g} "
        f"new_priority={alternative_score:.8g}\n",
        encoding="utf-8",
    )
    return evidence


def _active_checkpoint_available(specification: Mapping[str, Any]) -> bool:
    """Verify an active Run has an owned durable checkpoint before requesting a pause."""
    raw_manifest = specification.get("hpo_checkpoint_manifest_path")
    raw_execution = specification.get("execution_dir")
    if raw_manifest is None or raw_execution is None:
        return False
    manifest = Path(str(raw_manifest))
    if not manifest.is_file() or manifest.is_symlink():
        return False
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
        raw_root = value.get("checkpoint_root") if isinstance(value, Mapping) else None
        if not isinstance(raw_root, str):
            return False
        declared_root = Path(raw_root)
        if declared_root.is_symlink():
            return False
        root = declared_root.resolve(strict=True)
        owned = (Path(str(raw_execution)).resolve() / "runs").resolve()
        if not root.is_relative_to(owned) or not root.is_dir():
            return False
        for path in root.rglob("*"):
            if path.is_symlink():
                continue
            if path.is_file():
                return True
        return False
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _request_early_stops(
    specifications: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    mode: str,
    min_step: int,
    confirmations: int = 2,
    probability_threshold: float = 0.1,
    margin: float = 0.0,
    historical_results: Sequence[WorkResult] = (),
    objective: Mapping[str, Any] | None = None,
) -> None:
    """Request candidate-level pruning against active and comparable historical evidence."""
    if confirmations < 1:
        raise ValueError("early-stopping confirmations must be positive.")
    evaluator = ObjectiveUtility(objective or {"metric": metric, "mode": mode})
    histories: list[tuple[Mapping[str, Any], tuple[tuple[int, float], ...]]] = []
    for ordinal, specification in enumerate(specifications, 1):
        if specification.get("hpo_phase") == "confirmation":
            continue
        resolved = dict(specification)
        resolved.setdefault("trial_index", ordinal)
        resolved.setdefault("seed", None)
        history = _utility_history((Path(str(specification["hpo_metrics_path"])),), evaluator)
        if history and history[-1][0] >= min_step:
            histories.append((resolved, history))
    if not histories:
        return
    common_step = min(history[-1][0] for _, history in histories)
    latest_values = [
        [value for step, value in history if step <= common_step][-1]
        for _, history in histories
        if any(step <= common_step for step, _ in history)
    ]
    pooled_deviation = statistics.stdev(latest_values) if len(latest_values) > 1 else 1.0
    calibration = _pruner_calibration(
        historical_results,
        evaluator,
        min_step=min_step,
        confirmations=confirmations,
        probability_threshold=probability_threshold,
        margin=margin,
    )
    calibrated_prior = max(
        pooled_deviation,
        float(calibration.get("curve_rmse", 0.0) or 0.0),
        1e-6,
    )
    fidelity_targets = [
        int(value["hpo_fidelity"]["target"])
        for value, _history in histories
        if isinstance(value.get("hpo_fidelity"), Mapping)
    ]
    # Predict to the next real decision boundary when fidelity defines one. Without it, keep the
    # horizon deliberately local; uncertainty still expands with distance.
    target_step = (
        max(common_step + 1, min(fidelity_targets))
        if fidelity_targets
        else common_step + max(1, min_step)
    )
    observed: list[tuple[int, int | None, float, float, int, str]] = []
    for specification, history in histories:
        comparable = [(step, value) for step, value in history if step <= common_step]
        if comparable:
            mean, deviation = predict_curve(
                comparable,
                target_step=target_step,
                prior_deviation=calibrated_prior,
            )
            observed.append(
                (
                    int(specification["trial_index"]),
                    int(specification["seed"]) if specification.get("seed") is not None else None,
                    mean,
                    deviation,
                    comparable[-1][0],
                    "active",
                )
            )
    active_trials = {value[0] for value in observed}
    active_seed_keys = {(trial, seed) for trial, seed, *_rest in observed}
    latest_historical: dict[tuple[int, int | None], WorkResult] = {}
    for result in historical_results:
        if (
            result.termination_type != "completed"
            or result.study_phase == "confirmation"
            or result.trial is None
        ):
            continue
        trial = int(result.trial["index"])
        key = (trial, result.seed)
        if key in active_seed_keys:
            # A resumed seed's active cumulative curve already contains its earlier rung. Counting
            # the prior result as another seed would understate uncertainty and over-weight it.
            continue
        previous = latest_historical.get(key)
        target = int((result.fidelity or {}).get("target", 2**31 - 1))
        previous_target = int((previous.fidelity or {}).get("target", -1)) if previous else -1
        if target >= previous_target:
            latest_historical[key] = result
    for (trial, seed), result in latest_historical.items():
        history = _utility_history(
            (result.run_dir / "metrics.jsonl", result.run_dir / "training-metrics.jsonl"),
            evaluator,
        )
        comparable = [(step, value) for step, value in history if step <= common_step]
        if not comparable:
            continue
        mean, deviation = predict_curve(
            comparable,
            target_step=target_step,
            prior_deviation=calibrated_prior,
        )
        observed.append((trial, seed, mean, deviation, comparable[-1][0], "historical"))
    grouped: dict[int, list[tuple[int | None, float, float, int, str]]] = {}
    for trial, seed, mean, deviation, step, source in observed:
        grouped.setdefault(trial, []).append((seed, mean, deviation, step, source))
    estimates: dict[int, tuple[float, float, int]] = {}
    for trial, values in grouped.items():
        means = [value[1] for value in values]
        # Between-seed variance is estimable only after a second seed. The curve model already
        # carries a conservative prior for a single seed; adding the full cross-candidate spread
        # here would mistake signal between candidates for seed noise and disable pruning.
        between = statistics.variance(means) if len(means) > 1 else 0.0
        curve_variance = statistics.fmean(value[2] ** 2 for value in values)
        estimates[trial] = (
            statistics.fmean(means),
            math.sqrt(max(1e-12, between + curve_variance) / len(values)),
            len(values),
        )
    if len(estimates) < 2:
        return
    incumbent_trial = (max if mode == "max" else min)(
        estimates, key=lambda trial: estimates[trial][0]
    )
    incumbent = estimates[incumbent_trial]
    sign = 1.0 if mode == "max" else -1.0
    for trial in active_trials:
        if trial == incumbent_trial:
            continue
        mean, deviation, samples = estimates[trial]
        trial_by_seed = {
            seed: predicted
            for seed, predicted, _error, _step, _source in grouped[trial]
            if seed is not None
        }
        incumbent_by_seed = {
            seed: predicted
            for seed, predicted, _error, _step, _source in grouped[incumbent_trial]
            if seed is not None
        }
        shared_seeds = sorted(set(trial_by_seed).intersection(incumbent_by_seed))
        paired = [sign * (trial_by_seed[seed] - incumbent_by_seed[seed]) for seed in shared_seeds]
        if len(paired) >= 2:
            difference = statistics.fmean(paired)
            difference_error = statistics.stdev(paired) / math.sqrt(len(paired))
            comparison_method = "paired-seed-predicted-differences"
        else:
            difference = sign * (mean - incumbent[0])
            difference_error = math.sqrt(deviation**2 + incumbent[1] ** 2)
            comparison_method = "independent-candidate-posterior"
        probability = (
            1.0
            if difference_error <= 0 and difference >= -margin
            else 0.0
            if difference_error <= 0
            else 1.0 - statistics.NormalDist(difference, difference_error).cdf(-margin)
        )
        trial_specifications = [
            specification
            for specification, _history in histories
            if int(specification["trial_index"]) == trial
        ]
        if not trial_specifications:
            continue
        evidence_path = Path(str(trial_specifications[0]["hpo_stop_path"])).with_name(
            f"trial-{trial:05d}.prune-evidence.json"
        )
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
        evidence = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "common_step": common_step,
            "confirmations": count,
            "required_confirmations": confirmations,
            "probability_competitive": probability,
            "threshold": probability_threshold,
            "equivalence_margin": margin,
            "candidate": trial,
            "reference_candidate": incumbent_trial,
            "current_utility": statistics.fmean(
                value[1] for value in grouped[trial] if value[4] == "active"
            ),
            "predicted_utility": mean,
            "predicted_standard_error": deviation,
            "reference_predicted_utility": incumbent[0],
            "reference_standard_error": incumbent[1],
            "candidate_seed_evidence": samples,
            "reference_seed_evidence": incumbent[2],
            "shared_seed_evidence": len(shared_seeds),
            "comparison_method": comparison_method,
            "target_step": target_step,
            "fidelity_targets": sorted(set(fidelity_targets)),
            "curve_model": "conservative-local-linear-last-5",
            "retrospective_calibration": calibration,
        }
        atomic_json(evidence_path, evidence)
        if count >= confirmations:
            for specification in trial_specifications:
                stop = Path(str(specification["hpo_stop_path"]))
                if stop.exists():
                    continue
                run_evidence = stop.with_name(stop.name + ".evidence.json")
                atomic_json(run_evidence, {**evidence, "seed": specification.get("seed")})
                stop.parent.mkdir(parents=True, exist_ok=True)
                stop.write_text(
                    "candidate-level probabilistic performance pruning requested: "
                    f"candidate={trial} reference={incumbent_trial} "
                    f"p_competitive={probability:.6f} threshold={probability_threshold:.6f} "
                    f"common_step={common_step} confirmations={count}/{confirmations}\n",
                    encoding="utf-8",
                )


def _cancel_queued_pruned_candidates(
    queued: deque[dict[str, Any]],
    active: Sequence[Mapping[str, Any]],
    *,
    telemetry: StudyTelemetry | None,
    on_cancel: Callable[[Mapping[str, Any]], None] | None,
) -> None:
    """Remove not-yet-started seeds after their candidate receives a prune decision."""
    pruned_trials = {
        int(specification["trial_index"])
        for specification in active
        if specification.get("hpo_phase") != "confirmation"
        and Path(str(specification["hpo_stop_path"])).is_file()
    }
    if not pruned_trials or not queued:
        return
    retained: deque[dict[str, Any]] = deque()
    while queued:
        specification = queued.popleft()
        trial = int(specification["trial_index"])
        if trial not in pruned_trials or specification.get("hpo_phase") == "confirmation":
            retained.append(specification)
            continue
        if telemetry is not None:
            telemetry.queued_action_cancelled(
                specification,
                reason="candidate received a candidate-level performance-prune decision",
            )
        if on_cancel is not None:
            on_cancel(specification)
    queued.extend(retained)


def _pruner_calibration(
    results: Sequence[WorkResult],
    evaluator: ObjectiveUtility,
    *,
    min_step: int,
    confirmations: int,
    probability_threshold: float,
    margin: float,
) -> dict[str, Any]:
    """Build the bounded candidate-level retrospective policy report."""
    completed_identity = tuple(
        (
            result.run_id,
            result.attempt_id,
            str(result.run_dir),
            result.termination_type,
        )
        for result in results
        if result.termination_type == "completed" and result.trial is not None
    )
    cache_key = (
        completed_identity,
        json.dumps(evaluator.objective, sort_keys=True, separators=(",", ":")),
        min_step,
        confirmations,
        probability_threshold,
        margin,
    )
    cached = _PRUNER_AUDIT_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    curves: list[CompletedCurve] = []
    for result in results:
        if result.termination_type != "completed" or result.trial is None:
            continue
        history = _utility_history(
            (result.run_dir / "metrics.jsonl", result.run_dir / "training-metrics.jsonl"),
            evaluator,
        )
        if history:
            curves.append(
                CompletedCurve(
                    int(result.trial["index"]),
                    result.seed,
                    history,
                    result.duration_seconds,
                    result.gpu_index is not None or result.gpu_token is not None,
                )
            )
    report = audit_pruner(
        curves,
        mode=evaluator.mode,
        min_step=min_step,
        confirmations=confirmations,
        probability_threshold=probability_threshold,
        margin=margin,
    )
    if len(_PRUNER_AUDIT_CACHE) >= 64:
        _PRUNER_AUDIT_CACHE.pop(next(iter(_PRUNER_AUDIT_CACHE)))
    _PRUNER_AUDIT_CACHE[cache_key] = report
    return dict(report)


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


def _utility_history(
    paths: Sequence[Path], evaluator: ObjectiveUtility
) -> tuple[tuple[int, float], ...]:
    """Build a same-checkpoint utility curve without mixing metric epochs."""
    records: dict[int, dict[str, float]] = {}
    synthesized: dict[int, float] = {}
    for path in paths:
        if not path.is_file() or path.is_symlink():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                value = json.loads(line)
                step = value.get("step")
                raw = value.get("value")
                if not isinstance(step, int) or isinstance(step, bool):
                    continue
                if not isinstance(raw, int | float) or isinstance(raw, bool):
                    continue
                key = (
                    f"{value['split']}_{value['name']}"
                    if value.get("split")
                    else str(value.get("name", ""))
                )
                if key == evaluator.metric:
                    synthesized[step] = float(raw)
                elif key in evaluator.required_metrics:
                    records.setdefault(step, {})[key] = float(raw)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
    history: dict[int, float] = dict(synthesized)
    for step, values in records.items():
        if step in history:
            continue
        evaluated = evaluator.evaluate(values)
        if evaluated is not None:
            history[step] = float(evaluated["value"])
    return tuple(sorted(history.items()))


def _prune_reason(path: Path) -> str:
    """Return a compact persisted pruning explanation without making it resumable state."""
    try:
        detail = path.read_text(encoding="utf-8").strip()
    except OSError:
        detail = ""
    return f"adaptive-early-stopping: {detail}" if detail else "adaptive-early-stopping"


def _run_termination(
    *,
    status: str,
    failure: Mapping[str, Any] | None,
    stop_path: Path | None,
    completed_step: int | None = None,
    target_step: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Classify operational termination without confusing it with scientific quality."""
    # A stop request is only a pruning/preemption outcome when the Work reached its safe boundary
    # and returned normally. A coincident OOM or consumer exception remains the real termination.
    if status == "succeeded" and stop_path is not None and stop_path.is_file():
        evidence_path = stop_path.with_name(stop_path.name + ".evidence.json")
        evidence: dict[str, Any] = {}
        if evidence_path.is_file() and not evidence_path.is_symlink():
            try:
                decoded = json.loads(evidence_path.read_text(encoding="utf-8"))
                evidence = dict(decoded) if isinstance(decoded, Mapping) else {}
            except (OSError, json.JSONDecodeError):
                evidence = {}
        requested_type = str(evidence.get("termination_type", "performance_pruned"))
        if completed_step is not None and target_step is not None and completed_step >= target_step:
            return "completed", {
                "type": "completed",
                "stop_request_observed_after_target": requested_type,
                "completed_step": completed_step,
                "target_step": target_step,
            }
        if requested_type == "scheduler_preempted":
            return "scheduler_preempted", {
                "type": "scheduler_preempted",
                "reason": evidence.get("reason") or _prune_reason(stop_path),
                **evidence,
            }
        return "performance_pruned", {
            "type": "performance_pruned",
            "reason": _prune_reason(stop_path),
            **evidence,
        }
    if status == "succeeded":
        return "completed", {"type": "completed"}
    failure_type = str((failure or {}).get("type", ""))
    message = str((failure or {}).get("message", "")).lower()
    resource = failure_type in {"OutOfMemoryError", "CUDAOutOfMemoryError"} or any(
        marker in message for marker in ("out of memory", "cuda error", "resource exhausted")
    )
    selected = "resource_failed" if resource else "scientific_failed"
    return selected, {
        "type": selected,
        "failure_type": failure_type or None,
        "reason": str((failure or {}).get("message", "")) or None,
    }


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
    records: list[Mapping[str, Any]] = []
    for path in paths:
        if not path.is_file() or path.is_symlink():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                value = json.loads(line)
                if isinstance(value, Mapping):
                    records.append(value)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
    return ObjectiveUtility(objective).observation(records, fallback=fallback)


def _constraint_satisfied(value: Any, rule: Mapping[str, Any]) -> bool:
    """Evaluate one explicit outcome bound; missing evidence fails closed."""
    return constraint_satisfied(value, rule)


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


def _candidate_values_by_rung(
    results_by_trial: Mapping[int, Sequence[WorkResult]],
    *,
    metric: str,
    objective: Mapping[str, Any],
) -> dict[tuple[int, int], dict[int, dict[int | None, float]]]:
    """Group candidate seed evidence by one exact cumulative-fidelity rung.

    A candidate may contribute observations at several rungs, but a mean is formed only from seeds
    that reached the same ``(target, maximum)`` pair. One performance-pruned seed censors the whole
    candidate at every rung, preventing survivor-only fitting.
    """
    selected: dict[tuple[int, int], dict[int, dict[int | None, tuple[int, float, WorkResult]]]] = {}
    for raw_trial, results in results_by_trial.items():
        trial = int(raw_trial)
        if any(
            result.pruned or result.termination_type == "performance_pruned" for result in results
        ):
            continue
        for result in results:
            value = _result_objective(result, metric)
            if result.termination_type != "completed" or value is None:
                continue
            fidelity = result.fidelity or {}
            target = int(fidelity.get("target", 1))
            maximum = max(1, int(fidelity.get("maximum", target)))
            rung = (target, maximum)
            prior = selected.setdefault(rung, {}).setdefault(trial, {}).get(result.seed)
            if prior is None or result.attempt_number >= prior[0]:
                selected[rung][trial][result.seed] = (result.attempt_number, value, result)
    grouped: dict[tuple[int, int], dict[int, dict[int | None, float]]] = {}
    for rung, by_trial in selected.items():
        for trial, by_seed in by_trial.items():
            evidence = [result for _attempt, _value, result in by_seed.values()]
            if not _candidate_constraints(evidence, objective)["feasible"]:
                continue
            grouped.setdefault(rung, {})[trial] = {
                seed: value for seed, (_attempt, value, _result) in by_seed.items()
            }
    return grouped


def _candidate_observations(
    results_by_trial: Mapping[int, Sequence[WorkResult]],
    *,
    metric: str,
    objective: Mapping[str, Any],
    racer: AdaptiveSeedRacer,
) -> tuple[CandidateObservation, ...]:
    """Build the exact rung-aware observations consumed by every proposal sampler."""
    observations: list[CandidateObservation] = []
    grouped = _candidate_values_by_rung(results_by_trial, metric=metric, objective=objective)
    for (target, maximum), outcomes in sorted(
        grouped.items(), key=lambda item: (item[0][0] / max(1, item[0][1]), item[0])
    ):
        for trial, estimate in sorted(racer.estimates(outcomes).items()):
            observations.append(
                CandidateObservation(
                    trial=trial,
                    value=estimate.mean,
                    standard_error=estimate.standard_error,
                    fidelity=target / maximum,
                )
            )
    return tuple(observations)


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
        summary[str(name)] = aggregate_constraint(values, rule, missing=missing)
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
        str(
            value.get(
                "termination_type",
                "performance_pruned"
                if value.get("pruned")
                else "completed"
                if value.get("status") == "succeeded"
                else "scientific_failed",
            )
        ),
        value.get("termination", {}),
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
