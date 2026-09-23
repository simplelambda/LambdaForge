"""Strict, intentionally small YAML model for Work execution."""

from __future__ import annotations

import copy
import importlib
import inspect
import itertools
import math
import os
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.hpo.AdaptiveSearch import SEARCH_POLICY_FIELDS, AdaptiveSearchPolicy
from lambdaforge.hpo.ObjectiveUtility import ObjectiveUtility
from lambdaforge.metrics.MetricRegistry import MetricRegistry
from lambdaforge.ProjectContext import ProjectContext
from lambdaforge.reproducibility.SeedProvider import SeedIdentity, SeedProvider
from lambdaforge.work.design import (
    EvidenceKind,
    EvidencePlan,
    ExecutionPolicy,
    ResolvedStudyConfiguration,
    StudyDesign,
)
from lambdaforge.work.models import immutable_mapping
from lambdaforge.work.Work import Work

_TOP_FIELDS = frozenset(
    {
        "name",
        "run",
        "with",
        "resources",
        "seeds",
        "replicates",
        "search",
        "sweep",
        "execution",
        "objective",
        "steps",
    }
)
_RUN_FIELDS = frozenset(
    {
        "name",
        "run",
        "with",
        "resources",
        "seeds",
        "replicates",
        "search",
        "sweep",
        "execution",
        "objective",
    }
)


class WorkYamlError(yaml.YAMLError):
    """Readable syntax error for an authored Work YAML document."""


def _yaml_error_message(source: Path, text: str, error: yaml.YAMLError) -> str:
    """Turn PyYAML parser internals into a bounded, source-aware diagnostic."""
    problem = str(getattr(error, "problem", "") or "invalid YAML syntax").strip().rstrip(".")
    context = str(getattr(error, "context", "") or "").strip().rstrip(".")
    problem_mark = getattr(error, "problem_mark", None)
    context_mark = getattr(error, "context_mark", None)
    mark = (
        context_mark
        if context.lower() == "while scanning a simple key" and context_mark is not None
        else problem_mark or context_mark
    )
    lines = text.splitlines()
    location = ""
    excerpt = ""
    hint = "Review the indicated YAML indentation, key and ':' separator."
    if mark is not None:
        line_number = int(mark.line) + 1
        column_number = int(mark.column) + 1
        location = f" at line {line_number}, column {column_number}"
        if 0 <= int(mark.line) < len(lines):
            raw_line = lines[int(mark.line)].expandtabs(4)
            visible_line = raw_line[:160]
            caret_column = min(max(int(mark.column), 0), len(visible_line))
            excerpt = f"\n  {line_number:>4} | {visible_line}\n       | {' ' * caret_column}^"
            if raw_line.strip() and ":" not in raw_line:
                hint = (
                    "This line is a standalone value. Remove it if it is accidental, or write "
                    "it as a YAML key followed by ':'."
                )
    detail = problem
    if context and context.lower() not in problem.lower():
        detail = f"{problem} ({context})"
    return f"Invalid YAML in {source}{location}: {detail}.{excerpt}\nHint: {hint}"


@dataclass(frozen=True, slots=True)
class WorkValidationReport:
    """Side-effect-free validation facts for one Work document."""

    source: Path | None
    name: str | None
    valid: bool
    errors: tuple[str, ...] = ()
    work_classes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a stable machine-readable report."""
        return {
            "source": str(self.source) if self.source else None,
            "name": self.name,
            "valid": self.valid,
            "errors": list(self.errors),
            "work_classes": list(self.work_classes),
        }

    def summary(self) -> str:
        """Render a compact human result."""
        label = str(self.source) if self.source else self.name or "<mapping>"
        if self.valid:
            return f"Valid Work: {label} ({len(self.work_classes)} executable class(es))."
        return "Invalid Work: " + label + "\n" + "\n".join(f"  - {error}" for error in self.errors)


@dataclass(frozen=True, slots=True)
class RunDefinition:
    """One normalized Work class invocation before seed/search expansion."""

    name: str
    work_class: str
    parameters: Mapping[str, Any]
    resources: ResourceRequest
    seeds: tuple[int | None, ...] = (None,)
    variants: tuple[Mapping[str, Any], ...] = ({},)
    objective: Mapping[str, Any] | None = None
    search_policy: AdaptiveSearchPolicy | None = None
    execution_policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    study_design: StudyDesign | None = None
    study_expected: bool = False
    # Last on purpose: older internal callers constructed this dataclass
    # positionally before seed provenance existed.
    seed_metadata: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", immutable_mapping(self.parameters))
        object.__setattr__(
            self,
            "variants",
            tuple(immutable_mapping(variant) for variant in self.variants),
        )
        if self.objective is not None:
            object.__setattr__(self, "objective", immutable_mapping(self.objective))
        object.__setattr__(
            self,
            "seed_metadata",
            tuple(immutable_mapping(value) for value in self.seed_metadata),
        )

    @property
    def run_count(self) -> int:
        if self.study_design is not None and self.study_design.kind in {"sweep", "repeated"}:
            return len(self.study_design.evidence.required)
        candidates = (
            min(self.search_policy.candidate_budget or len(self.variants), len(self.variants))
            if self.search_policy is not None
            else len(self.variants)
        )
        confirmation = (
            min(self.search_policy.confirmation_top_k, candidates)
            * len(self.search_policy.confirmation_seeds)
            if self.search_policy is not None
            else 0
        )
        return len(self.seeds) * candidates + confirmation


@dataclass(frozen=True, slots=True)
class WorkLevel:
    """One sequential workflow level whose members may execute independently."""

    runs: tuple[RunDefinition, ...]


@dataclass(frozen=True, slots=True)
class WorkConfig:
    """Validated single Work or sequence/parallel composition loaded from YAML."""

    name: str
    levels: tuple[WorkLevel, ...]
    source: Path | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw", immutable_mapping(self.raw))

    @classmethod
    def from_yaml(cls, path: str | Path) -> WorkConfig:
        """Parse and validate one current Work YAML document."""
        source = Path(path).expanduser().resolve()
        text = source.read_text(encoding="utf-8")
        try:
            value = yaml.safe_load(text)
        except yaml.YAMLError as error:
            raise WorkYamlError(_yaml_error_message(source, text, error)) from error
        if not isinstance(value, Mapping):
            raise TypeError("LambdaForge YAML must contain one mapping.")
        return cls.from_mapping(value, source=source)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, source: str | Path | None = None
    ) -> WorkConfig:
        """Normalize the only supported YAML language and reject historical fields."""
        data = copy.deepcopy(dict(value))
        unexpected = set(data) - _TOP_FIELDS
        if unexpected:
            raise ValueError(
                f"Unknown top-level field(s): {sorted(unexpected)}. "
                f"Allowed fields: {sorted(_TOP_FIELDS)}."
            )
        source_path = Path(source).expanduser().resolve() if source is not None else None
        name = _nonempty(
            data.get("name", source_path.stem if source_path is not None else None), "name"
        )
        project = ProjectContext.discover(source_path or Path.cwd())
        if ("run" in data) == ("steps" in data):
            raise ValueError("A Work YAML must define exactly one run or steps.")
        default_resources = _work_resources(data.get("resources"))
        if "run" in data:
            levels: tuple[WorkLevel, ...] = (
                WorkLevel(
                    (
                        _run_definition(
                            data,
                            default_name=name,
                            inherited_resources=None,
                            project=project,
                        ),
                    )
                ),
            )
        else:
            steps = data.get("steps")
            if not isinstance(steps, Sequence) or isinstance(steps, str | bytes) or not steps:
                raise TypeError("steps must be a non-empty list.")
            levels_list: list[WorkLevel] = []
            seen: set[str] = set()
            for index, step in enumerate(steps, 1):
                if not isinstance(step, Mapping):
                    raise TypeError(f"steps[{index - 1}] must be a mapping.")
                if set(step) == {"parallel"}:
                    parallel = step["parallel"]
                    if (
                        not isinstance(parallel, Sequence)
                        or isinstance(parallel, str | bytes)
                        or not parallel
                    ):
                        raise TypeError(f"steps[{index - 1}].parallel must be a non-empty list.")
                    definitions = tuple(
                        _run_definition(
                            item,
                            default_name=f"step-{index}-{offset}",
                            inherited_resources=default_resources,
                            project=project,
                        )
                        for offset, item in enumerate(parallel, 1)
                    )
                else:
                    definitions = (
                        _run_definition(
                            step,
                            default_name=f"step-{index}",
                            inherited_resources=default_resources,
                            project=project,
                        ),
                    )
                duplicates = seen.intersection(run.name for run in definitions)
                if duplicates:
                    raise ValueError(f"Workflow step names must be unique: {sorted(duplicates)}.")
                seen.update(run.name for run in definitions)
                levels_list.append(WorkLevel(definitions))
            levels = tuple(levels_list)
        for level in levels:
            if len(level.runs) > 1 and any(run.search_policy is not None for run in level.runs):
                raise ValueError(
                    "Adaptive search cannot be nested in a YAML parallel level. One adaptive "
                    "Work already owns and schedules its fixed resource allocation."
                )
        config = cls(name, levels, source_path, data)
        errors = config.validation_errors(check_inputs=True)
        if errors:
            raise ValueError("Invalid Work configuration:\n  - " + "\n  - ".join(errors))
        return config

    @classmethod
    def validate_file(cls, path: str | Path) -> WorkValidationReport:
        """Return all locally discoverable errors without launching a job."""
        source = Path(path).expanduser().resolve()
        try:
            config = cls.from_yaml(source)
        except Exception as error:
            message = (
                str(error)
                if isinstance(error, WorkYamlError)
                else f"{type(error).__name__}: {error}"
            )
            return WorkValidationReport(source, None, False, (message,))
        errors = config.validation_errors(check_inputs=True)
        classes = tuple(run.work_class for level in config.levels for run in level.runs)
        return WorkValidationReport(source, config.name, not errors, tuple(errors), classes)

    def validation_errors(self, *, check_inputs: bool) -> list[str]:
        """Validate Work inheritance, signatures, types, markers and references."""
        errors: list[str] = []
        prior: dict[str, RunDefinition] = {}
        for level in self.levels:
            for definition in level.runs:
                try:
                    target = import_work_class(definition.work_class)
                    variant_errors: set[str] = set()
                    for variant in definition.variants:
                        variant_errors.update(
                            _signature_errors(
                                target,
                                {**dict(definition.parameters), **dict(variant)},
                            )
                        )
                    errors.extend(sorted(variant_errors))
                except Exception as error:
                    errors.append(str(error))
                for parameter, marker in _markers(definition.parameters):
                    if "file" in marker and check_inputs:
                        source_dir = self.source.parent if self.source else Path.cwd()
                        path = Path(str(marker["file"]))
                        resolved = path if path.is_absolute() else (source_dir / path).resolve()
                        if not resolved.exists() or resolved.is_symlink():
                            errors.append(
                                f"Parameter {parameter!r} file input is missing or "
                                f"symbolic: {resolved}"
                            )
                    if "dataset" in marker and check_inputs:
                        try:
                            from lambdaforge.data.DatasetReference import DatasetReference
                            from lambdaforge.data.DatasetRegistry import DatasetRegistry

                            source_dir = self.source.parent if self.source else Path.cwd()
                            configured = str(marker["dataset"])
                            reference = (
                                configured
                                if configured.startswith("dataset:")
                                else f"dataset:{configured}"
                            )
                            parsed = DatasetReference.parse(reference)
                            if "@" not in parsed.selector:
                                raise ValueError(
                                    "Work dataset inputs require an exact NAME@VERSION selector."
                                )
                            registry_path = os.environ.get(
                                "LAMBDAFORGE_DATASET_REGISTRY"
                            ) or DatasetRegistry.project_path(source_dir)
                            DatasetRegistry(registry_path).get(parsed.selector)
                        except Exception as error:
                            errors.append(
                                f"Parameter {parameter!r} dataset cannot be resolved: {error}"
                            )
                    if "from" in marker:
                        producer, separator, output = str(marker["from"]).partition(".")
                        if not separator or not producer or not output:
                            errors.append(
                                f"Parameter {parameter!r} from marker must be STEP.OUTPUT."
                            )
                        elif producer not in prior:
                            errors.append(
                                f"Parameter {parameter!r} references unknown or later "
                                f"step {producer!r}."
                            )
                        elif prior[producer].run_count != 1:
                            errors.append(
                                f"Output reference {marker['from']!r} is ambiguous because "
                                f"{producer!r} expands to {prior[producer].run_count} Runs."
                            )
                prior[definition.name] = definition
        return errors

    @property
    def resources(self) -> ResourceRequest:
        """Return a conservative fixed outer allocation for sequence/parallel levels."""
        additive = ("cpu_cores", "ram_bytes", "gpu_count", "storage_bytes", "processes")
        capacity = {name: 0 for name in additive}
        gpu_memory = 0
        runtime: float | None = 0.0
        for level in self.levels:
            for field_name in additive:
                capacity[field_name] = max(
                    capacity[field_name],
                    sum(getattr(run.resources, field_name) for run in level.runs),
                )
            gpu_memory = max(
                gpu_memory,
                *(run.resources.gpu_memory_bytes for run in level.runs),
            )
            level_time = [run.resources.runtime_seconds for run in level.runs]
            if runtime is not None:
                if any(value is None for value in level_time):
                    runtime = None
                else:
                    runtime += max(value or 0.0 for value in level_time)
        return ResourceRequest(
            **capacity,
            gpu_memory_bytes=gpu_memory,
            runtime_seconds=runtime,
        )

    @property
    def planned_runs(self) -> int:
        return sum(run.run_count for level in self.levels for run in level.runs)

    @property
    def has_parameter_study(self) -> bool:
        """Return whether search or repeated seeds declare comparable Runs."""
        return any(run.study_expected for level in self.levels for run in level.runs)

    def to_dict(self) -> dict[str, Any]:
        """Return the authored current-schema representation without internal IR."""
        return _plain(self.raw)

    def explanation(self) -> dict[str, Any]:
        """Describe classes, signatures, defaults, parameters and resources."""
        levels: list[list[dict[str, Any]]] = []
        for level in self.levels:
            current = []
            for definition in level.runs:
                target = import_work_class(definition.work_class)
                method = target.run
                signature = inspect.signature(method)
                hints = _hints(method)
                parameters = []
                for parameter in tuple(signature.parameters.values())[1:]:
                    parameters.append(
                        {
                            "name": parameter.name,
                            "type": _type_name(hints.get(parameter.name, parameter.annotation)),
                            "required": parameter.default is inspect.Parameter.empty,
                            "default": (
                                None
                                if parameter.default is inspect.Parameter.empty
                                else parameter.default
                            ),
                            "configured": definition.parameters.get(parameter.name, None),
                        }
                    )
                current.append(
                    {
                        "name": definition.name,
                        "class": definition.work_class,
                        "class_doc": inspect.getdoc(target),
                        "run_doc": inspect.getdoc(method),
                        "parameters": parameters,
                        "resources": definition.resources.to_dict(),
                        "runs": definition.run_count,
                        "objective": dict(definition.objective or {}),
                        "search": (
                            definition.search_policy.to_dict()
                            if definition.search_policy is not None
                            else (
                                {"strategy": "fixed", "design": definition.study_design.to_dict()}
                                if definition.study_design is not None
                                else None
                            )
                        ),
                        "execution": definition.execution_policy.to_dict(),
                    }
                )
            levels.append(current)
        return {
            "name": self.name,
            "levels": levels,
            "planned_runs": self.planned_runs,
            "resources": self.resources.to_dict(),
            "resolved_configuration": self.resolved_configuration(),
        }

    def resolved_configuration(self) -> dict[str, Any]:
        """Return the complete versioned policy authority consumed by execution."""
        resolved: list[list[dict[str, Any]]] = []
        for level in self.levels:
            values: list[dict[str, Any]] = []
            for definition in level.runs:
                design = definition.study_design
                study_type = design.kind if design is not None else "single"
                configuration = ResolvedStudyConfiguration(
                    name=definition.name,
                    study_type=study_type,
                    design=(
                        design.to_dict()
                        if design is not None
                        else {
                            "type": "single",
                            "seed_source": (
                                dict(definition.seed_metadata[0])
                                if definition.seed_metadata
                                else None
                            ),
                        }
                    ),
                    search=(
                        {
                            **definition.search_policy.to_dict(),
                            "candidate_generation": {
                                "mode": (
                                    "deterministic-incremental"
                                    if definition.search_policy.candidate_budget is None
                                    else "deterministic-bounded"
                                ),
                                "generator": "scrambled-sobol-prefix-v1",
                                "materialized_prefix": len(definition.variants),
                                "candidate_cap": definition.search_policy.candidate_budget,
                            },
                            "confirmation_policy": (
                                "automatic-fresh-project-stream"
                                if definition.search_policy.confirmation_auto
                                else "explicit"
                            ),
                        }
                        if definition.search_policy is not None
                        else None
                    ),
                    execution=definition.execution_policy.to_dict(),
                    objective=dict(definition.objective or {}),
                    resources=definition.resources.to_dict(),
                )
                values.append(configuration.to_dict())
            resolved.append(values)
        return {
            "resolved_work_configuration_version": 1,
            "name": self.name,
            "source": str(self.source) if self.source is not None else None,
            "levels": resolved,
        }


def import_work_class(path: str) -> type[Work]:
    """Resolve the explicit executable boundary and reject functions/foreign classes."""
    if not isinstance(path, str) or "." not in path:
        raise ValueError("run must be a dotted Work class path such as 'my_project.Train'.")
    module_name, symbol_name = path.rsplit(".", 1)
    value = getattr(importlib.import_module(module_name), symbol_name)
    if inspect.isroutine(value):
        raise TypeError(
            f"{path!r} resolves to a Python function. LambdaForge YAML can only execute "
            "Work classes. Define a class inheriting lambdaforge.Work and implement run()."
        )
    if not inspect.isclass(value):
        raise TypeError(f"{path!r} does not resolve to a Python class.")
    if not issubclass(value, Work):
        raise TypeError(
            f"{path!r} is a Python class but it is not a LambdaForge Work. "
            "Executable classes must inherit lambdaforge.Work."
        )
    if value is Work or value.run is Work.run:
        raise TypeError(f"{path!r} must implement run().")
    constructor = inspect.signature(value)
    try:
        constructor.bind()
    except TypeError as error:
        raise TypeError(
            f"{path!r} cannot be created without arguments. Work parameters belong to run()."
        ) from error
    return value


def _run_definition(
    value: Any,
    *,
    default_name: str,
    inherited_resources: ResourceRequest | None,
    project: ProjectContext,
) -> RunDefinition:
    if not isinstance(value, Mapping):
        raise TypeError("Every Work step must be a mapping.")
    data = copy.deepcopy(dict(value))
    unexpected = set(data) - _RUN_FIELDS
    if unexpected:
        raise ValueError(
            f"Unknown Work field(s): {sorted(unexpected)}. Allowed fields: {sorted(_RUN_FIELDS)}."
        )
    work_class = _nonempty(data.get("run"), "run")
    name = _nonempty(data.get("name", default_name), "name")
    parameters = data.get("with", {})
    if not isinstance(parameters, Mapping):
        raise TypeError(f"Work {name!r} with must be a mapping.")
    resources = (
        _work_resources(data.get("resources"))
        if "resources" in data
        else inherited_resources or ResourceRequest()
    )
    objective = _objective(data.get("objective"))
    raw_search = data.get("search")
    raw_sweep = data.get("sweep")
    if raw_search is not None and raw_sweep is not None:
        raise ValueError("A Work may define search or sweep, never both.")
    normalized_search = _normalize_search(raw_search, objective=objective)
    unseeded_single = (
        normalized_search is None
        and raw_sweep is None
        and "seeds" not in data
        and "replicates" not in data
    )
    seed_identities: tuple[SeedIdentity, ...]
    if unseeded_single:
        seed_identities = ()
        automatic_seed_source = False
        automatic_sweep = False
        seeds: tuple[int | None, ...] = (None,)
    else:
        seed_identities, automatic_seed_source, automatic_sweep = _seed_identities(
            data,
            search=normalized_search,
            sweep=raw_sweep,
            project=project,
        )
        seeds = tuple(value.value for value in seed_identities)
    seed_source = {
        "kind": (
            "none" if unseeded_single else "project-stream" if automatic_seed_source else "explicit"
        ),
        "namespace": project.project_id,
        "role": "replicate" if automatic_seed_source else "explicit",
        "stream_version": (
            "project-sha256-v1" if automatic_seed_source else "authored-v1"
        ),
        "extendable": automatic_seed_source,
        "resolved": [value.to_dict() for value in seed_identities],
    }
    execution = _execution_policy(data.get("execution"), normalized_search)
    if normalized_search is not None:
        normalized_search = _apply_execution_policy(normalized_search, execution)
    policy = _search_policy(
        normalized_search,
        seeds=seeds,
        extendable_seeds=automatic_seed_source,
        objective=objective,
        resources=resources,
    )
    design: StudyDesign | None
    if raw_sweep is not None:
        sweep_space, sweep_reference = _sweep(raw_sweep)
        variants = _sweep_variants(sweep_space)
        _validate_sweep_reference(sweep_reference, variants)
        if execution.max_runs is not None and execution.max_runs < len(variants) * len(seeds):
            raise ValueError(
                "The fixed sweep requires "
                f"{len(variants) * len(seeds)} Runs, but execution.max_runs="
                f"{execution.max_runs}. A sweep cannot silently start with an impossible "
                "evidence budget."
            )
        design = StudyDesign(
            "sweep",
            sweep_space,
            sweep_reference,
            EvidencePlan.fixed(candidates=len(variants), seeds=seeds),
            replication="auto-blocks" if automatic_sweep else "fixed",
            seed_source=seed_source,
        )
    elif normalized_search is not None and policy is None:
        exhaustive_space = _search_space(normalized_search)
        variants = _exhaustive_variants(normalized_search)
        if execution.max_runs is not None and execution.max_runs < len(variants) * len(seeds):
            raise ValueError(
                "The fixed sweep requires "
                f"{len(variants) * len(seeds)} Runs, but execution.max_runs="
                f"{execution.max_runs}. A sweep cannot silently start with an impossible "
                "evidence budget."
            )
        design = StudyDesign(
            "sweep",
            exhaustive_space,
            None,
            EvidencePlan.fixed(candidates=len(variants), seeds=seeds),
            replication="fixed",
            seed_source=seed_source,
        )
    elif policy is not None:
        assert normalized_search is not None
        variants = _variants(normalized_search, adaptive=True)
        design = StudyDesign(
            "adaptive",
            _search_space(normalized_search),
            None,
            EvidencePlan.adaptive(
                candidates=min(policy.candidate_budget or len(variants), len(variants)),
                seeds=seeds,
                minimum=policy.min_seeds,
            ),
            goal=policy.goal,
            replication="adaptive",
            seed_source=seed_source,
        )
    else:
        variants = ({},)
        design = (
            StudyDesign(
                "repeated",
                {},
                None,
                EvidencePlan.fixed(
                    candidates=1,
                    seeds=seeds,
                    kind=EvidenceKind.SWEEP_REQUIRED,
                ),
                replication="fixed",
                seed_source=seed_source,
            )
            if len(seeds) > 1
            else None
        )
    return RunDefinition(
        name=name,
        work_class=work_class,
        parameters=dict(parameters),
        resources=resources,
        seeds=seeds,
        seed_metadata=tuple(value.to_dict() for value in seed_identities),
        variants=variants,
        objective=objective,
        search_policy=policy,
        execution_policy=execution,
        study_design=design,
        study_expected=design is not None,
    )


def _seeds(value: Any) -> tuple[int | None, ...]:
    if value is None:
        return (None,)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or not value:
        raise TypeError("seeds must be a non-empty list of integers.")
    seeds = tuple(value)
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise TypeError("seeds must contain integers.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds cannot contain duplicates.")
    return seeds


def _seed_identities(
    data: Mapping[str, Any],
    *,
    search: Mapping[str, Any] | None,
    sweep: Any,
    project: ProjectContext,
) -> tuple[tuple[SeedIdentity, ...], bool, bool]:
    """Resolve authored values or a project stream without hiding the concrete integers."""
    if "seeds" in data and "replicates" in data:
        raise ValueError("seeds and replicates are alternative authorities; use only one.")
    provider = SeedProvider(project)
    if "seeds" in data:
        explicit = _seeds(data.get("seeds"))
        values = tuple(int(value) for value in explicit if value is not None)
        return provider.explicit(values), False, False
    sweep_replicates: Any = None
    if isinstance(sweep, Mapping):
        sweep_replicates = sweep.get("replicates")
    if data.get("replicates") is not None and sweep_replicates is not None:
        raise ValueError("Top-level replicates and sweep.replicates cannot both be set.")
    raw_count = data.get("replicates", sweep_replicates)
    automatic_sweep = sweep is not None and raw_count is None
    if raw_count is None:
        raw_count = search.get("min_seeds", 1) if search is not None else 1
    if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 1:
        raise ValueError("replicates must be a positive integer.")
    identities = provider.stream("replicate").take(raw_count)
    return identities, True, automatic_sweep


def _work_resources(value: Any) -> ResourceRequest:
    if value is not None and not isinstance(value, Mapping):
        raise TypeError("resources must be a mapping.")
    allowed = {"cpu", "memory", "gpu", "gpu_memory", "time", "storage", "processes"}
    unknown = set(value or {}) - allowed
    if unknown:
        raise ValueError(
            f"Unknown Work resource field(s): {sorted(unknown)}. Allowed fields: {sorted(allowed)}."
        )
    return ResourceRequest.from_mapping(value)


_EXECUTION_FIELDS = frozenset(
    {"runs_per_gpu", "max_parallel", "failure_retries", "max_runs", "max_time"}
)
_STRUCTURED_SEARCH_FIELDS = frozenset({"budget", "space", "replication", "pruning"})


def _normalize_search(
    value: Any,
    *,
    objective: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Normalize modern and legacy search syntax before any downstream interpretation."""
    if value is None:
        return None
    if not isinstance(value, Mapping) or not value:
        raise TypeError("search must be a non-empty mapping.")
    raw = copy.deepcopy(dict(value))
    normalized = {
        str(name): item
        for name, item in raw.items()
        if name not in _STRUCTURED_SEARCH_FIELDS
    }
    raw_space = raw.get("space")
    if raw_space is not None:
        if not isinstance(raw_space, Mapping) or not raw_space:
            raise TypeError("search.space must be a non-empty mapping.")
        legacy_dimensions = {
            str(name)
            for name in raw
            if name not in SEARCH_POLICY_FIELDS
            and name not in _STRUCTURED_SEARCH_FIELDS
            and name != "trials"
        }
        if legacy_dimensions:
            raise ValueError(
                "search.space cannot be combined with legacy direct parameter fields: "
                f"{sorted(legacy_dimensions)}."
            )
        normalized["parameter_space"] = {
            str(name): _dimension_mapping(descriptor) for name, descriptor in raw_space.items()
        }
    raw_budget = raw.get("budget")
    if raw_budget is not None:
        if not isinstance(raw_budget, Mapping):
            raise TypeError("search.budget must be a mapping.")
        unknown = set(raw_budget) - {"candidates", "runs", "time"}
        if unknown:
            raise ValueError(f"Unknown search.budget field(s): {sorted(unknown)}.")
        aliases = {
            "candidates": "trials",
            "runs": "max_runs",
            "time": "max_time",
        }
        for modern, legacy in aliases.items():
            if modern not in raw_budget:
                continue
            _merge_alias(
                normalized,
                legacy,
                raw_budget[modern],
                modern=f"search.budget.{modern}",
                legacy=f"search.{legacy}",
            )
    raw_replication = raw.get("replication")
    if raw_replication is not None:
        if not isinstance(raw_replication, Mapping):
            raise TypeError("search.replication must be a mapping.")
        unknown = set(raw_replication) - {"minimum", "confirmation"}
        if unknown:
            raise ValueError(f"Unknown search.replication field(s): {sorted(unknown)}.")
        if "minimum" in raw_replication:
            _merge_alias(
                normalized,
                "min_seeds",
                raw_replication["minimum"],
                modern="search.replication.minimum",
                legacy="search.min_seeds",
            )
        if "confirmation" in raw_replication:
            confirmation = raw_replication["confirmation"]
            if not (isinstance(confirmation, str) and confirmation.lower() == "auto"):
                _merge_alias(
                    normalized,
                    "confirmation_seeds",
                    confirmation,
                    modern="search.replication.confirmation",
                    legacy="search.confirmation_seeds",
                )
    if "pruning" in raw:
        _merge_alias(
            normalized,
            "early_stopping",
            raw["pruning"],
            modern="search.pruning",
            legacy="search.early_stopping",
        )

    practical_margin = objective.get("practical_margin") if objective is not None else None
    authored_margins: list[tuple[str, float]] = []
    if isinstance(practical_margin, int | float) and not isinstance(practical_margin, bool):
        authored_margins.append(("objective.practical_margin", float(practical_margin)))
    seed_racing = normalized.get("seed_racing")
    if isinstance(seed_racing, Mapping) and isinstance(
        seed_racing.get("equivalence_margin"), int | float
    ):
        authored_margins.append(
            (
                "search.seed_racing.equivalence_margin",
                float(seed_racing["equivalence_margin"]),
            )
        )
    pruning = normalized.get("early_stopping")
    if isinstance(pruning, Mapping) and isinstance(
        pruning.get("equivalence_margin"), int | float
    ):
        authored_margins.append(
            (
                "search.early_stopping.equivalence_margin",
                float(pruning["equivalence_margin"]),
            )
        )
    if isinstance(normalized.get("equivalence_margin"), int | float):
        authored_margins.append(
            ("search.equivalence_margin", float(normalized["equivalence_margin"]))
        )
    if authored_margins and any(
        not math.isclose(value, authored_margins[0][1], rel_tol=0.0, abs_tol=0.0)
        for _name, value in authored_margins[1:]
    ):
        details = ", ".join(f"{name}={margin}" for name, margin in authored_margins)
        raise ValueError(
            "All practical-equivalence aliases describe one scientific margin and cannot "
            f"disagree: {details}."
        )
    if authored_margins:
        canonical_margin = authored_margins[0][1]
        normalized["equivalence_margin"] = canonical_margin
        if objective is not None:
            objective["practical_margin"] = canonical_margin
    return normalized


def _merge_alias(
    target: dict[str, Any],
    key: str,
    value: Any,
    *,
    modern: str,
    legacy: str,
) -> None:
    if key in target and target[key] != value:
        raise ValueError(f"{modern} and {legacy} cannot disagree.")
    target[key] = value


def _execution_policy(value: Any, search: Mapping[str, Any] | None) -> ExecutionPolicy:
    if value is not None and not isinstance(value, Mapping):
        raise TypeError("execution must be a mapping.")
    normalized = dict(value or {})
    for name in _EXECUTION_FIELDS:
        if search is None or name not in search:
            continue
        if name in normalized and normalized[name] != search[name]:
            raise ValueError(f"execution.{name} and search.{name} cannot disagree.")
        normalized[name] = search[name]
    return ExecutionPolicy.from_mapping(normalized)


def _apply_execution_policy(
    search: Mapping[str, Any], execution: ExecutionPolicy
) -> dict[str, Any]:
    normalized = dict(search)
    normalized.update(execution.to_dict())
    return normalized


def _dimension_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return {"values": list(value)}
    raise TypeError("Search/sweep dimensions must be lists or mappings.")


def _search_space(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = value.get("parameter_space")
    if isinstance(raw, Mapping):
        return {str(name): _dimension_mapping(descriptor) for name, descriptor in raw.items()}
    return {
        str(name): _dimension_mapping(descriptor)
        for name, descriptor in value.items()
        if str(name) not in SEARCH_POLICY_FIELDS and str(name) != "trials"
    }


def _sweep(value: Any) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not isinstance(value, Mapping):
        raise TypeError("sweep must be a mapping containing space.")
    unknown = set(value) - {"space", "reference", "replicates"}
    if unknown:
        forbidden = sorted(unknown)
        raise ValueError(
            "A sweep executes every authored combination with every authored seed; adaptive "
            "candidate/seed controls are therefore not applicable: "
            f"{forbidden}."
        )
    raw_space = value.get("space")
    if not isinstance(raw_space, Mapping) or not raw_space:
        raise TypeError("sweep.space must be a non-empty mapping.")
    space = {str(name): _dimension_mapping(descriptor) for name, descriptor in raw_space.items()}
    raw_reference = value.get("reference")
    if raw_reference is not None and not isinstance(raw_reference, Mapping):
        raise TypeError("sweep.reference must map parameter names to authored values.")
    reference = (
        {str(name): selected for name, selected in raw_reference.items()}
        if isinstance(raw_reference, Mapping)
        else None
    )
    return space, reference


def _sweep_variants(space: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    normalized: dict[str, Any] = {}
    for name, raw_descriptor in space.items():
        descriptor = dict(raw_descriptor)
        if "range" in descriptor:
            points = descriptor.get("points")
            if isinstance(points, bool) or not isinstance(points, int) or points < 2:
                raise ValueError(
                    f"sweep.space.{name}.range requires integer points >= 2; a sweep never "
                    "interprets a continuous range as random search."
                )
            bounds = descriptor["range"]
            if (
                not isinstance(bounds, Sequence)
                or isinstance(bounds, str | bytes)
                or len(bounds) != 2
            ):
                raise TypeError(f"sweep.space.{name}.range must contain [low, high].")
            low, high = float(bounds[0]), float(bounds[1])
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError(f"sweep.space.{name}.range requires finite low < high.")
            scale = str(descriptor.get("scale", "linear"))
            if scale == "log":
                if low <= 0:
                    raise ValueError(f"sweep.space.{name} log range requires positive bounds.")
                ratio = (high / low) ** (1.0 / (points - 1))
                values = [low * ratio**index for index in range(points)]
            elif scale == "linear":
                values = [low + (high - low) * index / (points - 1) for index in range(points)]
            else:
                raise ValueError(f"sweep.space.{name}.scale must be linear or log.")
            descriptor = {
                "values": values,
                **({"when": descriptor["when"]} if "when" in descriptor else {}),
            }
        elif "points" in descriptor:
            raise ValueError(f"sweep.space.{name}.points is valid only with range.")
        normalized[name] = descriptor
    return _exhaustive_variants(normalized)


def _validate_sweep_reference(
    reference: Mapping[str, Any] | None, variants: Sequence[Mapping[str, Any]]
) -> None:
    if reference is None:
        return
    matches = [
        variant
        for variant in variants
        if all(variant.get(name) == value for name, value in reference.items())
    ]
    if not matches:
        raise ValueError("sweep.reference must identify at least one authored sweep combination.")
    if len(matches) > 1:
        raise ValueError(
            "sweep.reference must identify exactly one authored combination; include the "
            "remaining dimensions to avoid an ambiguous control."
        )


def _variants(value: Any, *, adaptive: bool = False) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ({},)
    if not isinstance(value, Mapping) or not value:
        raise TypeError("search must map run parameter names to values/ranges.")
    finite_names: list[str] = []
    finite_values: list[tuple[Any, ...]] = []
    random_space: dict[str, dict[str, Any]] = {}
    raw_trials = value.get("trials")
    random_count = int(raw_trials) if raw_trials is not None else 320
    raw_pool_size = value.get("proposal_pool_size")
    proposal_pool_size = int(raw_pool_size) if raw_pool_size is not None else None
    conditional = False
    parameter_space = value.get("parameter_space")
    structured_space = isinstance(parameter_space, Mapping)
    dimensions: Mapping[Any, Any] = (
        parameter_space if isinstance(parameter_space, Mapping) else value
    )
    for raw_name, raw_descriptor in dimensions.items():
        name = str(raw_name)
        if not structured_space and name == "trials":
            random_count = int(raw_descriptor)
            continue
        if not structured_space and name == "proposal_pool_size":
            proposal_pool_size = int(raw_descriptor)
            continue
        if not structured_space and name in SEARCH_POLICY_FIELDS:
            continue
        descriptor = (
            raw_descriptor if isinstance(raw_descriptor, Mapping) else {"values": raw_descriptor}
        )
        condition = descriptor.get("when")
        if condition is not None:
            if not isinstance(condition, Mapping) or not condition:
                raise TypeError(f"search.{name}.when must be a non-empty mapping.")
            conditional = True
        if "values" in descriptor:
            choices = descriptor["values"]
            if not isinstance(choices, Sequence) or isinstance(choices, str | bytes) or not choices:
                raise TypeError(f"search.{name}.values must be a non-empty list.")
            finite_names.append(name)
            finite_values.append(tuple(choices))
            random_space[name] = {"type": "choice", "values": list(choices)}
        elif "range" in descriptor:
            bounds = descriptor["range"]
            if (
                not isinstance(bounds, Sequence)
                or isinstance(bounds, str | bytes)
                or len(bounds) != 2
            ):
                raise TypeError(f"search.{name}.range must contain [low, high].")
            kind = str(descriptor.get("type", "uniform"))
            kind = "uniform" if kind == "float" else kind
            kind = "loguniform" if descriptor.get("scale") == "log" else kind
            random_space[name] = {"type": kind, "low": bounds[0], "high": bounds[1]}
        else:
            raise ValueError(f"search.{name} requires values or range.")
        if condition is not None:
            random_space[name]["when"] = {str(key): item for key, item in condition.items()}
    has_range = any(
        isinstance(descriptor, Mapping) and "range" in descriptor
        for descriptor in dimensions.values()
    )
    if not adaptive:
        return _exhaustive_variants(value)
    if has_range or conditional or raw_trials is not None:
        if random_count < 1:
            raise ValueError("search.trials must be >= 1.")
        from lambdaforge.hpo.CandidateGenerator import DeterministicCandidateGenerator

        requested = (
            proposal_pool_size
            if proposal_pool_size is not None
            else (
                max(random_count, min(4096, random_count * 16))
                if raw_trials is not None
                else random_count
            )
        )
        finite_cardinality = (
            math.prod(len(values) for values in finite_values) if not has_range else None
        )
        pool_count = min(requested, finite_cardinality) if finite_cardinality else requested
        return DeterministicCandidateGenerator(random_space).prefix(pool_count)
    return tuple(
        dict(zip(finite_names, combination, strict=True))
        for combination in itertools.product(*finite_values)
    )


def _exhaustive_variants(value: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Enumerate an exact finite conditional sweep in authored parameter order."""
    dimensions: list[tuple[str, tuple[Any, ...], Mapping[str, Any] | None]] = []
    known: set[str] = set()
    parameter_space = value.get("parameter_space")
    structured_space = isinstance(parameter_space, Mapping)
    authored_dimensions: Mapping[str, Any] = (
        parameter_space if isinstance(parameter_space, Mapping) else value
    )
    for raw_name, raw_descriptor in authored_dimensions.items():
        name = str(raw_name)
        if not structured_space and (name in SEARCH_POLICY_FIELDS or name == "trials"):
            continue
        descriptor = (
            raw_descriptor if isinstance(raw_descriptor, Mapping) else {"values": raw_descriptor}
        )
        if "range" in descriptor:
            raise ValueError(
                f"search.{name}.range is continuous and cannot be exhaustive; use "
                "strategy=adaptive or replace it with an explicit values list."
            )
        choices = descriptor.get("values")
        if not isinstance(choices, Sequence) or isinstance(choices, str | bytes) or not choices:
            raise TypeError(f"search.{name}.values must be a non-empty list.")
        raw_condition = descriptor.get("when")
        condition: Mapping[str, Any] | None = None
        if raw_condition is not None:
            if not isinstance(raw_condition, Mapping) or not raw_condition:
                raise TypeError(f"search.{name}.when must be a non-empty mapping.")
            unknown = set(map(str, raw_condition)) - known
            if unknown:
                raise ValueError(
                    f"search.{name}.when must reference parameters declared earlier: "
                    f"{sorted(unknown)}."
                )
            condition = {str(key): item for key, item in raw_condition.items()}
        dimensions.append((name, tuple(choices), condition))
        known.add(name)

    expanded: list[dict[str, Any]] = [{}]
    for name, choices, condition in dimensions:
        next_values: list[dict[str, Any]] = []
        for current in expanded:
            active = condition is None or all(
                current.get(key) == expected for key, expected in condition.items()
            )
            if active:
                next_values.extend({**current, name: choice} for choice in choices)
            else:
                next_values.append(current)
        expanded = next_values
    return tuple(expanded)


def _search_policy(
    value: Any,
    *,
    seeds: tuple[int | None, ...],
    extendable_seeds: bool,
    objective: Mapping[str, Any] | None,
    resources: ResourceRequest,
) -> AdaptiveSearchPolicy | None:
    if value is None:
        return None
    assert isinstance(value, Mapping)
    raw_strategy = value.get("strategy")
    strategy = (
        str(raw_strategy).lower()
        if raw_strategy is not None
        else ("adaptive" if objective is not None else "exhaustive")
    )
    if strategy not in {"adaptive", "exhaustive"}:
        raise ValueError("search.strategy must be adaptive or exhaustive.")
    if strategy == "exhaustive":
        # Physical limits are normalized into ``ExecutionPolicy`` for both controllers.  They
        # remain in this mapping only so legacy ``search.runs_per_gpu`` et al. can share the same
        # downstream representation as top-level ``execution``.
        unexpected = set(
            SEARCH_POLICY_FIELDS.intersection(value) - {"strategy"} - _EXECUTION_FIELDS
        )
        if "trials" in value:
            unexpected.add("trials")
        if unexpected:
            raise ValueError(
                "Adaptive search options are not valid with search.strategy=exhaustive: "
                f"{sorted(unexpected)}."
            )
        return None
    if objective is None:
        raise ValueError("Adaptive search requires a scalar or composite objective.")
    policy = AdaptiveSearchPolicy.from_search(value)
    if policy.min_seeds > len(seeds) and not extendable_seeds:
        raise ValueError("search.min_seeds cannot exceed the number of configured seeds.")
    variants = _variants(value, adaptive=True)
    proposed_candidates = min(policy.candidate_budget or len(variants), len(variants))
    required_minimum_runs = proposed_candidates * policy.min_seeds
    if policy.max_runs is not None and policy.max_runs < required_minimum_runs:
        raise ValueError(
            "search/ execution run budget cannot satisfy the declared minimum replication: "
            f"{required_minimum_runs} required Runs but max_runs={policy.max_runs}."
        )
    overlap = set(policy.confirmation_seeds).intersection(
        seed for seed in seeds if seed is not None
    )
    if overlap:
        raise ValueError(
            f"search.confirmation_seeds must be disjoint from search seeds: {sorted(overlap)}."
        )
    # gpu_memory remains an optional user safety floor.  When omitted, the adaptive resource
    # planner starts conservatively and learns candidate-specific future envelopes; runs_per_gpu
    # is only the hard cap and therefore no longer requires a homogeneous per-Run declaration.
    return policy


def _objective(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        resolved = MetricRegistry.resolve(value)
        if resolved is None:
            raise ValueError(
                f"Objective shorthand {value!r} is not a registered standard metric; use "
                "objective: {metric: ..., mode: max|min}."
            )
        return resolved
    if not isinstance(value, Mapping):
        raise TypeError("objective must be a registered metric name or a mapping.")
    raw = dict(value)
    practical_margin = raw.pop("practical_margin", None)
    if practical_margin is not None and (
        isinstance(practical_margin, bool)
        or not isinstance(practical_margin, int | float)
        or not math.isfinite(float(practical_margin))
        or float(practical_margin) < 0
    ):
        raise ValueError("objective.practical_margin must be finite and non-negative.")
    normalized = ObjectiveUtility.normalize(raw)
    if "metrics" not in normalized and "range" not in normalized:
        standard = MetricRegistry.resolve(str(normalized.get("metric", "")))
        if standard is not None and standard.get("mode") == normalized.get("mode"):
            known_range = standard.get("range")
            if isinstance(known_range, list):
                normalized["range"] = list(known_range)
    if practical_margin is not None:
        normalized["practical_margin"] = float(practical_margin)
    return normalized


def _signature_errors(target: type[Work], configured: Mapping[str, Any]) -> list[str]:
    signature = inspect.signature(target.run)
    parameters = tuple(signature.parameters.values())[1:]
    if any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
        return [f"{target.__name__}.run() cannot use *args; YAML parameters must be named."]
    accepted = {parameter.name: parameter for parameter in parameters}
    unknown = sorted(set(configured) - set(accepted))
    errors = [
        f"Unknown parameter {name!r}. {target.__name__}.run() accepts: "
        + ", ".join(_parameter_display(parameter) for parameter in parameters)
        for name in unknown
    ]
    missing = [
        parameter.name
        for parameter in parameters
        if parameter.default is inspect.Parameter.empty
        and parameter.kind not in {inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL}
        and parameter.name not in configured
    ]
    if missing:
        errors.append(f"Missing required parameter(s) for {target.__name__}.run(): {missing}.")
    hints = _hints(target.run)
    for name, value in configured.items():
        if name not in accepted or _is_marker(value):
            continue
        annotation = hints.get(name, accepted[name].annotation)
        if annotation is not inspect.Parameter.empty and not _obvious_type(value, annotation):
            errors.append(
                f"Parameter {name!r} expects {_type_name(annotation)} but YAML supplies "
                f"{type(value).__name__}."
            )
    return errors


def _markers(value: Any, prefix: str = "") -> list[tuple[str, Mapping[str, Any]]]:
    found: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(value, Mapping):
        if _is_marker(value):
            found.append((prefix, value))
        else:
            for key, item in value.items():
                found.extend(_markers(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_markers(item, f"{prefix}[{index}]"))
    return found


def _is_marker(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and len(value) == 1
        and next(iter(value), None) in {"file", "dataset", "from"}
    )


def _obvious_type(value: Any, annotation: Any) -> bool:
    if annotation is Any:
        return True
    origin = get_origin(annotation)
    if origin in {Union, types.UnionType}:
        return any(_obvious_type(value, nested) for nested in get_args(annotation))
    if origin is not None:
        return isinstance(value, origin) if isinstance(origin, type) else True
    if annotation is float:
        return isinstance(value, int | float) and not isinstance(value, bool)
    if annotation is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if annotation in {str, bool, list, dict}:
        return isinstance(value, annotation)
    if annotation is Path:
        return isinstance(value, str | Path)
    return True


def _hints(method: Any) -> dict[str, Any]:
    try:
        return get_type_hints(method)
    except (NameError, TypeError):
        return {}


def _parameter_display(parameter: inspect.Parameter) -> str:
    annotation = _type_name(parameter.annotation)
    default = "" if parameter.default is inspect.Parameter.empty else f" = {parameter.default!r}"
    return f"{parameter.name}: {annotation}{default}"


def _type_name(value: Any) -> str:
    if value is inspect.Parameter.empty:
        return "Any"
    return getattr(value, "__name__", str(value).replace("typing.", ""))


def _nonempty(value: Any, field_name: str) -> str:
    selected = str(value or "").strip()
    if not selected:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return selected


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    return copy.deepcopy(value)
