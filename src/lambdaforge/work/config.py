"""Strict, intentionally small YAML model for Work execution."""

from __future__ import annotations

import copy
import importlib
import inspect
import itertools
import os
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.work.models import immutable_mapping
from lambdaforge.work.Work import Work

_TOP_FIELDS = frozenset(
    {"name", "run", "with", "resources", "seeds", "search", "objective", "steps"}
)
_RUN_FIELDS = frozenset({"name", "run", "with", "resources", "seeds", "search", "objective"})


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
    objective: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", immutable_mapping(self.parameters))
        object.__setattr__(
            self,
            "variants",
            tuple(immutable_mapping(variant) for variant in self.variants),
        )
        if self.objective is not None:
            object.__setattr__(self, "objective", immutable_mapping(self.objective))

    @property
    def run_count(self) -> int:
        return len(self.seeds) * len(self.variants)


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
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
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
        name = _nonempty(data.get("name"), "name")
        source_path = Path(source).expanduser().resolve() if source is not None else None
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
                        )
                        for offset, item in enumerate(parallel, 1)
                    )
                else:
                    definitions = (
                        _run_definition(
                            step,
                            default_name=f"step-{index}",
                            inherited_resources=default_resources,
                        ),
                    )
                duplicates = seen.intersection(run.name for run in definitions)
                if duplicates:
                    raise ValueError(f"Workflow step names must be unique: {sorted(duplicates)}.")
                seen.update(run.name for run in definitions)
                levels_list.append(WorkLevel(definitions))
            levels = tuple(levels_list)
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
            return WorkValidationReport(source, None, False, (f"{type(error).__name__}: {error}",))
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
                    }
                )
            levels.append(current)
        return {
            "name": self.name,
            "levels": levels,
            "planned_runs": self.planned_runs,
            "resources": self.resources.to_dict(),
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
    return RunDefinition(
        name,
        work_class,
        dict(parameters),
        resources,
        _seeds(data.get("seeds")),
        _variants(data.get("search")),
        _objective(data.get("objective")),
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


def _variants(value: Any) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ({},)
    if not isinstance(value, Mapping) or not value:
        raise TypeError("search must map run parameter names to values/ranges.")
    finite_names: list[str] = []
    finite_values: list[tuple[Any, ...]] = []
    random_space: dict[str, dict[str, Any]] = {}
    random_count = 20
    for raw_name, raw_descriptor in value.items():
        name = str(raw_name)
        if name == "trials":
            random_count = int(raw_descriptor)
            continue
        descriptor = (
            raw_descriptor if isinstance(raw_descriptor, Mapping) else {"values": raw_descriptor}
        )
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
    has_range = any(
        "range" in descriptor for descriptor in value.values() if isinstance(descriptor, Mapping)
    )
    if has_range:
        if random_count < 1:
            raise ValueError("search.trials must be >= 1.")
        from lambdaforge.hpo.RandomSearch import RandomSearch

        return tuple(
            dict(trial.parameters) for trial in RandomSearch(random_space).trials(random_count)
        )
    return tuple(
        dict(zip(finite_names, combination, strict=True))
        for combination in itertools.product(*finite_values)
    )


def _objective(value: Any) -> Mapping[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"metric", "mode"}:
        raise ValueError("objective must contain exactly metric and mode.")
    metric = _nonempty(value["metric"], "objective.metric")
    mode = str(value["mode"]).lower()
    if mode not in {"min", "max"}:
        raise ValueError("objective.mode must be min or max.")
    return {"metric": metric, "mode": mode}


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
