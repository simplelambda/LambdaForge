"""Side-effect-free validation for generic task configurations."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lambdaforge.experiments.ObjectFactory import ObjectFactory
from lambdaforge.plugins.PluginReference import PluginReference
from lambdaforge.plugins.PluginRegistry import PluginRegistry
from lambdaforge.tasks.TaskConfig import TaskConfig
from lambdaforge.tasks.TaskSchemaCatalog import TaskSchemaCatalog
from lambdaforge.tasks.TaskValidationReport import TaskValidationReport


class TaskValidator:
    """Validate task structure and imports without constructing user objects."""

    def __init__(
        self,
        plugins: PluginRegistry | None = None,
        schema_catalog: TaskSchemaCatalog | None = None,
    ) -> None:
        self.plugins = plugins or PluginRegistry.default()
        self.schema_catalog = schema_catalog or TaskSchemaCatalog()

    def validate_file(
        self,
        path: str | Path,
        *,
        check_imports: bool = True,
    ) -> TaskValidationReport:
        """Load and validate one task YAML without creating output paths."""
        source = Path(path).resolve()
        try:
            config = TaskConfig.from_yaml(source)
        except Exception as error:
            return TaskValidationReport(
                source=str(source),
                errors=(self._format_error(error),),
                imports_checked=False,
            )
        return self.validate(config, check_imports=check_imports)

    def validate(
        self,
        config: TaskConfig | Mapping[str, Any],
        *,
        source: str | Path | None = None,
        check_imports: bool = True,
    ) -> TaskValidationReport:
        """Return every discoverable task error in one immutable report."""
        data = config.redacted_dict() if isinstance(config, TaskConfig) else dict(config)
        resolved_source = (
            str(config.source)
            if isinstance(config, TaskConfig) and config.source is not None
            else str(source)
            if source is not None
            else None
        )
        errors = list(self.schema_catalog.validation_errors(data))
        warnings: list[str] = []
        normalized: TaskConfig | None = None
        if not errors:
            errors.extend(self._path_errors(data))
            try:
                normalized = TaskConfig(data, source=resolved_source)
                _ = normalized.resolved_inputs
            except Exception as error:
                errors.append(self._format_error(error))
        if normalized is not None and not errors:
            errors.extend(self._built_in_preprocessing_path_errors(data, normalized))
        if check_imports:
            errors.extend(self._import_errors(data))
            if not errors:
                contract_errors, contract_warnings = self._task_contract(
                    data["task"],
                    has_inputs=bool(data.get("inputs")),
                )
                errors.extend(contract_errors)
                warnings.extend(contract_warnings)
        else:
            warnings.append("Import references and task constructor contracts were not checked.")
        return TaskValidationReport(
            source=resolved_source,
            errors=tuple(dict.fromkeys(errors)),
            warnings=tuple(dict.fromkeys(warnings)),
            imports_checked=check_imports,
            schema_version=(
                str(data.get("schema_version")) if data.get("schema_version") is not None else None
            ),
        )

    def _import_errors(self, value: Any, path: str = "<root>") -> list[str]:
        errors: list[str] = []
        if isinstance(value, Mapping):
            if "plugin" in value and isinstance(value["plugin"], Mapping):
                try:
                    self.plugins.resolve(
                        PluginReference.from_value(value["plugin"]),
                        record_usage=False,
                    )
                except Exception as error:
                    errors.append(f"import {path}.plugin: {self._format_error(error)}")
            for key in ("target", "ref"):
                if key in value and isinstance(value[key], str):
                    try:
                        ObjectFactory.import_object(value[key])
                    except Exception as error:
                        errors.append(f"import {path}.{key}: {self._format_error(error)}")
            for key, item in value.items():
                errors.extend(self._import_errors(item, f"{path}.{key}"))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, item in enumerate(value):
                errors.extend(self._import_errors(item, f"{path}[{index}]"))
        return errors

    def _task_contract(
        self,
        spec: Mapping[str, Any],
        *,
        has_inputs: bool,
    ) -> tuple[list[str], list[str]]:
        errors: list[str] = []
        warnings: list[str] = []
        if "plugin" in spec:
            return errors, warnings
        target = ObjectFactory.import_object(str(spec["target"]))
        if not isinstance(target, type):
            errors.append("task.target must resolve to a class so construction remains explicit.")
            return errors, warnings
        if not callable(getattr(target, "run", None)):
            errors.append("task.target must expose a callable run method.")
        if (
            target.__module__ == "lambdaforge.preprocessing.PreprocessingTask"
            and target.__name__ == "PreprocessingTask"
            and not has_inputs
        ):
            errors.append(
                "Built-in PreprocessingTask requires at least one top-level 'inputs' entry so "
                "mutable source content participates in task identity."
            )
        params = spec.get("params", {})
        try:
            inspect.signature(target).bind(**dict(params))
        except TypeError as error:
            errors.append(
                f"task.params do not match {target.__module__}.{target.__name__}: {error}"
            )
        except ValueError as error:
            warnings.append(f"Could not inspect task constructor signature: {error}")
        return errors, warnings

    @staticmethod
    def _path_errors(data: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []
        for index, value in enumerate(data.get("required_artifacts", ())):
            path = Path(str(value))
            if path.is_absolute() or ".." in path.parts:
                errors.append(
                    f"required_artifacts[{index}] must be relative and cannot traverse parents."
                )
        return errors

    @staticmethod
    def _built_in_preprocessing_path_errors(
        data: Mapping[str, Any],
        config: TaskConfig,
    ) -> list[str]:
        task = data.get("task", {})
        if not isinstance(task, Mapping) or task.get("target") != (
            "lambdaforge.preprocessing.PreprocessingTask"
        ):
            return []
        params = task.get("params", {})
        source = params.get("source", {}) if isinstance(params, Mapping) else {}
        if not isinstance(source, Mapping):
            return []
        source_params = source.get("params", {})
        if not isinstance(source_params, Mapping):
            return []
        target = source.get("target")
        field = (
            "path"
            if target == "lambdaforge.preprocessing.JsonLinesSource"
            else "root"
            if target == "lambdaforge.preprocessing.FileTreeSource"
            else None
        )
        if field is None or field not in source_params:
            return []
        configured = Path(str(source_params[field]))
        candidate = (
            configured if configured.is_absolute() else config.source_dir / configured
        ).resolve(strict=False)
        declared = tuple(Path(value.resolved_path) for value in config.resolved_inputs)
        if not declared:
            return []
        if any(candidate == root or candidate.is_relative_to(root) for root in declared):
            return []
        return [
            f"Built-in preprocessing source {field}={str(configured)!r} is not covered by a "
            "top-level content-addressed input."
        ]

    @staticmethod
    def _format_error(error: Exception) -> str:
        return f"{error.__class__.__name__}: {error}"
