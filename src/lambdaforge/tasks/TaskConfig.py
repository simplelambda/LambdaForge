"""Immutable configuration object for one generic task YAML document."""

from __future__ import annotations

import copy
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from lambdaforge.configuration.AuthoringConfigNormalizer import AuthoringConfigNormalizer
from lambdaforge.configuration.ConfigurationKind import ConfigurationKind
from lambdaforge.configuration.ResolvedConfiguration import ResolvedConfiguration
from lambdaforge.data.DataCatalog import DataCatalog
from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping
from lambdaforge.experiments.migrations.RoundTripYamlCodec import RoundTripYamlCodec
from lambdaforge.reproducibility.CodeIdentity import CodeIdentity
from lambdaforge.tasks.TaskFingerprint import TaskFingerprint
from lambdaforge.tasks.TaskInput import TaskInput
from lambdaforge.tasks.TaskSchemaCatalog import TaskSchemaCatalog


class TaskConfig(Mapping[str, Any]):
    """Own strict task configuration, source-relative paths and stable identity."""

    DEFAULT_OUTPUT_ROOT = Path("runs/tasks")

    def __init__(
        self,
        data: Mapping[str, Any],
        source: str | Path | None = None,
        *,
        resolution: ResolvedConfiguration | None = None,
    ) -> None:
        if not isinstance(data, Mapping):
            raise TypeError("Task config must be a mapping.")
        self.source = Path(source).resolve() if source is not None else None
        self.resolution = resolution
        materialized = AuthoringConfigNormalizer().normalize(data, source=self.source)
        if materialized.kind is not ConfigurationKind.TASK:
            raise ValueError("Configuration does not describe a generic task.")
        self._data = FrozenJsonMapping(materialized.values)
        self._resolved_inputs: tuple[TaskInput, ...] | None = None
        self._fingerprint: str | None = None
        self._validate_identity()

    @classmethod
    def from_yaml(cls, path: str | Path) -> TaskConfig:
        """Load one safely composed, duplicate-key-safe UTF-8 task document."""
        from lambdaforge.configuration.ConfigurationComposer import ConfigurationComposer

        source = Path(path).resolve()
        resolution = ConfigurationComposer().resolve(source)
        return cls(resolution.values, source=source, resolution=resolution)

    @classmethod
    def is_task_file(cls, path: str | Path) -> bool:
        """Return whether a YAML root is an explicit or concise task document."""
        try:
            loaded, _ = RoundTripYamlCodec().load_file(path)
            detected = AuthoringConfigNormalizer().detect(loaded)
            if detected is ConfigurationKind.TASK:
                return True
            if "extends" in loaded or "include" in loaded:
                from lambdaforge.configuration.ConfigurationComposer import (
                    ConfigurationComposer,
                )

                return ConfigurationComposer().resolve(path).values.get("kind") == "task"
            return False
        except Exception:
            return False

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def name(self) -> str:
        """Return the validated task name."""
        return str(self._data["name"])

    @property
    def source_dir(self) -> Path:
        """Return the YAML directory or the current directory for mapping configs."""
        return self.source.parent if self.source is not None else Path.cwd().resolve()

    @property
    def fingerprint(self) -> str:
        """Return the scientific identity of this task definition."""
        if self._fingerprint is None:
            materialized = self.as_dict()
            materialized["_resolved_inputs"] = [
                {
                    "name": task_input.name,
                    "dataset_reference": task_input.dataset_reference,
                    "identity": task_input.identity.to_dict(),
                }
                for task_input in self.resolved_inputs
            ]
            materialized["_code_identity"] = self.code_identity.to_dict()
            self._fingerprint = TaskFingerprint.digest(materialized)
        return self._fingerprint

    @property
    def code_identity(self) -> CodeIdentity:
        """Return automatic Git or explicit consumer-code identity."""
        explicit = self._authoring.get("code_version")
        return CodeIdentity.capture(
            self.source_dir,
            explicit_version=str(explicit) if explicit is not None else None,
        )

    def scientific_payload(self) -> dict[str, Any]:
        """Return the exact normalized values behind the task fingerprint."""
        materialized = self.as_dict()
        materialized["_resolved_inputs"] = [
            {
                "name": task_input.name,
                "dataset_reference": task_input.dataset_reference,
                "identity": task_input.identity.to_dict(),
            }
            for task_input in self.resolved_inputs
        ]
        materialized["_code_identity"] = self.code_identity.to_dict()
        return TaskFingerprint.payload(materialized)

    @property
    def resolved_inputs(self) -> tuple[TaskInput, ...]:
        """Return content-addressed local inputs in configuration order."""
        if self._resolved_inputs is None:
            catalog = self.data_catalog
            self._resolved_inputs = tuple(
                TaskInput.materialize(
                    value,
                    self.source_dir,
                    index,
                    catalog=catalog,
                    environment=self.environment,
                )
                for index, value in enumerate(self._data.get("inputs", ()))
            )
            names = [value.name for value in self._resolved_inputs]
            if len(names) != len(set(names)):
                raise ValueError("Task input names must be unique.")
        return self._resolved_inputs

    @property
    def outputs(self) -> Mapping[str, str]:
        """Return logical output names mapped to safe run-relative paths."""
        authoring = self._authoring
        values = authoring.get("outputs", {})
        if not isinstance(values, Mapping):
            return {}
        return {str(name): str(path) for name, path in values.items()}

    @property
    def environment(self) -> str:
        """Return the catalogue environment used for local materialization."""
        return str(self._authoring.get("environment", "local"))

    @property
    def data_catalog(self) -> DataCatalog | None:
        """Load the explicitly configured data catalogue, if any."""
        configured = self._authoring.get("data_catalog")
        if configured is None:
            return None
        path = Path(str(configured))
        path = path if path.is_absolute() else (self.source_dir / path).resolve()
        return DataCatalog.from_yaml(path)

    @property
    def _authoring(self) -> Mapping[str, Any]:
        extensions = self._data.get("extensions", {})
        if not isinstance(extensions, Mapping):
            return {}
        value = extensions.get("authoring", {})
        return value if isinstance(value, Mapping) else {}

    @property
    def suite_dir(self) -> Path:
        """Return the task root shared by every scientific configuration."""
        configured = Path(str(self._data.get("output_root", self.DEFAULT_OUTPUT_ROOT)))
        root = configured if configured.is_absolute() else self.source_dir / configured
        return root.resolve() / self.slugify(self.name)

    @property
    def run_dir(self) -> Path:
        """Return the content-addressed directory for this scientific task."""
        digest = self.fingerprint.removeprefix("sha256:")[:16]
        return self.suite_dir / digest

    @property
    def resume(self) -> bool:
        """Return whether a matching partial attempt may reuse task-owned state."""
        return bool(self._data.get("resume", True))

    @property
    def rerun_completed(self) -> bool:
        """Return whether a valid successful task should execute again."""
        return bool(self._data.get("rerun_completed", False))

    @property
    def resources(self) -> ResourceRequest:
        """Return the portable resource request from concise authoring metadata."""
        value = self._authoring.get("resources", {})
        if not isinstance(value, Mapping):
            raise TypeError("Task resources must be a mapping.")
        return ResourceRequest.from_mapping(value)

    def with_execution_policy(
        self,
        *,
        force: bool = False,
        restart: bool = False,
        no_resume: bool = False,
    ) -> TaskConfig:
        """Return a copy with explicit CLI lifecycle policy overrides."""
        values = self.as_dict()
        if force or restart:
            values["rerun_completed"] = True
        if restart or no_resume:
            values["resume"] = False
        return TaskConfig(values, source=self.source, resolution=self.resolution)

    @property
    def required_artifacts(self) -> tuple[str, ...]:
        """Return explicit run-relative completion requirements."""
        return tuple(str(value) for value in self._data.get("required_artifacts", ()))

    def as_dict(self) -> dict[str, Any]:
        """Return an independent YAML-compatible configuration mapping."""
        return copy.deepcopy(self._data)

    def redacted_dict(self) -> dict[str, Any]:
        """Return a persistable snapshot with every composed secret removed."""
        if self.resolution is None:
            return self.as_dict()
        return (
            AuthoringConfigNormalizer()
            .normalize(self.resolution.materialized(), source=self.source)
            .to_dict()
        )

    def schema_errors(self) -> tuple[str, ...]:
        """Validate this mapping against the packaged task Schema."""
        return TaskSchemaCatalog().validation_errors(self.redacted_dict())

    @staticmethod
    def slugify(value: str) -> str:
        """Convert a task name into one portable path segment."""
        normalized = value.strip().replace("\\", "/")
        normalized = re.sub(r"[^A-Za-z0-9_.=-]+", "-", normalized).strip("-")
        return normalized or "task"

    def _validate_identity(self) -> None:
        if self._data.get("kind") != "task":
            raise ValueError("Generic task documents require 'kind: task'.")
        if self._data.get("schema_version") != TaskSchemaCatalog.CURRENT_VERSION:
            raise ValueError(
                f"Task schema_version must be the quoted string "
                f"{TaskSchemaCatalog.CURRENT_VERSION!r}."
            )
        name = self._data.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Task name must be a non-empty string.")
