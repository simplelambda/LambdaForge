"""Version-to-resource catalog for packaged experiment JSON Schemas."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from importlib.resources import files
from typing import Any, ClassVar

from jsonschema import Draft202012Validator

from lambdaforge.experiments.migrations.ExperimentSchemaVersion import (
    ExperimentSchemaVersion,
)


class ExperimentSchemaCatalog:
    """Load exact packaged Schemas and keep their version declarations aligned."""

    SCHEMA_DIRECTORY: ClassVar[str] = "schemas"
    DEFAULT_SCHEMA_FILES: ClassVar[dict[str, str]] = {
        "1.0": "experiment-1.0.schema.json",
        ExperimentSchemaVersion.CURRENT_VALUE: "experiment.schema.json",
    }

    def __init__(self, schema_files: Mapping[str, str] | None = None) -> None:
        self._schema_files = dict(
            self.DEFAULT_SCHEMA_FILES if schema_files is None else schema_files
        )
        if ExperimentSchemaVersion.CURRENT_VALUE not in self._schema_files:
            raise ValueError("The Schema catalog must contain the current version.")
        self._schema_cache: dict[ExperimentSchemaVersion, dict[str, Any]] = {}
        self._validator_cache: dict[ExperimentSchemaVersion, Draft202012Validator] = {}

    @property
    def current_version(self) -> ExperimentSchemaVersion:
        """Return the framework's current experiment Schema version."""
        return ExperimentSchemaVersion.current()

    @property
    def supported_versions(self) -> tuple[ExperimentSchemaVersion, ...]:
        """Return packaged Schema versions in forward order."""
        return tuple(sorted(ExperimentSchemaVersion(value) for value in self._schema_files))

    def schema(self, version: ExperimentSchemaVersion | str | None = None) -> dict[str, Any]:
        """Load and defensively copy one exact Draft 2020-12 Schema."""
        resolved = (
            self.current_version
            if version is None
            else version
            if isinstance(version, ExperimentSchemaVersion)
            else ExperimentSchemaVersion(version)
        )
        if resolved.is_unversioned:
            raise ValueError("Unversioned input has no standalone JSON Schema.")
        return copy.deepcopy(self._cached_schema(resolved))

    def validation_errors(
        self,
        config: Mapping[str, Any],
        version: ExperimentSchemaVersion | str | None = None,
    ) -> tuple[str, ...]:
        """Return stable path-qualified Schema errors without importing user code."""
        resolved = (
            self.current_version
            if version is None
            else version
            if isinstance(version, ExperimentSchemaVersion)
            else ExperimentSchemaVersion(version)
        )
        validator = self._validator_cache.get(resolved)
        if validator is None:
            validator = Draft202012Validator(self._cached_schema(resolved))
            self._validator_cache[resolved] = validator
        errors: list[str] = []
        for error in sorted(
            validator.iter_errors(config),
            key=lambda item: list(item.absolute_path),
        ):
            path = ".".join(str(part) for part in error.absolute_path) or "<root>"
            errors.append(f"schema {path}: {error.message}")
        return tuple(errors)

    def _cached_schema(
        self,
        version: ExperimentSchemaVersion,
    ) -> dict[str, Any]:
        cached = self._schema_cache.get(version)
        if cached is not None:
            return cached
        if version.is_unversioned:
            raise ValueError("Unversioned input has no standalone JSON Schema.")
        filename = self._schema_files.get(version.value)
        if filename is None:
            raise ValueError(f"No packaged JSON Schema for version {version}.")
        resource = files("lambdaforge").joinpath(self.SCHEMA_DIRECTORY).joinpath(filename)
        with resource.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise TypeError(f"Packaged Schema {filename!r} must be a JSON object.")
        Draft202012Validator.check_schema(value)
        declared = value.get("properties", {}).get("schema_version", {}).get("const")
        if declared != version.value:
            raise ValueError(
                f"Schema resource {filename!r} declares {declared!r}, expected {version.value!r}."
            )
        self._schema_cache[version] = value
        return value
