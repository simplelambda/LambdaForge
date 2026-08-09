"""Packaged JSON Schema loader for generic task documents."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from importlib.resources import files
from typing import Any

from jsonschema import Draft202012Validator


class TaskSchemaCatalog:
    """Load and validate the independent versioned task Schema family."""

    CURRENT_VERSION = "1.0"
    SCHEMA_FILE = "task.schema.json"

    def __init__(self) -> None:
        self._schema: dict[str, Any] | None = None
        self._validator: Draft202012Validator | None = None

    def schema(self) -> dict[str, Any]:
        """Return a defensive copy of the current task Schema."""
        return copy.deepcopy(self._cached_schema())

    def validation_errors(self, config: Mapping[str, Any]) -> tuple[str, ...]:
        """Return stable path-qualified task Schema errors."""
        if self._validator is None:
            self._validator = Draft202012Validator(self._cached_schema())
        errors: list[str] = []
        for error in sorted(
            self._validator.iter_errors(config),
            key=lambda item: list(item.absolute_path),
        ):
            path = ".".join(str(part) for part in error.absolute_path) or "<root>"
            errors.append(f"schema {path}: {error.message}")
        return tuple(errors)

    def _cached_schema(self) -> dict[str, Any]:
        if self._schema is not None:
            return self._schema
        resource = files("lambdaforge").joinpath("schemas").joinpath(self.SCHEMA_FILE)
        with resource.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise TypeError("The packaged task Schema must be a JSON object.")
        Draft202012Validator.check_schema(value)
        declared = value.get("properties", {}).get("schema_version", {}).get("const")
        if declared != self.CURRENT_VERSION:
            raise ValueError(
                f"Task Schema declares {declared!r}, expected {self.CURRENT_VERSION!r}."
            )
        self._schema = value
        return value
