"""Packaged workflow JSON Schema loader."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from importlib.resources import files
from typing import Any

from jsonschema import Draft202012Validator


class WorkflowSchemaCatalog:
    """Load and validate the independent workflow Schema family."""

    CURRENT_VERSION = "1.0"
    SCHEMA_FILE = "workflow.schema.json"

    def __init__(self) -> None:
        resource = files("lambdaforge").joinpath("schemas").joinpath(self.SCHEMA_FILE)
        with resource.open("r", encoding="utf-8") as handle:
            schema = json.load(handle)
        if not isinstance(schema, dict):
            raise TypeError("The packaged workflow Schema must be a JSON object.")
        Draft202012Validator.check_schema(schema)
        self._schema = schema
        self._validator = Draft202012Validator(schema)

    def schema(self) -> dict[str, Any]:
        """Return a defensive Schema copy."""
        return copy.deepcopy(self._schema)

    def validation_errors(self, config: Mapping[str, Any]) -> tuple[str, ...]:
        """Return stable path-qualified workflow errors."""
        errors: list[str] = []
        for error in sorted(
            self._validator.iter_errors(config),
            key=lambda item: list(item.absolute_path),
        ):
            path = ".".join(str(part) for part in error.absolute_path) or "<root>"
            errors.append(f"schema {path}: {error.message}")
        return tuple(errors)
