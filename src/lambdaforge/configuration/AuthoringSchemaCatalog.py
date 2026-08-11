"""Packaged Schema for the concise authoring layer."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from importlib.resources import files
from typing import Any

from jsonschema import Draft202012Validator


class AuthoringSchemaCatalog:
    """Expose the formal Authoring Schema 1.0 used before strict materialization."""

    CURRENT_VERSION = "1.0"

    def __init__(self) -> None:
        self._schema: dict[str, Any] | None = None

    def schema(self) -> dict[str, Any]:
        """Return a detached Schema document."""
        if self._schema is None:
            resource = files("lambdaforge").joinpath("schemas/authoring.schema.json")
            with resource.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            if not isinstance(value, dict):
                raise TypeError("Authoring Schema must be a JSON object.")
            Draft202012Validator.check_schema(value)
            self._schema = value
        return copy.deepcopy(self._schema)

    def validation_errors(self, value: Mapping[str, Any]) -> tuple[str, ...]:
        """Return stable errors for authoring-layer structure."""
        errors = Draft202012Validator(self.schema()).iter_errors(value)
        return tuple(
            f"schema {'.'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in sorted(errors, key=lambda item: list(item.absolute_path))
        )
