"""Packaged JSON Schema loader for dataset recipes."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from importlib.resources import files
from typing import Any

from jsonschema import Draft202012Validator


class DatasetRecipeSchemaCatalog:
    """Validate the independent `kind: dataset` authoring family."""

    CURRENT_VERSION = "1.0"

    def __init__(self) -> None:
        resource = files("lambdaforge").joinpath("schemas/dataset.schema.json")
        with resource.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise TypeError("Dataset recipe Schema must contain an object.")
        Draft202012Validator.check_schema(value)
        self._schema = value
        self._validator = Draft202012Validator(value)

    def schema(self) -> dict[str, Any]:
        """Return a defensive schema copy."""
        return copy.deepcopy(self._schema)

    def validation_errors(self, value: Mapping[str, Any]) -> tuple[str, ...]:
        """Return stable path-qualified validation errors."""
        return tuple(
            f"schema {'.'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in sorted(
                self._validator.iter_errors(value), key=lambda item: list(item.absolute_path)
            )
        )
