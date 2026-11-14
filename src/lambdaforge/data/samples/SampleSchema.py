"""Versioned identified multi-asset sample contract, independent of a GUI."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lambdaforge.catalog.errors import ContractError
from lambdaforge.data.samples.FieldSpec import FieldSpec


class SampleSchema:
    """Fields, explicit model bindings and batching rules. No hidden learned transform."""

    def __init__(self, fields: list[FieldSpec], bindings: Mapping[str, str] | None = None) -> None:
        self.fields = tuple(fields)
        self.bindings = dict(bindings or {})
        names = {field.name for field in fields}
        if not fields or len(names) != len(fields):
            raise ContractError("INVALID_SCHEMA", "Sample fields must be nonempty and unique.")
        if any(value not in names for value in self.bindings.values()):
            raise ContractError("INVALID_BINDING", "Model bindings reference unknown fields.")
        by_name = {field.name: field for field in fields}
        if any(
            by_name[name].role not in {"input", "mask", "index", "coordinates", "vector"}
            for name in self.bindings.values()
        ):
            raise ContractError("INVALID_BINDING", "Targets/loss/metadata cannot be model inputs.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SampleSchema:
        if value.get("schema_version") != "sample-schema-v1" or set(value) - {
            "schema_version",
            "fields",
            "bindings",
        }:
            raise ContractError(
                "CONTRACT_VERSION", "Expected sample-schema-v1 with fields/bindings."
            )
        try:
            return cls([FieldSpec.from_mapping(v) for v in value["fields"]], value.get("bindings"))
        except (KeyError, TypeError, AttributeError) as error:
            raise ContractError("INVALID_SCHEMA", "Expected fields and named bindings.") from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "sample-schema-v1",
            "fields": [v.to_dict() for v in self.fields],
            "bindings": dict(self.bindings),
        }

    def model_inputs(self, fields: Mapping[str, Any]) -> dict[str, Any]:
        return {argument: fields[name] for argument, name in self.bindings.items()}
