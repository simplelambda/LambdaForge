"""Versioned table schema; scientific sample/asset contracts remain independent."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from lambdaforge.catalog.errors import ContractError
from lambdaforge.etl.tables.Column import Column


class TableSchema:
    """Explicit columns and keys; never infer scalar types from a small preview."""

    def __init__(self, columns: Sequence[Column], keys: Sequence[str] = ()) -> None:
        self.columns, self.keys = tuple(columns), tuple(keys)
        names = [c.name for c in self.columns]
        if not names or len(set(names)) != len(names) or set(keys) - set(names):
            raise ContractError(
                "INVALID_SCHEMA", "Unique columns and declared key columns required."
            )
        if len(set(keys)) != len(keys) or any(c.nullable for c in columns if c.name in keys):
            raise ContractError("INVALID_SCHEMA", "Key columns must be unique/nonnullable.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TableSchema:
        if (
            not isinstance(value, Mapping)
            or value.get("schema_version") != "table-schema-v1"
            or set(value)
            - {
                "schema_version",
                "columns",
                "keys",
            }
        ):
            raise ContractError("CONTRACT_VERSION", "Expected table-schema-v1.")
        try:
            columns = []
            for raw in value["columns"]:
                row = dict(raw)
                for key in ("null_values", "categories"):
                    if key in row:
                        row[key] = tuple(row[key])
                columns.append(Column(**row))
            return cls(columns, value.get("keys", ()))
        except (TypeError, KeyError, AttributeError) as error:
            raise ContractError("INVALID_SCHEMA", "Malformed table columns.") from error

    def convert(
        self, row: Mapping[str, Any], *, decimal: str = ".", retain_extra: bool = True
    ) -> dict[str, Any]:
        result = dict(row) if retain_extra else {}
        for column in self.columns:
            result[column.name] = column.convert(row.get(column.name), decimal=decimal)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "table-schema-v1",
            "columns": [c.to_dict() for c in self.columns],
            "keys": list(self.keys),
        }
