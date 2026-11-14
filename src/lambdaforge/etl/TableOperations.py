"""Bounded scalar table transformations callable from ordinary consumer Python."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Literal

from lambdaforge.catalog.errors import ContractError
from lambdaforge.etl.PreviewBudget import PreviewBudget
from lambdaforge.etl.tables.predicates import matches, validate_predicate
from lambdaforge.etl.tables.TableSchema import TableSchema

PROTECTED = {"id", "partitions", "provenance"}


class CastTable:
    """Convert declared columns, preserving identity/partition/provenance verbatim."""

    def __init__(self, schema: Mapping[str, Any]) -> None:
        self.schema = TableSchema.from_mapping(schema)
        if any(c.name in PROTECTED for c in self.schema.columns):
            raise ContractError(
                "IDENTITY_CHANGE", "Casting protected fields requires a new declared design."
            )

    def __call__(self, rows: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
        for row in rows:
            yield self.schema.convert(row)


class FillNulls:
    """Constant fill only, not a learned imputer and never fitting validation/test."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        self.values = dict(values)
        json.dumps(self.values, allow_nan=False)
        if PROTECTED & set(values):
            raise ContractError(
                "IDENTITY_CHANGE", "Cannot fill protected identity/partition fields."
            )

    def __call__(self, rows: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
        for row in rows:
            if set(self.values) - set(row):
                raise ContractError("MISSING_FIELD", "Fill references absent columns.")
            yield {**row, **{k: v for k, v in self.values.items() if row[k] is None}}


class FilterTable:
    """Keep/drop rows using a bounded declarative scalar predicate, never eval."""

    def __init__(self, predicate: Mapping[str, Any], keep: bool = True) -> None:
        validate_predicate(predicate)
        self.predicate, self.keep = dict(predicate), keep

    def __call__(self, rows: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
        for row in rows:
            if matches(row, self.predicate) == self.keep:
                yield dict(row)


class SelectTable:
    """Column projection/drop/rename; always preserve protected fields unchanged."""

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        drop: Sequence[str] = (),
        rename: Mapping[str, str] | None = None,
    ) -> None:
        self.columns, self.drop, self.rename = columns, tuple(drop), dict(rename or {})
        if PROTECTED & (set(drop) | set(self.rename) | set(self.rename.values())):
            raise ContractError("IDENTITY_CHANGE", "Cannot drop/rename protected columns.")
        if len(set(self.rename.values())) != len(self.rename):
            raise ContractError("COLUMN_COLLISION", "Renames must be unique.")

    def __call__(self, rows: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
        for row in rows:
            names = list(self.columns) if self.columns is not None else list(row)
            names += [k for k in PROTECTED if k in row and k not in names]
            selected = [k for k in names if k not in self.drop]
            if set(selected) - set(row):
                raise ContractError("MISSING_FIELD", "Unknown projection column.")
            result = {self.rename.get(k, k): row[k] for k in selected}
            if len(result) != len(selected):
                raise ContractError("COLUMN_COLLISION", "Rename overwrites an existing column.")
            yield result


def _bounded(
    rows: Iterable[Mapping[str, Any]], max_rows: int, max_bytes: int
) -> list[dict[str, Any]]:
    budget = PreviewBudget(
        max_rows=max_rows, max_bytes=max_bytes, max_memory=max_bytes, max_seconds=30
    )
    result, size = [], 0
    for row in rows:
        budget.row(row)
        size += budget.size(row)
        if size > max_bytes:
            raise ContractError("READ_LIMIT", "Materialized table exceeds allocation guard.")
        result.append(dict(row))
    return result


class SortTable:
    """Stable sort with explicit null position; bounded materialization, not streaming sort."""

    def __init__(
        self,
        by: Sequence[str],
        descending: bool = False,
        nulls: Literal["first", "last"] = "last",
        max_rows: int = 10_000,
        max_bytes: int = 8 * 1024**2,
    ) -> None:
        if not by or nulls not in {"first", "last"}:
            raise ContractError("INVALID_PARAMETERS", "Sort requires columns/null policy.")
        self.by, self.descending, self.nulls = tuple(by), descending, nulls
        self.max_rows, self.max_bytes = max_rows, max_bytes

    def __call__(self, rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        result = _bounded(rows, self.max_rows, self.max_bytes)
        try:
            for name in reversed(self.by):
                nonnull = [r for r in result if r[name] is not None]
                missing = [r for r in result if r[name] is None]
                nonnull.sort(key=lambda r: r[name], reverse=self.descending)
                result = missing + nonnull if self.nulls == "first" else nonnull + missing
        except (TypeError, KeyError):
            raise ContractError("INVALID_DATA", "Sort keys are missing or incomparable.") from None
        return result


class DeduplicateTable:
    """Keep first/last exact key, refusing evidence loss from different identified rows."""

    def __init__(
        self,
        keys: Sequence[str],
        keep: Literal["first", "last"] = "first",
        max_rows: int = 10_000,
        max_bytes: int = 8 * 1024**2,
    ) -> None:
        if not keys or keep not in {"first", "last"}:
            raise ContractError("INVALID_PARAMETERS", "Explicit dedup keys/keep policy required.")
        self.keys, self.keep = tuple(keys), keep
        self.max_rows, self.max_bytes = max_rows, max_bytes

    def __call__(self, rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for row in _bounded(rows, self.max_rows, self.max_bytes):
            key = json.dumps([row[k] for k in self.keys], sort_keys=True, allow_nan=False)
            old = result.get(key)
            if old is not None and any(old.get(k) != row.get(k) for k in PROTECTED):
                raise ContractError(
                    "IDENTITY_CHANGE", "Dedup would merge distinct identity/partition evidence."
                )
            if old is None or self.keep == "last":
                result[key] = row
        return list(result.values())


class AggregateTable:
    """Explicit derived groups; partitions cannot mix. Original member IDs remain lineage.

    No silent survivor merge: this creates a new table with deterministic derived ids and
    full bounded member id lineage. Null numeric observations are excluded, not zeros.
    """

    def __init__(
        self,
        by: Sequence[str],
        aggregations: Mapping[str, Mapping[str, str]],
        identity: Literal["derive"],
        max_rows: int = 10_000,
        max_bytes: int = 8 * 1024**2,
    ) -> None:
        if identity != "derive" or not aggregations or set(aggregations) & PROTECTED:
            raise ContractError(
                "IDENTITY_CHANGE", "Aggregation needs explicit derived identity policy."
            )
        if any(
            set(v) != {"column", "op"} or v["op"] not in {"count", "sum", "mean", "min", "max"}
            for v in aggregations.values()
        ):
            raise ContractError("INVALID_PARAMETERS", "Invalid column aggregation.")
        self.by, self.aggregations = tuple(by), dict(aggregations)
        self.max_rows, self.max_bytes = max_rows, max_bytes

    def __call__(self, rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        seen_ids = set()
        for row in _bounded(rows, self.max_rows, self.max_bytes):
            if "id" not in row:
                raise ContractError("MISSING_ID", "Derived groups require identified input rows.")
            if row["id"] in seen_ids:
                raise ContractError(
                    "DUPLICATE_ID", "Aggregation cannot duplicate identified evidence."
                )
            seen_ids.add(row["id"])
            key = json.dumps([[row[k] for k in self.by], row.get("partitions", {})], sort_keys=True)
            grouped.setdefault(key, []).append(row)
        result = []
        for key, members in grouped.items():
            output = {k: members[0][k] for k in self.by}
            for name, spec in self.aggregations.items():
                values = [r[spec["column"]] for r in members if r[spec["column"]] is not None]
                if spec["op"] == "count":
                    value: Any = len(values)
                elif not values:
                    value = None
                else:
                    if any(type(v) not in (int, float) for v in values):
                        raise ContractError(
                            "INVALID_DATA", "Numeric aggregation requires scalar numeric values."
                        )
                    if spec["op"] == "sum":
                        value = sum(values)
                    elif spec["op"] == "mean":
                        value = sum(values) / len(values)
                    elif spec["op"] == "min":
                        value = min(values)
                    else:
                        value = max(values)
                output[name] = value
            output.update(
                id="aggregate-" + hashlib.sha256(key.encode()).hexdigest(),
                partitions=members[0].get("partitions", {}),
                provenance={"operation": "aggregate", "member_ids": [r["id"] for r in members]},
            )
            result.append(output)
        return result


class ConcatTables:
    """Exact-schema concatenation; refuse duplicate ids rather than inventing new identities."""

    def __init__(self, max_rows: int = 10_000, max_bytes: int = 8 * 1024**2) -> None:
        self.max_rows, self.max_bytes = max_rows, max_bytes

    def __call__(self, tables: Iterable[Iterable[Mapping[str, Any]]]) -> list[dict[str, Any]]:
        rows = _bounded((row for table in tables for row in table), self.max_rows, self.max_bytes)
        names = set(rows[0]) if rows else set()
        if any(set(r) != names for r in rows):
            raise ContractError("INVALID_SCHEMA", "Concatenation requires exact column sets.")
        ids = [r["id"] for r in rows if "id" in r]
        if len(set(ids)) != len(ids):
            raise ContractError("DUPLICATE_ID", "Concatenation would duplicate identified rows.")
        return rows


class FromRows:
    """Create a validated bounded table from explicitly declared portable rows, not eval."""

    def __init__(
        self, schema: Mapping[str, Any], max_rows: int = 10_000, max_bytes: int = 8 * 1024**2
    ) -> None:
        self.schema = TableSchema.from_mapping(schema)
        self.max_rows, self.max_bytes = max_rows, max_bytes

    def __call__(self, rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        result = _bounded((self.schema.convert(r) for r in rows), self.max_rows, self.max_bytes)
        if self.schema.keys:
            keys = [json.dumps([r[k] for k in self.schema.keys], sort_keys=True) for r in result]
            if len(set(keys)) != len(keys):
                raise ContractError("DUPLICATE_ID", "Declared table keys are not unique.")
        return result
