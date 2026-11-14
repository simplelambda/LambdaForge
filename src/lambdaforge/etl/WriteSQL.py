"""Explicit bounded transactional table writes, never run by native preview APIs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from lambdaforge.catalog.errors import ContractError
from lambdaforge.etl.PreviewBudget import PreviewBudget
from lambdaforge.etl.sql_backend import connection, identifier, validate_connection
from lambdaforge.etl.SQLWritePolicy import SQLWritePolicy
from lambdaforge.etl.tables.TableSchema import TableSchema


class WriteSQL:
    """insert rejects duplicate keys; upsert is idempotent on explicit actual unique keys.

    replace deletes rows transactionally, never drops tables. create requires separate DDL
    authority; no raw SQL/DDL interpolation. Existing target columns/unique constraints are
    checked by the database, conversion/key validation by TableSchema. All writes rollback
    together on parse/schema/key/budget/cancellation failure. No indefinite retries.
    """

    def __init__(
        self,
        dialect: Literal["sqlite", "postgresql"],
        table: str,
        schema: Mapping[str, Any],
        file: str | None = None,
        credential: str | None = None,
        mode: Literal["insert", "upsert", "replace"] = "insert",
        create: bool = False,
        timeout: float = 10,
        statement_timeout: float = 10,
        sslmode: Literal["verify-full", "verify-ca", "require"] = "verify-full",
    ) -> None:
        validate_connection(dialect, file, credential, timeout, statement_timeout, sslmode)
        if mode not in {"insert", "upsert", "replace"} or type(create) is not bool:
            raise ContractError("INVALID_PARAMETERS", "Explicit SQL write mode required.")
        self.dialect, self.table, self.schema = (
            dialect,
            identifier(table),
            TableSchema.from_mapping(schema),
        )
        self.file, self.credential, self.mode, self.create = file, credential, mode, create
        self.timeout, self.statement_timeout, self.sslmode = timeout, statement_timeout, sslmode
        if not self.schema.keys:
            raise ContractError("INVALID_SCHEMA", "Writers require explicit nonnullable keys.")
        for column in self.schema.columns:
            identifier(column.name)
            if column.type == "json":
                raise ContractError(
                    "UNSUPPORTED_VALUE",
                    "SQL JSON needs an explicit backend adapter; not silently flattened.",
                )

    def __call__(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        policy: SQLWritePolicy,
        budget: PreviewBudget | None = None,
    ) -> dict[str, Any]:
        if not isinstance(policy, SQLWritePolicy) or not policy.allow_insert:
            raise ContractError("EFFECT_FORBIDDEN", "SQL insertion requires operator policy.")
        if (
            self.mode == "upsert"
            and not policy.allow_update
            or self.mode == "replace"
            and not policy.allow_replace
        ):
            raise ContractError("EFFECT_FORBIDDEN", "SQL update/replace authority was not granted.")
        if self.create and not policy.allow_ddl:
            raise ContractError("EFFECT_FORBIDDEN", "DDL needs separate operator authorization.")
        selected = budget or PreviewBudget(
            max_rows=10_000, max_bytes=8 * 1024**2, max_seconds=self.statement_timeout
        )
        selected.open_file()
        names = [c.name for c in self.schema.columns]
        quoted = [identifier(c) for c in names]
        mark = "?" if self.dialect == "sqlite" else "%s"
        statement = (
            "INSERT INTO "
            + self.table
            + " ("
            + ",".join(quoted)
            + ") VALUES ("
            + ",".join([mark] * len(names))
            + ")"
        )
        if self.mode == "upsert":
            updates = [
                q + "=excluded." + q
                for name, q in zip(names, quoted, strict=True)
                if name not in self.schema.keys
            ]
            statement += " ON CONFLICT (" + ",".join(identifier(k) for k in self.schema.keys) + ") "
            statement += "DO UPDATE SET " + ",".join(updates) if updates else "DO NOTHING"
        count = 0
        seen = set()
        with connection(self, readonly=False, budget=selected) as conn:
            if self.create:
                types = {
                    "integer": "BIGINT",
                    "number": "DOUBLE PRECISION",
                    "boolean": "BOOLEAN",
                    "string": "TEXT",
                    "datetime": "TEXT",
                }
                definitions = [
                    identifier(c.name) + " " + types[c.type] + ("" if c.nullable else " NOT NULL")
                    for c in self.schema.columns
                ]
                definitions.append(
                    "PRIMARY KEY (" + ",".join(identifier(k) for k in self.schema.keys) + ")"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS " + self.table + " (" + ",".join(definitions) + ")"
                )
            if self.mode == "replace":
                conn.execute("DELETE FROM " + self.table)
            for raw in rows:
                selected.check()
                if set(raw) != set(names):
                    raise ContractError(
                        "INVALID_SCHEMA", "SQL write columns differ from declared schema."
                    )
                row = self.schema.convert(raw, retain_extra=False)
                selected.row(row)
                selected.charge(selected.size(row))
                key = tuple(row[k] for k in self.schema.keys)
                if key in seen:
                    raise ContractError("DUPLICATE_ID", "Input contains duplicate SQL keys.")
                seen.add(key)
                conn.execute(statement, tuple(row[name] for name in names))
                count += 1
        return {
            "schema_version": "sql-write-receipt-v1",
            "rows": count,
            "mode": self.mode,
            "transaction": "committed",
            "backend": self.dialect,
            "table": self.table,
        }
