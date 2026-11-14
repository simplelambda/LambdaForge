"""Read-only transactional SQL rows with actual cursor batching and explicit source evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Generator, Mapping, Sequence
from typing import Any, Literal
from uuid import uuid4

from lambdaforge.catalog.errors import ContractError
from lambdaforge.etl.PreviewBudget import PreviewBudget
from lambdaforge.etl.sql_backend import connection, identifier, validate_connection
from lambdaforge.etl.tables.TableSchema import TableSchema


class ReadSQL:
    """Read SQLite or optional PostgreSQL using parameters, never SQL string substitution.

    One transaction per iterator; close abandoned generators. PostgreSQL uses a server cursor,
    SQLite fetchmany. SQL text/parameters/snapshot are provenance, not immutable scientific
    content. Publish exact materialized bytes with native DatasetPublisher to pin inputs.
    """

    def __init__(
        self,
        dialect: Literal["sqlite", "postgresql"],
        file: str | None = None,
        credential: str | None = None,
        query: str | None = None,
        table: str | None = None,
        columns: Sequence[str] | None = None,
        order_by: Sequence[str] = (),
        parameters: Mapping[str, Any] | Sequence[Any] = (),
        chunk_size: int = 100,
        timeout: float = 10,
        statement_timeout: float = 10,
        sslmode: Literal["verify-full", "verify-ca", "require"] = "verify-full",
        schema: Mapping[str, Any] | None = None,
    ) -> None:
        validate_connection(dialect, file, credential, timeout, statement_timeout, sslmode)
        if (query is None) == (table is None) or type(chunk_size) is not int or chunk_size < 1:
            raise ContractError(
                "INVALID_PARAMETERS", "Choose query OR table and a positive chunk size."
            )
        if query is not None and (columns is not None or order_by):
            raise ContractError(
                "INVALID_PARAMETERS",
                "Query owns projection/order; do not supply ignored table options.",
            )
        if query is not None and not query.lstrip().lower().startswith(
            ("select ", "select\n", "with ", "with\n")
        ):
            raise ContractError(
                "SQL_READ_ONLY", "Readers accept SELECT/WITH only, not DDL or mutations."
            )
        self.dialect, self.file, self.credential = dialect, file, credential
        self.query = query or (
            "SELECT "
            + (", ".join(identifier(c) for c in columns) if columns else "*")
            + " FROM "
            + identifier(str(table))
            + (" ORDER BY " + ", ".join(identifier(c) for c in order_by) if order_by else "")
        )
        self.parameters = dict(parameters) if isinstance(parameters, Mapping) else tuple(parameters)
        json.dumps(self.parameters, allow_nan=False)
        self.chunk_size, self.timeout, self.statement_timeout, self.sslmode = (
            chunk_size,
            timeout,
            statement_timeout,
            sslmode,
        )
        self.schema = TableSchema.from_mapping(schema) if schema else None

    def iter_records(
        self, *, budget: PreviewBudget | None = None
    ) -> Generator[dict[str, Any], None, None]:
        """Budget caps the entire iterator, including fetched chunks; row limit is a prefix."""
        selected = budget or PreviewBudget(
            max_rows=10_000, max_bytes=8 * 1024**2, max_seconds=self.statement_timeout
        )
        selected.open_file()  # SQL connection is one source in the global source/file budget.
        with connection(self, readonly=True, budget=selected) as conn:
            cursor = (
                conn.cursor(name="lf_" + uuid4().hex)
                if self.dialect == "postgresql"
                else conn.cursor()
            )
            try:
                selected.check()
                cursor.execute(self.query, self.parameters)
                names = [v[0] for v in cursor.description]
                if len(set(names)) != len(names):
                    raise ContractError(
                        "COLUMN_COLLISION", "SQL output columns must have unique aliases."
                    )
                while not selected.full:
                    selected.check()
                    batch = cursor.fetchmany(
                        min(self.chunk_size, selected.max_rows - selected.rows)
                    )
                    if not batch:
                        return
                    memory = 0
                    for values in batch:
                        row = dict(zip(names, values, strict=True))
                        if self.schema:
                            row = self.schema.convert(row)
                        memory += selected.size(row)
                        if memory > selected.max_memory:
                            raise ContractError(
                                "READ_LIMIT", "Fetched SQL chunk exceeds allocation guard."
                            )
                        selected.charge(len(json.dumps(row, allow_nan=False).encode()))
                        selected.row(row)
                        yield row
            finally:
                cursor.close()

    def preview(self, *, budget: PreviewBudget | None = None) -> dict[str, Any]:
        selected = budget or PreviewBudget(max_seconds=self.statement_timeout)
        iterator = self.iter_records(budget=selected)
        rows, retained = [], 0
        try:
            for row in iterator:
                retained += selected.size(row)
                if retained > selected.max_memory:
                    raise ContractError("READ_LIMIT", "SQL preview exceeds retained-row guard.")
                rows.append(row)
                selected.retain(selected.size(row))
        finally:
            iterator.close()
        observed = hashlib.sha256(
            json.dumps(rows, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        query_id = hashlib.sha256(
            json.dumps([self.query, self.parameters], sort_keys=True).encode()
        ).hexdigest()
        return {
            "schema_version": "table-preview-v1",
            "rows": rows,
            "scope": "prefix",
            "complete": not selected.full,
            "content_identity": None,
            "observed_rows_sha256": observed,
            "provenance": {
                "backend": self.dialect,
                "query_sha256": query_id,
                "isolation": "sqlite-read-transaction"
                if self.dialect == "sqlite"
                else "repeatable-read",
                "ordering": "authored-query",
                "mutable_source": True,
                "row_ordinals": list(range(len(rows))),
            },
            "budget": selected.to_dict(),
        }

    def __call__(self, *, budget: PreviewBudget | None = None) -> list[dict[str, Any]]:
        return self.preview(budget=budget)["rows"]
