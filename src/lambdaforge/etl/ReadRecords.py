"""Explicit bounded records or selected arrays from common formats."""

from collections.abc import Generator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from lambdaforge.data.samples.readers import ReadBudget, read_field


class ReadRecords:
    """A selected file/field reader, not an implicit multi-file training dataset."""

    def __init__(
        self,
        format: Literal["npy", "npz", "json", "csv", "tsv"],
        path: tuple[str | int, ...] = (),
        max_bytes: int = 32 * 1024**2,
        max_seconds: float = 5,
        csv: Mapping[str, Any] | None = None,
        columns: Sequence[str] | None = None,
        max_rows: int | None = None,
        schema: Mapping[str, Any] | None = None,
        compression: Literal["gzip"] | None = None,
    ) -> None:
        self.format, self.path = format, tuple(path)
        self.max_bytes, self.max_seconds = max_bytes, max_seconds
        from lambdaforge.catalog.errors import ContractError
        from lambdaforge.etl.CsvOptions import CsvOptions
        from lambdaforge.etl.tables.TableSchema import TableSchema

        self.csv = CsvOptions(
            **({"delimiter": "\t" if format == "tsv" else ",", **dict(csv or {})})
        )
        self.columns = tuple(columns) if columns is not None else None
        self.max_rows = max_rows
        self.schema = TableSchema.from_mapping(schema) if schema is not None else None
        self.compression = compression
        if self.csv.write_bom:
            raise ContractError("INVALID_PARAMETERS", "write_bom is a writer-only setting.")
        if self.csv.decimal != "." and self.schema is None:
            raise ContractError(
                "INVALID_PARAMETERS", "Decimal conversion requires an explicit table schema."
            )
        if max_rows is not None and (type(max_rows) is not int or max_rows < 1):
            raise ContractError("INVALID_BUDGET", "max_rows must be a positive integer.")
        if format not in {"csv", "tsv"} and (
            csv or columns is not None or schema is not None or compression or max_rows is not None
        ):
            raise ContractError("INVALID_PARAMETERS", "Delimited options apply only to CSV/TSV.")

    def __call__(self, file: str | Path) -> Any:
        if self.format in {"csv", "tsv"}:
            from lambdaforge.etl.csv_io import csv_rows
            from lambdaforge.etl.PreviewBudget import PreviewBudget

            guard = PreviewBudget(max_memory=self.max_bytes, max_seconds=self.max_seconds)
            result, size = [], 0
            selected = PreviewBudget(
                max_rows=self.max_rows or 2**63 - 1,
                max_bytes=self.max_bytes,
                max_seconds=self.max_seconds,
                max_memory=self.max_bytes,
            )
            for row in csv_rows(
                Path(file),
                self.csv,
                selected,
                projection=self.columns,
                schema=self.schema,
                compression=self.compression,
                row_limit=self.max_rows,
            ):
                size += guard.size(row)
                if size > self.max_bytes:
                    from lambdaforge.catalog.errors import ContractError

                    raise ContractError(
                        "READ_LIMIT", "Materialized CSV exceeds memory guard; iterate instead."
                    )
                result.append(row)
            value: Any = result
            for key in self.path:
                value = value[key]
            return value
        return read_field(
            Path(file), self.format, self.path, ReadBudget(self.max_bytes, self.max_seconds)
        )

    def iter_records(
        self, file: str | Path, *, budget: Any = None
    ) -> Generator[dict[str, Any], None, None]:
        """Incremental CSV/TSV only: callers must close abandoned iterators to release files."""
        from lambdaforge.catalog.errors import ContractError
        from lambdaforge.etl.csv_io import csv_rows
        from lambdaforge.etl.PreviewBudget import PreviewBudget

        if self.format not in {"csv", "tsv"} or self.path:
            raise ContractError(
                "UNSUPPORTED_PREVIEW",
                "Incremental table reading needs CSV/TSV without internal path.",
            )
        selected = budget or PreviewBudget(
            max_rows=self.max_rows or 2**63 - 1,
            max_bytes=self.max_bytes,
            max_seconds=self.max_seconds,
            max_memory=self.max_bytes,
        )
        yield from csv_rows(
            Path(file),
            self.csv,
            selected,
            projection=self.columns,
            schema=self.schema,
            compression=self.compression,
            row_limit=self.max_rows,
        )

    def preview(self, file: str | Path, *, budget: Any = None) -> dict[str, Any]:
        """Explicit prefix, not random sample/full-file hash or proof about unread rows."""
        from lambdaforge.catalog.errors import ContractError
        from lambdaforge.etl.csv_io import csv_rows
        from lambdaforge.etl.PreviewBudget import PreviewBudget

        selected = budget or PreviewBudget(max_bytes=self.max_bytes, max_seconds=self.max_seconds)
        if self.format not in {"csv", "tsv"} or self.path:
            raise ContractError(
                "UNSUPPORTED_PREVIEW",
                "Use array preview for NPY/NPZ; whole JSON is not incremental.",
            )
        rows: list[dict[str, Any]] = []
        provenance: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        memory = 0
        for row in csv_rows(
            Path(file),
            self.csv,
            selected,
            projection=self.columns,
            schema=self.schema,
            compression=self.compression,
            provenance=provenance,
            diagnostics=diagnostics,
            row_limit=self.max_rows,
        ):
            memory += selected.size(row)
            if memory > selected.max_memory:
                raise ContractError("READ_LIMIT", "Preview retained rows exceed memory guard.")
            rows.append(row)
            selected.retain(selected.size(row))
        return {
            "schema_version": "table-preview-v1",
            "rows": rows,
            "provenance": provenance,
            "scope": "prefix",
            "complete": not selected.full
            and not (self.max_rows is not None and len(rows) >= self.max_rows),
            "content_identity": None,
            "schema": self.schema.to_dict() if self.schema else None,
            "budget": selected.to_dict(),
            "diagnostics": diagnostics,
        }
