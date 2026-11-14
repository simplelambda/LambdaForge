"""Incremental delimited rows with shared budgets, explicit projection and provenance."""

from __future__ import annotations

import codecs
import csv
import gzip
import itertools
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from lambdaforge.catalog.errors import ContractError
from lambdaforge.etl.CsvOptions import CsvOptions
from lambdaforge.etl.PreviewBudget import PreviewBudget
from lambdaforge.etl.tables.TableSchema import TableSchema


def csv_rows(
    file: Path,
    options: CsvOptions,
    budget: PreviewBudget,
    *,
    projection: Sequence[str] | None = None,
    schema: TableSchema | None = None,
    compression: str | None = None,
    provenance: list[dict[str, Any]] | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
    row_limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    from lambdaforge.work.snapshot import validate_path

    validate_path(file)
    budget.open_file()
    if compression not in {None, "gzip"}:
        raise ContractError("UNSUPPORTED_FORMAT", "CSV compression must be explicit gzip or null.")
    physical = 0
    record_start = 0
    decoder = codecs.getincrementaldecoder(options.encoding)(errors="strict")

    def lines(handle: Any) -> Iterator[str]:
        nonlocal physical
        while True:
            budget.check()
            limit = min(budget.max_bytes - budget.bytes, budget.max_memory // 4)
            raw = handle.readline(max(1, limit + 1))
            if not raw:
                rest = decoder.decode(b"", final=True)
                if rest:
                    yield rest
                return
            budget.charge(len(raw))  # gzip handle charges decompressed input, not compressed size.
            if (budget.bytes - record_start) * 4 > budget.max_memory:
                raise ContractError("READ_LIMIT", "CSV record exceeds decoded allocation guard.")
            if len(raw) > budget.max_memory // 4:
                raise ContractError("READ_LIMIT", "Decoded CSV line exceeds memory guard.")
            try:
                text = decoder.decode(raw)
            except UnicodeError:
                raise ContractError(
                    "PARSE_ERROR", "CSV encoding differs from declared encoding."
                ) from None
            physical += 1
            if physical == 1 and text.startswith("\ufeff"):
                if options.bom == "error":
                    raise ContractError("PARSE_ERROR", "Unexpected CSV BOM.")
                if options.bom == "strip":
                    text = text[1:]
            yield text

    open_file = gzip.open if compression == "gzip" else open
    with open_file(file, "rb") as handle:
        source = iter(lines(handle))
        for _ in range(options.skip_rows):
            next(source, None)

        def records() -> Iterator[list[str]]:
            nonlocal record_start
            while True:
                record_start = budget.bytes
                first = next(source, None)
                if first is None:
                    return
                if not first.strip() or options.comment and first.startswith(options.comment):
                    continue
                try:
                    yield next(
                        csv.reader(
                            itertools.chain((first,), source), strict=True, **options.dialect()
                        )
                    )
                except csv.Error:
                    if options.parse_errors == "error":
                        raise ContractError(
                            "PARSE_ERROR", "Malformed CSV record.", line=physical
                        ) from None
                    if diagnostics is not None and len(diagnostics) < 100:
                        diagnostics.append({"code": "SKIPPED_RECORD", "line": physical})

        values = records()
        header: list[str] = list(options.columns)
        for _ in range(options.header_rows):
            current = next(values, None)
            if current is None:
                return
            if not options.columns:
                header = current
        if len(set(header)) != len(header) or any(not v for v in header):
            raise ContractError("INVALID_SCHEMA", "CSV names must be unique/nonempty.")
        if projection is not None and (
            len(set(projection)) != len(projection) or set(projection) - set(header)
        ):
            raise ContractError(
                "INVALID_PROJECTION", "Projection references duplicate/unknown columns."
            )
        emitted = 0
        while not budget.full and (row_limit is None or emitted < row_limit):
            budget.check()
            value = next(values, None)
            if value is None:
                return
            if len(value) != len(header):
                if options.parse_errors == "skip":
                    if diagnostics is not None and len(diagnostics) < 100:
                        diagnostics.append({"code": "SKIPPED_RECORD", "line": physical})
                    continue
                raise ContractError(
                    "PARSE_ERROR", "CSV record width differs from header.", line=physical
                )
            row = dict(zip(header, value, strict=True))
            if schema is not None:
                row = schema.convert(row, decimal=options.decimal)
            if projection is not None:
                row = {name: row[name] for name in projection}
            budget.row(row)
            emitted += 1
            if provenance is not None:
                provenance.append({"source": str(file), "physical_end_line": physical})
            yield row
