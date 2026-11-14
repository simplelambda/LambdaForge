"""Shared explicit delimited-text conventions for native readers and writers."""

from dataclasses import dataclass
from typing import Literal

from lambdaforge.catalog.errors import ContractError


@dataclass(frozen=True)
class CsvOptions:
    """Multiple header rows: last row owns column names; prior rows are skipped.

    header_rows=0 requires explicit columns. skip_rows counts physical lines before header.
    Comments apply only at record boundaries, never inside quoted multiline values.
    Decimal conversion is used only with an explicit TableSchema, never guessed.
    """

    delimiter: str = ","
    header_rows: int = 1
    columns: tuple[str, ...] = ()
    encoding: str = "utf-8"
    bom: Literal["preserve", "strip", "error"] = "preserve"
    quotechar: str | None = '"'
    escapechar: str | None = None
    doublequote: bool = True
    comment: str | None = None
    skip_rows: int = 0
    decimal: str = "."
    parse_errors: Literal["error", "skip"] = "error"
    write_bom: bool = False

    def __post_init__(self) -> None:
        import codecs

        codecs.lookup(self.encoding)
        if type(self.doublequote) is not bool or type(self.write_bom) is not bool:
            raise ContractError("INVALID_PARAMETERS", "CSV boolean flags must be booleans.")
        if isinstance(self.columns, str) or any(not isinstance(v, str) for v in self.columns):
            raise ContractError("INVALID_PARAMETERS", "Column names must be a sequence of strings.")
        if len(self.delimiter) != 1 or self.delimiter in "\r\n":
            raise ContractError(
                "INVALID_PARAMETERS", "Delimiter must be one non-newline character."
            )
        if any(v is not None and len(v) != 1 for v in (self.quotechar, self.escapechar)):
            raise ContractError("INVALID_PARAMETERS", "Quote/escape must be null or one character.")
        if any(type(v) is not int or v < 0 for v in (self.header_rows, self.skip_rows)):
            raise ContractError(
                "INVALID_PARAMETERS", "Header/skip counts must be nonnegative integers."
            )
        if self.bom not in {"preserve", "strip", "error"} or self.parse_errors not in {
            "error",
            "skip",
        }:
            raise ContractError("INVALID_PARAMETERS", "Invalid BOM/parse-error policy.")
        if self.decimal not in {".", ","} or self.comment == "":
            raise ContractError(
                "INVALID_PARAMETERS", "Decimal must be dot/comma; comments nonempty."
            )
        if self.header_rows == 0 and not self.columns:
            raise ContractError(
                "INVALID_PARAMETERS", "Headerless CSV requires explicit column names."
            )
        if len(set(self.columns)) != len(self.columns) or any(not c for c in self.columns):
            raise ContractError(
                "INVALID_PARAMETERS", "Explicit column names must be unique/nonempty."
            )

    def dialect(self) -> dict:
        import csv

        return {
            "delimiter": self.delimiter,
            "quotechar": self.quotechar,
            "escapechar": self.escapechar,
            "doublequote": self.doublequote,
            "quoting": csv.QUOTE_NONE if self.quotechar is None else csv.QUOTE_MINIMAL,
        }
