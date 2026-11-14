"""Portable scalar column conversions with explicit invalid/null/date policies."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo

from lambdaforge.catalog.errors import ContractError


@dataclass(frozen=True)
class Column:
    """No inferred conversions. Datetimes are emitted as ISO strings with explicit zones.

    Blank strings are not null unless declared in null_values. Booleans accept true/false
    and 1/0 only; integer conversion never truncates fractions. Arrays remain structured
    evidence and are refused by scalar column conversion, not flattened.
    """

    name: str
    type: Literal["string", "integer", "number", "boolean", "datetime", "json"] = "string"
    nullable: bool = False
    invalid: Literal["error", "null"] = "error"
    null_values: tuple[str, ...] = ()
    categories: tuple[Any, ...] = ()
    date_format: str | None = None
    timezone: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or self.type
            not in {
                "string",
                "integer",
                "number",
                "boolean",
                "datetime",
                "json",
            }
        ):
            raise ContractError("INVALID_SCHEMA", "Invalid column name/type.")
        if type(self.nullable) is not bool or self.invalid not in {"error", "null"}:
            raise ContractError("INVALID_SCHEMA", "Invalid nullable/conversion policy.")
        if self.invalid == "null" and not self.nullable:
            raise ContractError("INVALID_SCHEMA", "Invalid-to-null requires nullable column.")
        if self.timezone is not None:
            ZoneInfo(self.timezone)

    def convert(self, value: Any, *, decimal: str = ".") -> Any:
        if value is None or isinstance(value, str) and value in self.null_values:
            if not self.nullable:
                raise ContractError("NULL_VALUE", f"Required column {self.name!r} is null.")
            return None
        try:
            if self.type != "json" and isinstance(value, (dict, list, tuple)):
                raise ValueError("Structured value is not a scalar")
            if self.type == "string":
                if not isinstance(value, (str, int, float, bool)):
                    raise ValueError("Unsupported scalar")
                result: Any = str(value)
            elif self.type == "integer":
                if isinstance(value, bool) or isinstance(value, float) and not value.is_integer():
                    raise ValueError("Integer cannot truncate")
                result = int(value)
            elif self.type == "number":
                if isinstance(value, bool):
                    raise ValueError("Boolean is not numeric")
                result = float(value.replace(decimal, ".") if isinstance(value, str) else value)
                if not math.isfinite(result):
                    raise ValueError("Nonfinite number")
            elif self.type == "boolean":
                if value not in (True, False, 0, 1, "true", "false", "1", "0"):
                    raise ValueError("Invalid boolean")
                result = value in (True, 1, "true", "1")
            elif self.type == "datetime":
                if isinstance(value, datetime):
                    date = value
                elif self.date_format:
                    try:
                        date = datetime.strptime(value, self.date_format)
                    except ValueError:
                        # Native conversions emit a canonical aware ISO value. Revalidation
                        # must accept that value without weakening the authored raw format.
                        date = datetime.fromisoformat(value)
                        if date.tzinfo is None or date.isoformat() != value:
                            raise ValueError(
                                "Not the declared format or canonical aware ISO"
                            ) from None
                else:
                    date = datetime.fromisoformat(value)
                if date.tzinfo is None:
                    if self.timezone is None:
                        raise ValueError("Naive datetime needs an explicit timezone")
                    zone = ZoneInfo(self.timezone)
                    first, second = (
                        date.replace(tzinfo=zone, fold=0),
                        date.replace(tzinfo=zone, fold=1),
                    )
                    if first.utcoffset() != second.utcoffset():
                        raise ValueError("Ambiguous or nonexistent local time")
                    date = first
                elif self.timezone:
                    date = date.astimezone(ZoneInfo(self.timezone))
                result = date.isoformat()
            else:
                import json

                result = json.loads(value) if isinstance(value, str) else value
                json.dumps(result, allow_nan=False)
            if self.categories and result not in self.categories:
                raise ValueError("Value outside declared categories")
            return result
        except (ValueError, TypeError, OverflowError):
            if self.invalid == "null":
                return None
            # No raw value: a column may contain a credential or personal data.
            raise ContractError(
                "CONVERSION_ERROR",
                f"Cannot convert column {self.name!r}.",
                column=self.name,
                expected=self.type,
            ) from None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["null_values"], result["categories"] = list(self.null_values), list(self.categories)
        return result
