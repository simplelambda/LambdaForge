"""Explicit selection, renaming and filtering of identified records."""

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from lambdaforge.catalog.errors import ContractError


class SelectFields:
    """Preserve id and partitions; predicates are explicit column equalities, never eval."""

    def __init__(self, fields: Mapping[str, str], where: Mapping[str, Any] | None = None) -> None:
        self.fields, self.where = dict(fields), dict(where or {})
        if (
            self.fields.get("id", "id") != "id"
            or self.fields.get("partitions", "partitions") != "partitions"
        ):
            raise ContractError(
                "IDENTITY_CHANGE", "Selection cannot rename identity or partitions."
            )

    def __call__(self, records: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
        for row in records:
            if "id" not in row:
                raise ContractError("MISSING_ID", "Records require explicit stable ids.")
            if all(row.get(k) == v for k, v in self.where.items()):
                yield {
                    **{new: row[old] for new, old in self.fields.items()},
                    "id": row["id"],
                    "partitions": dict(row.get("partitions", {})),
                }
