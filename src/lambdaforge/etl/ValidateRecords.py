"""Stream identified decoded records through the native sample field validators."""

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from lambdaforge.catalog.errors import ContractError
from lambdaforge.data.samples.SampleSchema import SampleSchema


class ValidateRecords:
    """Check named fields, not asset reads or scientific equivalence.

    Values have already been decoded by the recipe. Reuse SampleSchema's field contracts
    for dtype/shape checks, preserve the original values/id/partitions, and reject missing
    required columns. Optional columns remain absent; this operation does not impute them.
    """

    def __init__(self, schema: Mapping[str, Any]) -> None:
        self.schema = SampleSchema.from_mapping(schema)

    def __call__(self, records: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
        for row in records:
            if not isinstance(row.get("id"), str) or not row["id"]:
                raise ContractError("MISSING_ID", "Records require explicit nonempty string ids.")
            for field in self.schema.fields:
                if field.name not in row:
                    if field.required:
                        raise ContractError(
                            "MISSING_FIELD", f"Required column {field.name!r} absent."
                        )
                    continue
                field.validate_value(row[field.name])
            yield dict(row)
