"""Explicit format writers for native managed outputs/checkpoint staging."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from lambdaforge.catalog.errors import ContractError


class WriteRecords:
    """Write new owned staging files; Dataset publication remains outputs.dataset.

    No overwrite, inferred formats or pickle. The caller declares the owned destination
    through Work outputs/checkpoints; this operation does not invent storage locations.
    """

    def __init__(
        self,
        format: Literal["npy", "npz", "json", "csv", "tsv"],
        csv: Mapping[str, Any] | None = None,
    ) -> None:
        if format not in {"json", "csv", "tsv", "npy", "npz"}:
            raise ContractError("UNSUPPORTED_FORMAT", "Expected json/csv/npy/npz.")
        self.format = format
        from lambdaforge.etl.CsvOptions import CsvOptions

        self.csv = CsvOptions(**{"delimiter": "\t" if format == "tsv" else ",", **dict(csv or {})})
        if csv and format not in {"csv", "tsv"}:
            raise ContractError("INVALID_PARAMETERS", "CSV options need CSV/TSV format.")
        if self.csv.bom != "preserve":
            raise ContractError(
                "INVALID_PARAMETERS", "BOM read policies are reader-only; use write_bom for output."
            )
        if (
            self.csv.header_rows > 1
            or self.csv.skip_rows
            or self.csv.comment
            or self.csv.parse_errors != "error"
        ):
            raise ContractError(
                "INVALID_PARAMETERS", "Read-only CSV options are invalid for writers."
            )

    def __call__(self, file: str | Path, value: Any) -> Path:
        import csv
        import json

        import numpy as np

        from lambdaforge.work.snapshot import validate_path

        target = Path(file)
        validate_path(target)
        if self.format in {"npy", "npz"}:
            arrays = (
                {k: np.asarray(v) for k, v in value.items()}
                if self.format == "npz"
                else {"array": np.asarray(value)}
            )
            if any(a.dtype.hasobject for a in arrays.values()):
                raise ContractError("UNSAFE_DTYPE", "Writers refuse object/pickle arrays.")
            with target.open("xb") as handle:
                if self.format == "npz":
                    np.savez_compressed(handle, **arrays)
                else:
                    np.save(handle, arrays["array"], allow_pickle=False)
        elif self.format == "json":
            encoded = json.dumps(value, allow_nan=False, ensure_ascii=False)
            with target.open("x", encoding="utf-8") as json_file:
                json_file.write(encoded)
        else:
            with target.open("x", encoding=self.csv.encoding, newline="") as csv_file:
                rows = iter(value)
                first = next(rows, None)
                if first is not None:
                    if self.csv.write_bom:
                        csv_file.write("\ufeff")
                    writer = csv.DictWriter(
                        csv_file, fieldnames=list(self.csv.columns or first), **self.csv.dialect()
                    )
                    if self.csv.header_rows:
                        writer.writeheader()

                    def converted(row):
                        return {
                            k: str(v).replace(".", self.csv.decimal) if isinstance(v, float) else v
                            for k, v in row.items()
                        }

                    writer.writerow(converted(first))
                    writer.writerows(converted(row) for row in rows)
        return target
