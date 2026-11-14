"""Explicit prefix inspection of large numeric assets without full materialization."""

from __future__ import annotations

import math
import zipfile
from pathlib import Path
from typing import Any, Literal

from lambdaforge.catalog.errors import ContractError
from lambdaforge.etl.PreviewBudget import PreviewBudget


class PreviewArray:
    """NPY uses a read-only mmap; NPZ decodes only the selected C-order prefix.

    Neither path verifies the checksum of unread bytes or claims a scientific content id.
    Selected NPZ member full decoded extent is reported, not allocated. Fortran-order NPZ
    needs a full decode for first-axis rows and is refused rather than pretending streaming.
    """

    def __init__(self, format: Literal["npy", "npz"], key: str | None = None) -> None:
        if format not in {"npy", "npz"} or (format == "npz" and not key):
            raise ContractError("INVALID_PARAMETERS", "Choose NPY or an explicit NPZ array key.")
        self.format, self.key = format, key

    def __call__(self, file: str | Path, *, budget: PreviewBudget | None = None) -> dict[str, Any]:
        import numpy as np

        from lambdaforge.work.snapshot import validate_path

        selected = budget or PreviewBudget()
        path = Path(file)
        validate_path(path)
        selected.open_file()
        if selected.full:
            raise ContractError("READ_LIMIT", "No array rows remain in the shared budget.")

        def header(handle):
            version = np.lib.format.read_magic(handle)
            if version not in {(1, 0), (2, 0)}:
                raise ContractError("UNSUPPORTED_FORMAT", "Only NPY headers 1/2 supported.")
            shape, fortran, dtype = (
                np.lib.format.read_array_header_1_0(handle)
                if version == (1, 0)
                else np.lib.format.read_array_header_2_0(handle)
            )
            selected.charge(handle.tell())
            if dtype.hasobject:
                raise ContractError("UNSAFE_DTYPE", "Object arrays are forbidden.")
            if dtype.kind not in "biufcSU":
                raise ContractError(
                    "UNSUPPORTED_DTYPE",
                    "Preview supports numeric/string arrays, not structured records.",
                )
            return shape, fortran, dtype

        def rows_count(shape, dtype):
            remaining = selected.max_rows - selected.rows
            row_bytes = math.prod(shape[1:]) * dtype.itemsize if shape else dtype.itemsize
            n = min(shape[0] if shape else 1, remaining)
            decoded_guard = n * max(1, math.prod(shape[1:])) * max(64, dtype.itemsize * 8)
            if decoded_guard > selected.max_memory:
                raise ContractError("READ_LIMIT", "Decoded array prefix exceeds allocation guard.")
            selected.charge(n * row_bytes)
            selected.retain(decoded_guard)
            return n

        if self.format == "npy":
            with path.open("rb") as handle:
                shape, fortran, dtype = header(handle)
            n = rows_count(shape, dtype)
            mapped = np.load(path, mmap_mode="r", allow_pickle=False)
            try:
                prefix = np.asarray(mapped[:n] if shape else mapped).copy()
            finally:
                mapped._mmap.close()
        else:
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                name = str(self.key) + ".npy"
                if names.count(name) != 1:
                    raise ContractError("MISSING_FIELD", "Missing/ambiguous NPZ member.")
                with archive.open(name) as member:
                    shape, fortran, dtype = header(member)
                    if fortran and len(shape) > 1:
                        raise ContractError(
                            "UNSUPPORTED_PREVIEW", "Fortran NPZ is not first-axis incremental."
                        )
                    n = rows_count(shape, dtype)
                    count = n * math.prod(shape[1:]) if shape else 1
                    raw = member.read(count * dtype.itemsize)
                    if len(raw) != count * dtype.itemsize:
                        raise ContractError("CONTENT_MISMATCH", "Truncated numeric array payload.")
                    prefix = np.frombuffer(raw, dtype=dtype).reshape(
                        (n, *shape[1:]) if shape else ()
                    )
        selected.check()
        selected.rows += n
        values = prefix.tolist()

        # Convert potentially nonfinite scientific values to an explicit diagnostic representation.
        def portable(value):
            if isinstance(value, list):
                return [portable(v) for v in value]
            if isinstance(value, float) and not math.isfinite(value):
                return {"nonfinite": str(value)}
            if isinstance(value, complex):
                return {"real": portable(value.real), "imag": portable(value.imag)}
            if isinstance(value, bytes):
                return {"bytes_hex": value.hex()}
            return value

        return {
            "schema_version": "array-preview-v1",
            "shape": list(shape),
            "dtype": str(dtype),
            "values": portable(values),
            "scope": "prefix",
            "content_identity": None,
            "complete": n == (shape[0] if shape else 1),
            "decoded_extent_bytes": math.prod(shape) * dtype.itemsize,
            "provenance": {"source": str(path), "array": self.key, "first_axis": [0, n]},
            "budget": selected.to_dict(),
        }
