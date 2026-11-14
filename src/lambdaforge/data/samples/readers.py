"""Bounded selected-field readers; no pickle, inferred formats or arbitrary object loading."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import time
import zipfile
from pathlib import Path
from typing import Any

from lambdaforge.catalog.errors import ContractError


class ReadBudget:
    """Cooperative deadline plus encoded/decoded bytes; cannot preempt native filesystem calls."""

    def __init__(self, max_bytes: int = 32 * 1024**2, max_seconds: float = 5) -> None:
        if (
            type(max_bytes) is not int
            or max_bytes <= 0
            or isinstance(max_seconds, bool)
            or not math.isfinite(max_seconds)
            or max_seconds <= 0
        ):
            raise ContractError("INVALID_BUDGET", "Positive byte/time limits required.")
        self.remaining = max_bytes
        self.deadline = time.monotonic() + max_seconds

    def charge(self, count: int = 0) -> None:
        self.remaining -= count
        if self.remaining < 0 or time.monotonic() > self.deadline:
            raise ContractError("READ_LIMIT", "Sample read exceeded its byte/time budget.")


def _array_header(handle: Any, budget: ReadBudget) -> None:
    import numpy as np

    version = np.lib.format.read_magic(handle)
    if version not in {(1, 0), (2, 0)}:
        raise ContractError("UNSUPPORTED_FORMAT", "Only NPY header versions 1/2 are supported.")
    reader = (
        np.lib.format.read_array_header_1_0
        if version == (1, 0)
        else np.lib.format.read_array_header_2_0
    )
    shape, _fortran, dtype = reader(handle)
    if dtype.hasobject:
        raise ContractError("UNSAFE_DTYPE", "Object arrays/pickle are not supported.")
    import math

    budget.charge(math.prod(shape) * dtype.itemsize)


def read_field(
    path: Path,
    format: str,
    selection: tuple[str | int, ...],
    budget: ReadBudget,
    *,
    sha256: str | None = None,
) -> Any:
    """Open only a selected local asset on demand after native placement resolution."""
    from lambdaforge.work.snapshot import validate_path

    validate_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    budget.charge(path.stat().st_size)
    if sha256:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                budget.charge()
                digest.update(chunk)
        if digest.hexdigest() != sha256.removeprefix("sha256:"):
            raise ContractError("CONTENT_MISMATCH", "Selected Dataset asset checksum differs.")
    if format in {"npy", "npz"}:
        import numpy as np

        if format == "npy":
            with path.open("rb") as file:
                _array_header(file, budget)
            value = np.load(path, allow_pickle=False)
        else:
            if len(selection) != 1 or not isinstance(selection[0], str):
                raise ContractError("INVALID_SCHEMA", "NPZ requires one explicit array key.")
            with zipfile.ZipFile(path) as archive:
                member = selection[0] + ".npy"
                info = archive.getinfo(member)
                if info.file_size > budget.remaining:
                    raise ContractError("READ_LIMIT", "Decoded NPZ member exceeds byte budget.")
                with archive.open(info) as file:
                    _array_header(file, budget)
            with np.load(path, allow_pickle=False) as arrays:
                value = arrays[selection[0]]
            selection = ()
    elif format == "json":
        value = json.loads(path.read_text(encoding="utf-8"))
    elif format == "csv":
        value = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
    else:
        raise ContractError("UNSUPPORTED_FORMAT", f"Unsupported reader {format!r}.")
    for component in selection:
        value = value[component]
    budget.charge()
    return value
