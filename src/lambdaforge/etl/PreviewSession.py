"""Scoped preview utilities for generated ordinary Python; no plugin or node executor."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from lambdaforge.catalog.errors import ContractError
from lambdaforge.etl.PreviewArray import PreviewArray
from lambdaforge.etl.PreviewBudget import PreviewBudget
from lambdaforge.etl.ReadRecords import ReadRecords
from lambdaforge.etl.ReadSQL import ReadSQL


class PreviewSession:
    """Operator pins allowed local roots, budget and explicit network authorization.

    Calls only known native read APIs, never arbitrary discovered plugin code. These guards
    do not sandbox unrestricted Python or a database function's external effects. Staging is
    disposable and never published; caller-owned input paths are never cleaned up.
    """

    def __init__(
        self,
        roots: Sequence[str | Path],
        budget: PreviewBudget | None = None,
        allow_network: bool = False,
    ) -> None:
        from lambdaforge.work.snapshot import validate_path

        self.roots = tuple(Path(p).absolute() for p in roots)
        if not self.roots:
            raise ContractError("INVALID_PARAMETERS", "Pin at least one preview input root.")
        for path in self.roots:
            if ".." in path.parts:
                raise ContractError("UNSAFE_PATH", "Preview roots cannot contain traversal.")
            validate_path(path)
        self.budget, self.allow_network = budget or PreviewBudget(), allow_network

    def _path(self, file: str | Path) -> Path:
        from lambdaforge.work.snapshot import validate_path

        path = Path(file).absolute()
        validate_path(path)
        if ".." in path.parts or not any(path.is_relative_to(root) for root in self.roots):
            raise ContractError("UNSAFE_PATH", "Preview source is outside approved roots.")
        return path

    def records(self, reader: ReadRecords, file: str | Path) -> dict[str, Any]:
        if type(reader) is not ReadRecords:
            raise ContractError(
                "EFFECT_FORBIDDEN", "Preview cannot invoke arbitrary plugin readers."
            )
        return reader.preview(self._path(file), budget=self.budget)

    def array(self, reader: PreviewArray, file: str | Path) -> dict[str, Any]:
        if type(reader) is not PreviewArray:
            raise ContractError(
                "EFFECT_FORBIDDEN", "Preview cannot invoke arbitrary plugin readers."
            )
        return reader(self._path(file), budget=self.budget)

    def sql(self, reader: ReadSQL) -> dict[str, Any]:
        if type(reader) is not ReadSQL:
            raise ContractError(
                "EFFECT_FORBIDDEN", "Only native SQL readers may preview, never writers."
            )
        if reader.dialect == "postgresql" and not self.allow_network:
            raise ContractError(
                "EFFECT_FORBIDDEN", "Network preview needs explicit operator authorization."
            )
        if reader.file is not None:
            self._path(reader.file)
        return reader.preview(budget=self.budget)

    @contextmanager
    def staging_directory(self) -> Iterator[Path]:
        """An explicit owned scratch scope; no publication or external source mutation."""
        self.budget.check()
        with tempfile.TemporaryDirectory(prefix="lf-preview-") as directory:
            yield Path(directory)
