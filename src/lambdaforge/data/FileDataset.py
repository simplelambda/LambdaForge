"""Lazy adapter for datasets whose samples live in explicit files."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


class FileDataset(Dataset[Any]):
    """Load one explicitly listed file per dataset index.

    The constructor normalizes paths but never opens sample files. The loader
    is called only by __getitem__, making the adapter suitable for importable
    YAML ref callables and spawned DataLoader workers.

    Parameters
    ----------
    files:
        Ordered sample paths. Relative paths are resolved against root or the
        construction-time working directory.
    loader:
        Pickle-safe callable receiving the resolved pathlib.Path.
    root:
        Optional containment root. When provided, paths escaping this root are
        rejected before any data is read.
    """

    def __init__(
        self,
        files: Sequence[str | Path],
        loader: Callable[[Path], Any],
        root: str | Path | None = None,
    ) -> None:
        if isinstance(files, (str, bytes, bytearray)):
            raise TypeError("files must be a sequence of paths, not a string.")
        if not callable(loader):
            raise TypeError("loader must be callable.")

        self.root = Path(root).expanduser().resolve() if root is not None else None
        self.files = tuple(self._resolve_path(path) for path in files)
        self.loader = loader

    def __len__(self) -> int:
        """Return the number of explicitly configured sample files."""
        return len(self.files)

    def __getitem__(self, index: int) -> Any:
        """Load one sample on demand without retaining it in this adapter."""
        path = self.files[index]
        if not path.is_file():
            raise FileNotFoundError(f"Dataset sample file does not exist: {path}")
        return self.loader(path)

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if self.root is not None and not path.is_absolute():
            path = self.root / path
        resolved = path.resolve()
        if self.root is not None and not resolved.is_relative_to(self.root):
            raise ValueError(f"Dataset path escapes root {self.root}: {value}")
        return resolved
