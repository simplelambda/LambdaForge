"""Deterministic file-tree record source."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from lambdaforge.preprocessing.PreprocessingRecord import PreprocessingRecord
from lambdaforge.preprocessing.PreprocessingSource import PreprocessingSource
from lambdaforge.tasks.TaskContext import TaskContext


class FileTreeSource(PreprocessingSource):
    """Yield regular files below a YAML-relative root using a glob pattern."""

    def __init__(self, root: str | Path, pattern: str = "**/*") -> None:
        if not str(pattern).strip():
            raise ValueError("FileTreeSource.pattern cannot be empty.")
        self.root = Path(root)
        self.pattern = str(pattern)

    def records(self, context: TaskContext) -> Iterable[PreprocessingRecord]:
        """Yield sorted non-symlink files with paths relative to the source root."""
        root = context.declared_input_path(self.root)
        if not root.is_dir():
            raise NotADirectoryError(f"FileTreeSource root is not a directory: {root}")
        for path in sorted(root.glob(self.pattern)):
            if path.is_symlink():
                raise ValueError(f"FileTreeSource does not follow symbolic links: {path}")
            if path.is_file():
                stat = path.stat()
                yield PreprocessingRecord(
                    key=path.relative_to(root).as_posix(),
                    value=path,
                    metadata={"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
                )
