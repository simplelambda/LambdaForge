"""Materialized, content-addressed task artifact."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping
from lambdaforge.experiments.JsonResult import JsonResult
from lambdaforge.tasks.ArtifactDeclaration import ArtifactDeclaration
from lambdaforge.tasks.ArtifactType import ArtifactType


class TaskArtifact(JsonResult):
    """Record a safe run-relative file or directory with its content digest."""

    def __init__(
        self,
        *,
        path: str,
        kind: ArtifactType | str,
        sha256: str,
        size_bytes: int,
        media_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not path or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("Task artifact paths must be non-empty and run-relative.")
        if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
            raise ValueError("Task artifact sha256 must be a lowercase hexadecimal SHA-256.")
        if isinstance(size_bytes, bool) or int(size_bytes) < 0:
            raise ValueError("Task artifact size_bytes must be non-negative.")
        self.path = Path(path).as_posix()
        self.kind = ArtifactType(kind)
        self.sha256 = sha256
        self.size_bytes = int(size_bytes)
        self.media_type = str(media_type) if media_type is not None else None
        self.metadata = FrozenJsonMapping(metadata or {})
        self._freeze_mapping(self.to_dict())

    @classmethod
    def materialize(
        cls,
        declaration: ArtifactDeclaration | str | Path | Mapping[str, Any],
        run_dir: str | Path,
    ) -> TaskArtifact:
        """Validate and hash one declared path below the owning task run."""
        declared = ArtifactDeclaration.from_value(declaration)
        root = Path(run_dir).resolve()
        unresolved = root / str(declared.path)
        if unresolved.is_symlink():
            raise ValueError(f"Task artifacts cannot be symbolic links: {unresolved}")
        path = unresolved.resolve(strict=False)
        if not path.is_relative_to(root):
            raise ValueError(f"Artifact path escapes the task run: {declared.path}")
        if not path.exists():
            raise FileNotFoundError(f"Declared task artifact does not exist: {path}")
        if not path.is_file() and not path.is_dir():
            raise ValueError(f"Task artifacts must be regular files or directories: {path}")
        digest, size = cls.fingerprint_path(path)
        inferred = ArtifactType.DIRECTORY if path.is_dir() else ArtifactType.FILE
        kind = inferred if declared.kind is ArtifactType.OTHER else declared.kind
        return cls(
            path=path.relative_to(root).as_posix(),
            kind=kind,
            sha256=digest,
            size_bytes=size,
            media_type=declared.media_type,
            metadata=declared.metadata,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TaskArtifact:
        """Restore an artifact from its persisted JSON mapping."""
        return cls(
            path=str(value["path"]),
            kind=str(value["kind"]),
            sha256=str(value["sha256"]),
            size_bytes=int(value["size_bytes"]),
            media_type=value.get("media_type"),
            metadata=value.get("metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a defensive JSON-compatible artifact representation."""
        payload: dict[str, Any] = {
            "path": self.path,
            "kind": self.kind.value,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "metadata": copy.deepcopy(self.metadata),
        }
        if self.media_type is not None:
            payload["media_type"] = self.media_type
        return payload

    @staticmethod
    def fingerprint_path(path: str | Path) -> tuple[str, int]:
        """Hash one regular file or directory tree deterministically."""
        path = Path(path)
        digest = hashlib.sha256()
        size = 0
        if path.is_symlink():
            raise ValueError(f"Task artifact paths cannot be symbolic links: {path}")
        if path.is_file():
            paths = [path]
        elif path.is_dir():
            entries = sorted(path.rglob("*"))
            symlinks = [item for item in entries if item.is_symlink()]
            if symlinks:
                raise ValueError(
                    f"Task artifact trees cannot contain symbolic links: {symlinks[0]}"
                )
            paths = [item for item in entries if item.is_file()]
        else:
            raise ValueError(f"Task artifact paths must be regular files or directories: {path}")
        for item in paths:
            relative = item.name if path.is_file() else item.relative_to(path).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            with item.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
        return digest.hexdigest(), size
