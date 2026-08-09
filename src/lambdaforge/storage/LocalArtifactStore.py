"""Local/shared-filesystem artifact store."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from uuid import uuid4

from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.storage.ArtifactReference import ArtifactReference
from lambdaforge.storage.ArtifactStore import ArtifactStore


class LocalArtifactStore(ArtifactStore):
    """Publish immutable files safely on local or shared POSIX/Windows filesystems."""

    def __init__(self, root: str | Path, *, name: str = "local") -> None:
        self.root = Path(root).resolve()
        self.name = name

    def publish(
        self, source: str | Path, *, key: str | None = None, media_type: str | None = None
    ) -> ArtifactReference:
        """Copy under a per-key lock and reject conflicting immutable content."""
        source_path = Path(source)
        if not source_path.is_file() or source_path.is_symlink():
            raise ValueError(f"Artifact source must be a regular file: {source_path}")
        digest = self._digest(source_path)
        resolved_key = key or f"sha256/{digest}/{source_path.name}"
        destination = self._path(resolved_key)
        reference = ArtifactReference(
            self.name, resolved_key, f"sha256:{digest}", source_path.stat().st_size, media_type
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with CrossProcessFileLock(
            destination.with_suffix(destination.suffix + ".lock"),
            shared=False,
            timeout_seconds=60,
            poll_interval_seconds=0.05,
        ):
            if destination.exists():
                if not self.exists(reference):
                    raise FileExistsError(
                        f"Immutable artifact key contains different content: {resolved_key}"
                    )
                return reference
            temporary = destination.with_name(
                f".{destination.name}.{os.getpid()}.{uuid4().hex}.tmp"
            )
            try:
                shutil.copyfile(source_path, temporary)
                if self._digest(temporary) != digest:
                    raise OSError("Artifact changed while it was being published.")
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        return reference

    def stage(self, reference: ArtifactReference, destination: str | Path) -> Path:
        """Copy verified content atomically to a consumer path."""
        if reference.store != self.name or not self.exists(reference):
            raise FileNotFoundError(
                f"Artifact reference is unavailable or corrupt: {reference.key}"
            )
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination_path.with_name(
            f".{destination_path.name}.{os.getpid()}.{uuid4().hex}.tmp"
        )
        try:
            shutil.copyfile(self._path(reference.key), temporary)
            if self._digest(temporary) != reference.sha256.removeprefix("sha256:"):
                raise OSError("Staged artifact failed checksum validation.")
            os.replace(temporary, destination_path)
        finally:
            temporary.unlink(missing_ok=True)
        return destination_path

    def exists(self, reference: ArtifactReference) -> bool:
        """Check store identity, size and checksum."""
        path = self._path(reference.key)
        return (
            reference.store == self.name
            and path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == reference.size_bytes
            and self._digest(path) == reference.sha256.removeprefix("sha256:")
        )

    def _path(self, key: str) -> Path:
        relative = Path(key)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe artifact key: {key!r}")
        destination = (self.root / relative).resolve(strict=False)
        if not destination.is_relative_to(self.root):
            raise ValueError(f"Artifact key escapes store: {key!r}")
        return destination

    @staticmethod
    def _digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
