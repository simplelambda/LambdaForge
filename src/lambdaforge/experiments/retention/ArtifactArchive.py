"""Verified immutable ZIP archive for retained run artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

from lambdaforge.experiments.retention.ArtifactPathGuard import ArtifactPathGuard
from lambdaforge.experiments.retention.ArtifactRetentionOperation import (
    ArtifactRetentionOperation,
)


class ArtifactArchive:
    """Stream, verify and atomically publish one immutable per-run ZIP."""

    MANIFEST_NAME = ".lambdaforge-retention-manifest.json"

    def __init__(
        self,
        run_dir: str | Path,
        *,
        configured_name: str,
        plan_id: str,
        compression_level: int,
    ) -> None:
        self.run_dir = Path(run_dir)
        stem = Path(configured_name).stem
        self.path = (
            self.run_dir
            / ".lambdaforge"
            / "retention"
            / f"{stem}-l{compression_level}-{plan_id[:12]}.zip"
        )
        self.plan_id = str(plan_id)
        self.compression_level = int(compression_level)

    def write(
        self,
        operations: tuple[ArtifactRetentionOperation, ...],
    ) -> tuple[dict[str, Any] | None, tuple[ArtifactRetentionOperation, ...]]:
        """Publish verified selected members without deleting their sources."""
        if not operations:
            return None, ()
        archive_dir = ArtifactPathGuard.ensure_directory(
            self.run_dir,
            ".lambdaforge/retention",
        )
        try:
            self.path.lstat()
        except FileNotFoundError:
            pass
        else:
            if ArtifactPathGuard.relative_regular_file(archive_dir, self.path) != self.path.name:
                raise ValueError(f"Unsafe existing retention archive: {self.path}.")
            self._validate_archive(self.path, operations, allow_incompressible=True)
            included = self._included_operations(self.path, operations)
            return self._metadata(self.path, included), included

        probe = self.path.with_name(f".{self.path.name}.{os.getpid()}.{uuid4().hex}.probe")
        final = self.path.with_name(f".{self.path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            self._write_zip(probe, operations, include_manifest=False)
            self._validate_archive(probe, operations, allow_incompressible=True)
            included = self._included_operations(probe, operations)
            if not included:
                return None, ()
            self._write_zip(final, included, include_manifest=True)
            self._validate_archive(final, included, allow_incompressible=True)
            if all(
                operation.only_if_smaller for operation in included
            ) and final.stat().st_size >= sum(operation.size_bytes for operation in included):
                return None, ()
            try:
                os.link(final, self.path)
            except FileExistsError as error:
                raise FileExistsError(f"Retention archive already exists: {self.path}") from error
            final.unlink()
            return self._metadata(self.path, included), included
        finally:
            probe.unlink(missing_ok=True)
            final.unlink(missing_ok=True)

    def _write_zip(
        self,
        path: Path,
        operations: tuple[ArtifactRetentionOperation, ...],
        *,
        include_manifest: bool,
    ) -> None:
        with ZipFile(
            path,
            mode="w",
            compression=ZIP_DEFLATED,
            compresslevel=self.compression_level,
            allowZip64=True,
            strict_timestamps=False,
        ) as archive:
            for operation in operations:
                source = self.run_dir.joinpath(*PurePosixPath(operation.relative_path).parts)
                self._verify_source(source, operation)
                archive.write(source, arcname=operation.relative_path)
            if include_manifest:
                manifest = {
                    "format_version": 1,
                    "plan_id": self.plan_id,
                    "compression_level": self.compression_level,
                    "members": [
                        {
                            "path": operation.relative_path,
                            "sha256": operation.sha256,
                            "size_bytes": operation.size_bytes,
                        }
                        for operation in operations
                    ],
                }
                archive.writestr(
                    self.MANIFEST_NAME,
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                )
        with path.open("r+b") as handle:
            os.fsync(handle.fileno())

    def _validate_archive(
        self,
        path: Path,
        operations: tuple[ArtifactRetentionOperation, ...],
        *,
        allow_incompressible: bool,
    ) -> None:
        try:
            with ZipFile(path, mode="r") as archive:
                corrupt = archive.testzip()
                if corrupt is not None:
                    raise ValueError(f"ZIP CRC validation failed for member {corrupt!r}.")
                names = archive.namelist()
                if len(names) != len(set(names)):
                    raise ValueError("Retention ZIP contains duplicate member names.")
                for name in names:
                    pure = PurePosixPath(name)
                    if pure.is_absolute() or ".." in pure.parts or "\\" in name:
                        raise ValueError(f"Unsafe retention ZIP member {name!r}.")
                expected_names = {operation.relative_path for operation in operations}
                actual_names = set(names) - {self.MANIFEST_NAME}
                if actual_names != expected_names:
                    raise ValueError("Retention ZIP members differ from the transaction plan.")
                for operation in operations:
                    digest = hashlib.sha256()
                    with archive.open(operation.relative_path, mode="r") as member:
                        while chunk := member.read(1024 * 1024):
                            digest.update(chunk)
                    if digest.hexdigest() != operation.sha256:
                        raise ValueError(
                            f"Retention ZIP hash mismatch for {operation.relative_path!r}."
                        )
                    info = archive.getinfo(operation.relative_path)
                    if info.file_size != operation.size_bytes:
                        raise ValueError(
                            f"Retention ZIP size mismatch for {operation.relative_path!r}."
                        )
                    if (
                        not allow_incompressible
                        and operation.only_if_smaller
                        and info.compress_size >= info.file_size
                    ):
                        raise ValueError(
                            f"Retention ZIP did not reduce {operation.relative_path!r}."
                        )
        except BadZipFile as error:
            raise ValueError(f"Invalid retention ZIP: {path}.") from error

    def _included_operations(
        self,
        path: Path,
        operations: tuple[ArtifactRetentionOperation, ...],
    ) -> tuple[ArtifactRetentionOperation, ...]:
        with ZipFile(path, mode="r") as archive:
            return tuple(
                operation
                for operation in operations
                if not operation.only_if_smaller
                or archive.getinfo(operation.relative_path).compress_size < operation.size_bytes
            )

    def _metadata(
        self,
        path: Path,
        operations: tuple[ArtifactRetentionOperation, ...],
    ) -> dict[str, Any]:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return {
            "run_relative": operations[0].run_relative,
            "path": path.relative_to(self.run_dir).as_posix(),
            "compression_level": self.compression_level,
            "sha256": digest.hexdigest(),
            "size_bytes": path.stat().st_size,
            "members": [operation.relative_path for operation in operations],
        }

    def _verify_source(
        self,
        path: Path,
        operation: ArtifactRetentionOperation,
    ) -> None:
        relative = ArtifactPathGuard.relative_regular_file(self.run_dir, path)
        if relative != operation.relative_path:
            raise ValueError(f"Unsafe or replaced retention source: {operation.key}.")
        metadata = path.stat()
        if metadata.st_size != operation.size_bytes or metadata.st_mtime_ns != operation.mtime_ns:
            raise ValueError(f"Retention source changed after preview: {operation.key}.")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != operation.sha256:
            raise ValueError(f"Retention source hash changed after preview: {operation.key}.")
