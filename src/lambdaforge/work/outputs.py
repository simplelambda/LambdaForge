"""Attempt-owned values, artifacts and immutable dataset publication."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from lambdaforge.data.DatasetPublisher import DatasetPublisher
from lambdaforge.data.DatasetRegistry import DatasetRegistry
from lambdaforge.work.atomic import atomic_build_file
from lambdaforge.work.managed import ManagedOutput, fingerprint, owned_path
from lambdaforge.work.models import WorkArtifact

if TYPE_CHECKING:
    from lambdaforge.work.runtime import WorkRuntime


@dataclass(frozen=True, slots=True)
class _PendingOutput:
    output: ManagedOutput
    role: str
    media_type: str | None
    metadata: Mapping[str, Any]
    publish_to: str | Path | None
    overwrite: bool
    retain_internal: bool


class OutputCollection:
    """Register named structured values, artifacts and immutable datasets."""

    def __init__(self, runtime: WorkRuntime) -> None:
        self._runtime = runtime
        self._values: dict[str, Any] = {}
        self._artifacts: dict[str, WorkArtifact] = {}
        self._datasets: dict[str, Mapping[str, Any]] = {}
        self._pending: dict[str, _PendingOutput] = {}

    def value(self, name: str, value: Any) -> None:
        """Register one immutable JSON-compatible named value."""
        selected = self._new_name(name)
        try:
            encoded = json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise TypeError(f"Output {selected!r} must be JSON-compatible: {error}") from error
        self._values[selected] = json.loads(encoded)

    def artifact(
        self,
        name: str,
        path: str | Path,
        *,
        role: str = "artifact",
        media_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Import one existing safe path into attempt-owned artifact storage."""
        selected = self._new_name(name)
        raw = Path(path)
        source = (
            raw.resolve(strict=False)
            if raw.is_absolute()
            else (self._runtime.run_dir / raw).resolve(strict=False)
        )
        if not source.exists() or source.is_symlink():
            raise FileNotFoundError(f"Registered artifact is missing or symbolic: {source}")
        if source.is_relative_to(self._runtime.run_dir.resolve()):
            managed = owned_path(self._runtime.run_dir, source, must_exist=True)
        else:
            destination = owned_path(
                self._runtime.run_dir,
                Path("artifacts") / self._safe_name(selected),
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(f"Managed artifact destination already exists: {destination}")
            if source.is_dir():
                shutil.copytree(source, destination)
            elif source.is_file():
                shutil.copy2(source, destination)
            else:
                raise ValueError(f"Unsupported artifact path: {source}")
            managed = destination
        digest, size = fingerprint(managed)
        self._artifacts[selected] = WorkArtifact(
            selected,
            managed.relative_to(self._runtime.run_dir).as_posix(),
            str(role),
            digest,
            size,
            media_type,
            dict(metadata or {}),
        )
        return managed

    def file(
        self,
        name: str,
        *,
        filename: str | None = None,
        role: str = "artifact",
        media_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        publish_to: str | Path | None = None,
        overwrite: bool = False,
        retain_internal: bool = False,
    ) -> ManagedOutput:
        """Declare a managed file and optionally publish a verified external copy.

        Published outputs default to one durable external copy. Set
        ``retain_internal=True`` only when a second Job-owned copy is deliberately needed.
        """
        self._validate_publication_options(publish_to, overwrite, retain_internal)
        selected = self._new_name(name)
        root = owned_path(
            self._runtime.run_dir,
            Path("artifacts") / self._safe_name(selected),
        )
        root.mkdir(parents=True, exist_ok=False)
        destination = owned_path(root, filename or selected)
        destination.parent.mkdir(parents=True, exist_ok=True)
        output = ManagedOutput(destination, kind="file")
        self._pending[selected] = _PendingOutput(
            output,
            str(role),
            media_type,
            dict(metadata or {}),
            publish_to,
            bool(overwrite),
            bool(retain_internal),
        )
        return output

    def directory(
        self,
        name: str,
        *,
        role: str = "artifact",
        media_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        publish_to: str | Path | None = None,
        overwrite: bool = False,
        retain_internal: bool = False,
    ) -> ManagedOutput:
        """Declare a managed directory and optionally publish one durable external copy."""
        self._validate_publication_options(publish_to, overwrite, retain_internal)
        selected = self._new_name(name)
        destination = owned_path(
            self._runtime.run_dir,
            Path("artifacts") / self._safe_name(selected),
        )
        destination.mkdir(parents=True, exist_ok=False)
        output = ManagedOutput(destination, kind="directory")
        self._pending[selected] = _PendingOutput(
            output,
            str(role),
            media_type,
            dict(metadata or {}),
            publish_to,
            bool(overwrite),
            bool(retain_internal),
        )
        return output

    def finalize(self) -> None:
        """Verify all declared managed outputs and register them as one atomic set."""
        verified: dict[str, tuple[_PendingOutput, Path, str, int]] = {}
        for name, pending in self._pending.items():
            path = owned_path(self._runtime.run_dir, pending.output.path, must_exist=True)
            expected = path.is_file() if pending.output.kind == "file" else path.is_dir()
            if not expected or path.is_symlink():
                raise ValueError(
                    f"Managed {pending.output.kind} output {name!r} was not created safely "
                    f"at {path}."
                )
            digest, size = fingerprint(path)
            verified[name] = (pending, path, digest, size)

        publication_paths: dict[Path, str] = {}
        for name, (pending, path, _digest, _size) in verified.items():
            if pending.publish_to is None:
                continue
            destination = self._publication_path(pending.publish_to)
            if destination in publication_paths:
                raise ValueError(
                    f"Outputs {publication_paths[destination]!r} and {name!r} publish to the "
                    f"same destination: {destination}."
                )
            publication_paths[destination] = name
            self._preflight_publication(path, destination, pending.overwrite)

        finalized: dict[str, WorkArtifact] = {}
        for name, (pending, path, digest, size) in verified.items():
            metadata = dict(pending.metadata)
            if pending.publish_to is not None:
                published = self._publish(path, pending)
                metadata["published_to"] = str(published)
                metadata["retention"] = (
                    "published-and-internal" if pending.retain_internal else "published-only"
                )
            finalized[name] = WorkArtifact(
                name,
                path.relative_to(self._runtime.run_dir).as_posix(),
                pending.role,
                digest,
                size,
                pending.media_type,
                metadata,
            )
        self._artifacts.update(finalized)
        self._pending.clear()

    def _preflight_publication(
        self,
        source: Path,
        destination: Path,
        overwrite: bool,
    ) -> None:
        self._reject_symlink_ancestors(destination)
        if source.is_dir() and destination != source:
            if destination.is_relative_to(source):
                raise ValueError(
                    "A managed directory cannot be published inside itself: "
                    f"{destination}."
                )
            if source.is_relative_to(destination):
                raise ValueError(
                    "A managed directory cannot replace a directory that contains its "
                    f"authoritative source: {destination}."
                )
        if not destination.exists():
            return
        if destination.is_symlink():
            raise ValueError(f"Output publication cannot replace a symbolic link: {destination}")
        same_kind = destination.is_file() if source.is_file() else destination.is_dir()
        if not same_kind:
            raise ValueError(
                "Output publication cannot replace a directory with a file or a file with "
                f"a directory: {destination}."
            )
        if same_kind and fingerprint(destination) == fingerprint(source):
            return
        if not overwrite:
            raise FileExistsError(
                f"Output publication destination already exists: {destination}. "
                "Choose another publish_to path or set overwrite=True explicitly."
            )

    def _publish(self, source: Path, pending: _PendingOutput) -> Path:
        destination = self._publication_path(pending.publish_to)
        if destination == source:
            return destination
        self._preflight_publication(source, destination, pending.overwrite)
        same_kind = (
            destination.is_file() if source.is_file() else destination.is_dir()
        )
        if destination.exists() and same_kind and fingerprint(destination) == fingerprint(source):
            return destination
        self._reject_symlink_ancestors(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._reject_symlink_ancestors(destination)
        if source.is_file():
            if pending.overwrite:
                return atomic_build_file(
                    destination, lambda temporary: shutil.copy2(source, temporary)
                )
            staging = destination.with_name(
                f".{destination.name}.{os.getpid()}.{uuid4().hex}.lambdaforge-tmp"
            )
            try:
                shutil.copy2(source, staging)
                with staging.open("rb") as handle:
                    os.fsync(handle.fileno())
                try:
                    os.link(staging, destination)
                except FileExistsError as error:
                    raise FileExistsError(
                        f"Output publication destination appeared concurrently: {destination}."
                    ) from error
            finally:
                staging.unlink(missing_ok=True)
            return destination
        staging = destination.with_name(
            f".{destination.name}.{os.getpid()}.{uuid4().hex}.lambdaforge-tmp"
        )
        backup = destination.with_name(
            f".{destination.name}.{os.getpid()}.{uuid4().hex}.lambdaforge-backup"
        )
        try:
            shutil.copytree(source, staging)
            if destination.exists():
                if not pending.overwrite:
                    raise FileExistsError(
                        f"Output publication destination appeared concurrently: {destination}."
                    )
                os.replace(destination, backup)
            try:
                os.replace(staging, destination)
            except BaseException:
                if backup.exists() and not destination.exists():
                    os.replace(backup, destination)
                raise
            if backup.exists():
                shutil.rmtree(backup)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
            if backup.exists() and destination.exists():
                shutil.rmtree(backup)
        return destination

    def _publication_path(self, value: str | Path | None) -> Path:
        if value is None:
            raise ValueError("Output publication requires a destination path.")
        return self._runtime.path_context.publication_path(value)

    @staticmethod
    def _validate_publication_options(
        publish_to: str | Path | None,
        overwrite: bool,
        retain_internal: bool,
    ) -> None:
        if not isinstance(overwrite, bool):
            raise TypeError("Output overwrite must be true or false.")
        if not isinstance(retain_internal, bool):
            raise TypeError("Output retain_internal must be true or false.")
        if publish_to is None:
            if overwrite:
                raise ValueError("Output overwrite=True requires publish_to.")
            if retain_internal:
                raise ValueError("Output retain_internal=True requires publish_to.")
            return
        if not isinstance(publish_to, str | Path) or not str(publish_to).strip():
            raise ValueError("Output publish_to must be a non-empty path.")

    @staticmethod
    def _reject_symlink_ancestors(destination: Path) -> None:
        for candidate in (destination, *destination.parents):
            if candidate.is_symlink():
                raise ValueError(
                    f"Output publication paths cannot traverse symbolic links: {candidate}"
                )

    def dataset(
        self,
        *,
        name: str,
        version: str,
        members: Iterable[Mapping[str, Any]],
        output: str = "dataset",
        metadata: Mapping[str, Any] | None = None,
        target_schema: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Stream, verify and atomically publish one independent DatasetVersion."""
        output_name = self._new_name(output)
        cluster = os.environ.get("LAMBDAFORGE_CLUSTER", "local")
        configured_root = os.environ.get("LAMBDAFORGE_DATASET_ROOT")
        if cluster != "local" and configured_root is None:
            raise RuntimeError(
                f"Cluster {cluster!r} has no permanent storage.dataset_root for publication."
            )
        publication_root = Path(
            configured_root or self._runtime.source_dir / ".lambdaforge" / "datasets" / "published"
        )
        registry = DatasetRegistry(
            os.environ.get("LAMBDAFORGE_DATASET_REGISTRY")
            or DatasetRegistry.project_path(self._runtime.source_dir)
        )
        record = DatasetPublisher(registry).publish_members(
            name,
            version,
            members,
            source_root=self._runtime.run_dir,
            publication_root=publication_root,
            build_provenance={
                "work_class": self._runtime.config.work_class,
                "scientific_fingerprint": self._runtime.scientific_fingerprint,
                "execution_id": self._runtime.execution_id,
                "run_id": self._runtime.run_id,
                "attempt_id": self._runtime.attempt_id,
            },
            cluster=cluster,
            metadata=metadata,
            target_schema=target_schema,
        )
        payload = record.to_dict()
        self._datasets[output_name] = payload
        return payload

    @property
    def values(self) -> Mapping[str, Any]:
        return dict(self._values)

    @property
    def artifacts(self) -> tuple[WorkArtifact, ...]:
        return tuple(self._artifacts.values())

    @property
    def datasets(self) -> Mapping[str, Mapping[str, Any]]:
        return dict(self._datasets)

    def _new_name(self, name: str) -> str:
        selected = str(name).strip()
        if not selected:
            raise ValueError("Output names cannot be empty.")
        if (
            selected in self._values
            or selected in self._artifacts
            or selected in self._datasets
            or selected in self._pending
        ):
            raise ValueError(f"Output {selected!r} is already registered in this attempt.")
        return selected

    @staticmethod
    def _safe_name(value: str) -> str:
        name = "".join(
            character if character.isalnum() or character in "-_." else "-" for character in value
        )
        name = name.strip(".-")
        if not name:
            raise ValueError("Artifact names require at least one portable character.")
        return f"{name[:80]}-{hashlib.sha256(value.encode()).hexdigest()[:10]}"


__all__ = ["OutputCollection"]
