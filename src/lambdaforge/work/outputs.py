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

    def html_section(
        self,
        name: str,
        *,
        title: str | None = None,
        section: str | None = None,
        filename: str = "index.html",
    ) -> ManagedOutput:
        """Declare self-contained project HTML for an isolated Study report tab.

        Write the document with ``write_text``. Repeated Runs share the same tab;
        ``section`` optionally groups several named HTML outputs in that tab.
        Documents must embed their own assets/data; external requests are blocked.
        HTML is collected only when explicitly generating a report, never by HPO.
        """
        selected_section = section or name
        selected_title = title or selected_section
        if not isinstance(selected_section, str) or not 1 <= len(selected_section.strip()) <= 80:
            raise ValueError("HTML section must have a non-empty name of at most 80 characters.")
        if not isinstance(selected_title, str) or not 1 <= len(selected_title.strip()) <= 120:
            raise ValueError("HTML section title must contain 1–120 characters.")
        return self.file(
            name,
            filename=filename,
            role="html-section",
            media_type="text/html",
            metadata={"html_section": {"name": selected_section, "title": selected_title}},
        )

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
                    f"A managed directory cannot be published inside itself: {destination}."
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
        from lambdaforge.work.snapshot import copy_file, copy_tree

        destination = self._publication_path(pending.publish_to)
        if destination == source:
            return destination
        self._preflight_publication(source, destination, pending.overwrite)
        same_kind = destination.is_file() if source.is_file() else destination.is_dir()
        if destination.exists() and same_kind and fingerprint(destination) == fingerprint(source):
            return destination
        self._reject_symlink_ancestors(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._reject_symlink_ancestors(destination)
        if source.is_file():
            if pending.overwrite:
                return atomic_build_file(
                    destination, lambda temporary: copy_file(source, temporary)
                )
            staging = destination.with_name(
                f".{destination.name}.{os.getpid()}.{uuid4().hex}.lambdaforge-tmp"
            )
            try:
                copy_file(source, staging)
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
            copy_tree(source, staging)
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
        source_checkpoint: str | None = None,
        release_checkpoints: Iterable[str] = (),
        scientific_identity: Mapping[str, Any] | None = None,
        intent: str = "publish",
    ) -> Mapping[str, Any]:
        """Stream, verify and atomically publish one independent DatasetVersion."""
        output_name = self._new_name(output)
        release_names = list(release_checkpoints)
        for checkpoint in release_names:
            selected = owned_path(self._runtime.checkpoints._root, checkpoint, must_exist=True)
            if selected == self._runtime.checkpoints._root:
                raise ValueError("Release named checkpoints, not the collection root.")
        source_root = (
            self._runtime.checkpoints.path(source_checkpoint, create_parent=False)
            if source_checkpoint is not None
            else self._runtime.run_dir
        )
        if source_checkpoint is not None:
            owned_path(self._runtime.checkpoints._root, source_root, must_exist=True)
            if source_root == self._runtime.checkpoints._root or not source_root.is_dir():
                raise ValueError("Dataset source_checkpoint must name an owned directory.")
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
        recovery_root = self._runtime.checkpoints.path(f"dataset-publications/{output_name}")
        if intent == "rebuild":
            DatasetPublisher(registry).preflight(name, version, intent="rebuild")
            publication_root = recovery_root
        from lambdaforge.work.models import atomic_json

        pending = self._runtime.run_dir / "dataset-publication-pending.json"
        atomic_json(
            pending,
            {"dataset": f"{name}@{version}", "intent": intent, "recovery_root": str(recovery_root)},
        )
        try:
            record = DatasetPublisher(registry).publish_members(
                name,
                version,
                members,
                source_root=source_root,
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
                scientific_identity=scientific_identity,
                intent=intent,
                recovery_root=recovery_root if intent == "publish" else None,
            )
        except Exception:
            self._runtime.checkpoints.save_json(
                "dataset-publication-failure.json",
                {
                    "dataset": f"{name}@{version}",
                    "recovery_root": str(recovery_root),
                    "preserve_artifacts": True,
                },
            )
            raise
        pending.unlink(missing_ok=True)
        payload = record.to_dict()
        if intent == "rebuild":
            payload["publication_status"] = "reconstructed-unregistered"
            self._runtime.checkpoints.pin(
                f"dataset-publications/{output_name}",
                reason="dataset reproducibility comparison",
            )
            self._values[output_name] = payload
            return payload
        self._datasets[output_name] = payload
        if release_names:
            self._runtime.checkpoints._publication_committed(
                release_names,
                {
                    "kind": "dataset",
                    "dataset_id": record.dataset_id,
                    "placements": [placement.to_dict() for placement in record.placements],
                },
            )
        return payload

    def dataset_preflight(
        self,
        *,
        name: str,
        version: str,
        intent: str = "publish",
    ) -> Mapping[str, Any]:
        """Check the publication target before any expensive scientific calculation.

        Call as the first operation in a producing Work. It does not enumerate
        members or execute computation and cannot waive the final identity check.
        """
        registry = DatasetRegistry(
            os.environ.get("LAMBDAFORGE_DATASET_REGISTRY")
            or DatasetRegistry.project_path(self._runtime.source_dir)
        )
        return DatasetPublisher(registry).preflight(name, version, intent=intent)

    def from_checkpoint(
        self,
        name: str,
        checkpoint: str,
        *,
        role: str = "model",
        release: bool = False,
    ) -> Path:
        """Seal a checkpoint into an independent durable artifact, without scratch trees."""
        from lambdaforge.work.snapshot import copy_file, copy_tree

        source = owned_path(self._runtime.checkpoints._root, checkpoint, must_exist=True)
        if source == self._runtime.checkpoints._root:
            raise ValueError("Publish a named checkpoint, not the collection root.")
        selected = self._new_name(name)
        destination = owned_path(
            self._runtime.run_dir, Path("artifacts") / self._safe_name(selected)
        )
        if source.is_file():
            destination /= source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        identity = fingerprint(source)
        if source.is_dir():
            copy_tree(source, destination)
        else:
            copy_file(source, destination)
        if fingerprint(source) != identity or fingerprint(destination) != identity:
            raise ValueError(
                "Checkpoint changed during publication; refusing an unsealed snapshot."
            )
        digest, size = identity
        self._artifacts[selected] = WorkArtifact(
            selected,
            destination.relative_to(self._runtime.run_dir).as_posix(),
            role,
            digest,
            size,
            None,
            {"checkpoint_source": checkpoint},
        )
        if release:
            self._runtime.checkpoints._publication_committed(
                [checkpoint],
                {
                    "kind": "artifact",
                    "path": str(destination),
                    "sha256": digest,
                    "size_bytes": size,
                },
            )
        return destination

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
