"""Atomic validation and publication boundary for DatasetArtifact v2."""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from lambdaforge.data.DatasetArtifact import DatasetArtifact
from lambdaforge.data.DatasetOperations import DatasetOperations
from lambdaforge.data.DatasetRecord import DatasetRecord
from lambdaforge.data.DatasetRegistry import DatasetRegistry
from lambdaforge.data.errors import InvalidDatasetPublicationError
from lambdaforge.data.index import DatasetAsset, DatasetIndex, DatasetMember
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock


class DatasetPublisher:
    """Publish complete validated bytes atomically and register only after commit."""

    def __init__(self, registry: DatasetRegistry | None = None) -> None:
        self.registry = registry or DatasetRegistry()

    def preflight(self, name: str, version: str, *, intent: str = "publish") -> dict[str, Any]:
        """Call before computation: publish new bytes, rebuild, or reuse exact bytes."""
        name = self._publication_label(name, field="name")
        version = self._publication_label(version, field="version")
        if intent not in {"publish", "rebuild", "reuse"}:
            raise ValueError("Dataset intent must be publish, rebuild or reuse.")
        try:
            existing = self.registry.get(f"{name}@{version}")
        except KeyError:
            existing = None
        if intent == "publish" and existing is not None:
            raise InvalidDatasetPublicationError(
                f"Dataset {existing.key} is already published ({existing.dataset_id}). "
                "Before computing: reuse/materialize exact bytes, choose rebuild for a "
                "comparison without registration, or publish under a new version."
            )
        if intent in {"reuse", "rebuild"} and existing is None:
            raise InvalidDatasetPublicationError(
                f"Dataset {name}@{version} is not registered; {intent} requires a reference."
            )
        return {
            "dataset": f"{name}@{version}",
            "intent": intent,
            "existing": existing.to_dict() if existing else None,
        }

    def publish_members(
        self,
        name: str,
        version: str,
        members: Iterable[Mapping[str, Any]],
        *,
        source_root: str | Path,
        publication_root: str | Path,
        build_provenance: Mapping[str, Any],
        cluster: str = "local",
        metadata: Mapping[str, Any] | None = None,
        target_schema: Mapping[str, Any] | None = None,
        scientific_identity: Mapping[str, Any] | None = None,
        intent: str = "publish",
        recovery_root: str | Path | None = None,
    ) -> DatasetRecord:
        """Stream ordinary member mappings into one verified immutable publication."""
        dataset_name = self._publication_label(name, field="name")
        dataset_version = self._publication_label(version, field="version")
        if intent not in {"publish", "rebuild"}:
            raise ValueError("Member publication intent must be publish or rebuild.")
        register = intent == "publish"
        from lambdaforge.work.snapshot import validate_path

        validate_path(Path(source_root).expanduser().absolute())
        validate_path(Path(publication_root).expanduser().absolute())
        source = Path(source_root).resolve()
        if not source.is_dir() or source.is_symlink():
            raise ValueError("Dataset member source_root must be a safe directory.")
        root = Path(publication_root).expanduser().resolve()
        staging_parent = root / dataset_name / dataset_version
        staging = staging_parent / f".publication.{os.getpid()}.{uuid4().hex}.tmp"
        staging.mkdir(parents=True, exist_ok=False)
        index_path = staging / "index.jsonl"
        sealed = False
        committed: Path | None = None
        commit_lock = CrossProcessFileLock(
            staging_parent.parent / f".{dataset_version}.publication.lock",
            shared=False,
            timeout_seconds=30,
            poll_interval_seconds=0.1,
        )
        try:
            index = DatasetIndex.write(
                index_path,
                self._materialize_members(members, source=source, staging=staging),
            )
            validation = index.validate(
                staging, target_schema=target_schema, require_checksums=True
            )
            if not validation["valid"]:
                raise InvalidDatasetPublicationError(
                    "DatasetIndex validation failed: " + "; ".join(validation["errors"])
                )
            artifact = DatasetArtifact.create_v2(
                name=dataset_name,
                version=dataset_version,
                index=index,
                index_path="index.jsonl",
                build_provenance=build_provenance,
                target_schema=target_schema,
                metadata=metadata,
                scientific_identity=scientific_identity,
            )
            artifact.write_json(staging / "dataset-artifact.json")
            verification = DatasetOperations.verify(staging, artifact.dataset_id)
            if not verification["valid"]:
                raise InvalidDatasetPublicationError(
                    "Staged dataset failed verification: " + "; ".join(verification["errors"])
                )
            sealed = True
            commit_lock.acquire()
            destination = staging_parent / artifact.dataset_id.removeprefix("sha256:")[:16]
            if destination.exists():
                existing = destination / "dataset-artifact.json"
                if (
                    existing.is_file()
                    and not existing.is_symlink()
                    and DatasetArtifact.read_json(existing).dataset_id == artifact.dataset_id
                    and DatasetArtifact.read_json(existing).scientific_id == artifact.scientific_id
                    and DatasetArtifact.read_json(existing).name == artifact.name
                    and DatasetArtifact.read_json(existing).version == artifact.version
                    and DatasetOperations.verify(destination, artifact.dataset_id)["valid"]
                ):
                    describe = (
                        self.registry.register_artifact
                        if register
                        else self.registry.artifact_record
                    )
                    return describe(
                        existing, cluster=cluster, root=destination, producer=build_provenance
                    )
                raise FileExistsError(
                    "Dataset publication destination already exists and is not reusable: "
                    f"{destination}"
                )
            if register:
                self._require_compatible_registration(
                    artifact.name,
                    artifact.version,
                    artifact.dataset_id,
                    scientific_identity=artifact.scientific_identity,
                )
            os.replace(staging, destination)
            committed = destination
            if not register:
                return self.registry.artifact_record(
                    destination / "dataset-artifact.json",
                    cluster=cluster,
                    root=destination,
                )
            try:
                return self.registry.register_artifact(
                    destination / "dataset-artifact.json",
                    cluster=cluster,
                    root=destination,
                    producer=build_provenance,
                )
            except Exception:
                # Registration is the last publication step. A concurrent
                # conflict or index-write failure must not leave untracked bytes.
                if recovery_root is None and destination.is_dir() and not destination.is_symlink():
                    shutil.rmtree(destination)
                raise
        except Exception as error:
            if sealed and recovery_root is not None:
                retained = Path(recovery_root).expanduser().absolute()
                validate_path(retained)
                candidate = retained / (
                    artifact.dataset_id.removeprefix("sha256:") + f"-{uuid4().hex}"
                )
                owned = committed or staging
                try:
                    retained.mkdir(parents=True, exist_ok=True)
                    try:
                        os.replace(owned, candidate)
                    except OSError as move_error:
                        if move_error.errno != errno.EXDEV:
                            raise
                        from lambdaforge.work.snapshot import copy_tree

                        copy_tree(owned, candidate)
                        if not DatasetOperations.verify(candidate, artifact.dataset_id)["valid"]:
                            raise OSError("Recovery copy failed exact verification.") from error
                except OSError:
                    # If the recovery volume is full, preserve on the original
                    # volume. The failure diagnostic records this fallback path.
                    if candidate.exists():
                        shutil.rmtree(candidate)
                    candidate = owned.parent / f".reconstruction.{uuid4().hex}"
                    os.replace(owned, candidate)
                if not DatasetOperations.verify(candidate, artifact.dataset_id)["valid"]:
                    raise InvalidDatasetPublicationError(
                        f"Recovery snapshot failed verification; inspect {candidate}."
                    ) from error
                if committed is not None and committed.exists():
                    shutil.rmtree(committed)
                raise InvalidDatasetPublicationError(
                    f"{error}\nSealed reconstruction preserved at {candidate}. "
                    "Retry publication only with lf datasets publish-candidate; "
                    "use a new version for changed content. Computation need not repeat."
                ) from error
            raise
        finally:
            try:
                if staging.exists():
                    shutil.rmtree(staging)
            finally:
                commit_lock.release()

    def _require_compatible_registration(
        self,
        name: str,
        version: str,
        dataset_id: str,
        *,
        scientific_identity: Mapping[str, Any] | None = None,
    ) -> None:
        """Reject a known immutable-version conflict before committing staged bytes."""
        key = f"{name}@{version}"
        try:
            existing = self.registry.get(key)
        except KeyError:
            return
        if existing.dataset_id != dataset_id:
            raise InvalidDatasetPublicationError(
                f"Dataset {key} already has a different immutable identity. "
                f"Existing content: {existing.dataset_id}. New content: {dataset_id}. "
                "Exact content differs; scientific equivalence has not been established. "
                "Publish changed bytes under a new dataset version."
            )
        if existing.metadata.get("lambdaforge_science") != scientific_identity:
            raise InvalidDatasetPublicationError(
                f"Dataset {key} has a different scientific declaration. "
                "Choose a new version; identical bytes cannot relabel a registered contract."
            )

    def publish_candidate(
        self,
        root: str | Path,
        *,
        publication_root: str | Path,
        cluster: str = "local",
        version: str | None = None,
        apply: bool = False,
    ) -> dict[str, Any]:
        """Retry only publication of a sealed reconstruction; never execute consumer code."""
        from lambdaforge.work.snapshot import copy_tree, validate_path

        candidate = Path(root).expanduser().absolute()
        validate_path(candidate / "dataset-artifact.json")
        artifact = DatasetArtifact.read_json(candidate / "dataset-artifact.json")
        self._publication_label(artifact.name, field="name")
        verification = DatasetOperations.verify(candidate, artifact.dataset_id)
        if not verification["valid"]:
            raise InvalidDatasetPublicationError(
                "Saved reconstruction is corrupt: " + "; ".join(verification["errors"])
            )
        selected_version = self._publication_label(version or artifact.version, field="version")
        self._require_compatible_registration(
            artifact.name,
            selected_version,
            artifact.dataset_id,
            scientific_identity=artifact.scientific_identity,
        )
        destination = (
            Path(publication_root).expanduser().absolute()
            / artifact.name
            / selected_version
            / artifact.dataset_id.removeprefix("sha256:")[:16]
        )
        validate_path(destination)
        payload: dict[str, Any] = {
            "dataset": f"{artifact.name}@{selected_version}",
            "content_id": artifact.dataset_id,
            "scientific_id": artifact.scientific_id,
            "candidate": str(candidate),
            "destination": str(destination),
            "applied": False,
        }
        if not apply:
            return payload
        # Serialize path publication as well as the registry's own identity check.
        destination.parent.mkdir(parents=True, exist_ok=True)
        with CrossProcessFileLock(
            destination.parent.parent / f".{selected_version}.publication.lock",
            shared=False,
            timeout_seconds=30,
            poll_interval_seconds=0.1,
        ):
            self._require_compatible_registration(
                artifact.name,
                selected_version,
                artifact.dataset_id,
                scientific_identity=artifact.scientific_identity,
            )
            created = False
            if destination.exists():
                previous = DatasetArtifact.read_json(destination / "dataset-artifact.json")
                if (previous.name, previous.version, previous.scientific_id) != (
                    artifact.name,
                    selected_version,
                    artifact.scientific_id,
                ):
                    raise InvalidDatasetPublicationError(
                        "Existing destination declaration differs."
                    )
                if not DatasetOperations.verify(destination, artifact.dataset_id)["valid"]:
                    raise InvalidDatasetPublicationError(
                        "Existing destination is not an exact copy."
                    )
            else:
                staging = destination.parent / f".publication.{uuid4().hex}.tmp"
                try:
                    copy_tree(candidate, staging)
                    if selected_version != artifact.version:
                        value = artifact.to_dict()
                        value["version"] = selected_version
                        value["lineage"] = {
                            **dict(artifact.lineage),
                            "reconstructed_from": f"{artifact.name}@{artifact.version}",
                            "original_content_id": artifact.dataset_id,
                        }
                        from lambdaforge.work.models import atomic_json

                        atomic_json(staging / "dataset-artifact.json", value)
                    if not DatasetOperations.verify(staging, artifact.dataset_id)["valid"]:
                        raise InvalidDatasetPublicationError("Candidate copy failed verification.")
                    os.replace(staging, destination)
                    created = True
                finally:
                    if staging.exists():
                        shutil.rmtree(staging)
            try:
                record = self.registry.register_artifact(
                    destination / "dataset-artifact.json",
                    root=destination,
                    cluster=cluster,
                )
            except Exception:
                if created:
                    shutil.rmtree(destination)
                raise
        return {**payload, "applied": True, "record": record.to_dict()}

    @classmethod
    def _materialize_members(
        cls,
        members: Iterable[Mapping[str, Any]],
        *,
        source: Path,
        staging: Path,
    ) -> Iterable[DatasetMember]:
        for index, raw in enumerate(members):
            if not isinstance(raw, Mapping):
                raise TypeError(f"Dataset member {index} must be a mapping.")
            value = dict(raw)
            member_id = str(value.pop("id", value.pop("member_id", ""))).strip()
            split = value.pop("split", None)
            partitions = dict(value.pop("partitions", {}))
            if split is not None:
                partitions.setdefault("split", split)
            raw_assets = value.pop("assets", None)
            shorthand_path = value.pop("path", None)
            if raw_assets is None and shorthand_path is not None:
                raw_assets = {"data": shorthand_path}
            if raw_assets is None:
                raw_assets = {}
            if not isinstance(raw_assets, Mapping):
                raise TypeError(f"Dataset member {member_id or index} assets must be a mapping.")
            unexpected = set(value) - {"targets", "metadata", "display"}
            if unexpected:
                raise ValueError(
                    f"Unexpected dataset member keys for {member_id or index}: "
                    f"{sorted(unexpected)}."
                )
            assets: dict[str, DatasetAsset] = {}
            for logical_name, raw_asset in raw_assets.items():
                descriptor = (
                    {"path": str(raw_asset)}
                    if isinstance(raw_asset, (str, Path))
                    else dict(raw_asset)
                    if isinstance(raw_asset, Mapping)
                    else None
                )
                if descriptor is None or "path" not in descriptor:
                    raise TypeError(f"Dataset asset {logical_name!r} requires a path.")
                configured = str(descriptor["path"])
                if "://" in configured:
                    descriptor.setdefault("kind", "uri")
                    assets[str(logical_name)] = DatasetAsset.from_mapping(descriptor)
                    continue
                candidate = Path(configured)
                resolved = cls._source_path(source, candidate, logical_name=str(logical_name))
                safe_id = cls._safe_segment(member_id or str(index))
                safe_name = cls._safe_segment(str(logical_name))
                relative = Path("assets") / safe_id / safe_name
                destination = staging / relative
                if resolved.is_dir():
                    cls._copy_tree(resolved, destination)
                    kind = "directory"
                elif resolved.is_file():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    from lambdaforge.work.snapshot import copy_file

                    copy_file(resolved, destination)
                    kind = "file"
                else:
                    raise ValueError(f"Unsupported dataset asset: {resolved}")
                digest, size = DatasetAsset.fingerprint_path(destination)
                asset_metadata = descriptor.get("metadata", {})
                if not isinstance(asset_metadata, Mapping):
                    raise TypeError(f"Dataset asset {logical_name!r} metadata must be a mapping.")
                assets[str(logical_name)] = DatasetAsset(
                    relative.as_posix(),
                    str(descriptor.get("kind", kind)),
                    f"sha256:{digest}",
                    size,
                    descriptor.get("media_type"),
                    asset_metadata,
                )
            yield DatasetMember(
                member_id,
                partitions,
                value.get("targets", {}),
                value.get("metadata", {}),
                value.get("display", {}),
                assets,
            )

    @staticmethod
    def _safe_segment(value: str) -> str:
        normalized = "".join(
            character if character.isalnum() or character in "-_." else "-" for character in value
        )
        normalized = normalized.strip(".-")
        if not normalized:
            raise ValueError("Dataset member and asset names must contain a portable character.")
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
        return f"{normalized[:80]}-{digest}"

    @staticmethod
    def _publication_label(value: object, *, field: str) -> str:
        label = str(value)
        if (
            not label
            or label != label.strip()
            or label in {".", ".."}
            or "/" in label
            or "\\" in label
            or "\0" in label
        ):
            raise ValueError(f"Published dataset {field} must be a non-empty path-free value.")
        return label

    @staticmethod
    def _source_path(source: Path, value: Path, *, logical_name: str) -> Path:
        unresolved = value if value.is_absolute() else source / value
        lexical = Path(os.path.abspath(unresolved))
        if not lexical.is_relative_to(source):
            raise ValueError(
                f"Dataset asset {logical_name!r} must exist below the active run: {value}"
            )
        cursor = source
        for part in lexical.relative_to(source).parts:
            cursor /= part
            if cursor.is_symlink():
                raise ValueError(
                    f"Dataset asset {logical_name!r} cannot traverse a symbolic link: {value}"
                )
        resolved = lexical.resolve(strict=False)
        if not resolved.is_relative_to(source) or not resolved.exists():
            raise ValueError(
                f"Dataset asset {logical_name!r} must exist below the active run: {value}"
            )
        return resolved

    @staticmethod
    def _copy_tree(source: Path, destination: Path) -> None:
        symlink = next((item for item in source.rglob("*") if item.is_symlink()), None)
        if symlink is not None:
            raise InvalidDatasetPublicationError(f"Dataset publication rejects symlink: {symlink}")
        from lambdaforge.work.snapshot import copy_tree

        copy_tree(source, destination)
