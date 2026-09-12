"""Atomic validation and publication boundary for DatasetArtifact v2."""

from __future__ import annotations

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


class DatasetPublisher:
    """Publish complete validated bytes atomically and register only after commit."""

    def __init__(self, registry: DatasetRegistry | None = None) -> None:
        self.registry = registry or DatasetRegistry()

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
    ) -> DatasetRecord:
        """Stream ordinary member mappings into one verified immutable publication."""
        dataset_name = self._publication_label(name, field="name")
        dataset_version = self._publication_label(version, field="version")
        source = Path(source_root).resolve()
        if not source.is_dir() or source.is_symlink():
            raise ValueError("Dataset member source_root must be a safe directory.")
        root = Path(publication_root).expanduser().resolve()
        staging_parent = root / dataset_name / dataset_version
        staging = staging_parent / f".publication.{os.getpid()}.{uuid4().hex}.tmp"
        staging.mkdir(parents=True, exist_ok=False)
        index_path = staging / "index.jsonl"
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
            )
            destination = staging_parent / artifact.dataset_id.removeprefix("sha256:")[:16]
            if destination.exists():
                existing = destination / "dataset-artifact.json"
                if (
                    existing.is_file()
                    and not existing.is_symlink()
                    and DatasetArtifact.read_json(existing).dataset_id == artifact.dataset_id
                    and DatasetOperations.verify(destination, artifact.dataset_id)["valid"]
                ):
                    return self.registry.register_artifact(
                        existing, cluster=cluster, root=destination, producer=build_provenance
                    )
                raise FileExistsError(
                    "Dataset publication destination already exists and is not reusable: "
                    f"{destination}"
                )
            artifact.write_json(staging / "dataset-artifact.json")
            verification = DatasetOperations.verify(staging, artifact.dataset_id)
            if not verification["valid"]:
                raise InvalidDatasetPublicationError(
                    "Staged dataset failed verification: " + "; ".join(verification["errors"])
                )
            self._require_compatible_registration(
                artifact.name,
                artifact.version,
                artifact.dataset_id,
            )
            os.replace(staging, destination)
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
                if destination.is_dir() and not destination.is_symlink():
                    shutil.rmtree(destination)
                raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def _require_compatible_registration(
        self,
        name: str,
        version: str,
        dataset_id: str,
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
                "Publish changed bytes under a new dataset version."
            )

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
                    shutil.copy2(resolved, destination)
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
        shutil.copytree(source, destination)
