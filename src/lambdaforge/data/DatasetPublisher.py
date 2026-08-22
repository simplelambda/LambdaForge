"""Atomic validation and publication boundary for DatasetArtifact v2."""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from lambdaforge.data.DatasetOperations import DatasetOperations
from lambdaforge.data.DatasetRecord import DatasetRecord
from lambdaforge.data.DatasetRegistry import DatasetRegistry
from lambdaforge.data.errors import InvalidDatasetBuildError
from lambdaforge.data.index import DatasetAsset, DatasetIndex, DatasetMember
from lambdaforge.data.recipe_config import DatasetRecipeConfig
from lambdaforge.preprocessing.DatasetArtifact import DatasetArtifact


class DatasetPublisher:
    """Publish complete validated bytes atomically and register only after commit."""

    def __init__(self, registry: DatasetRegistry | None = None) -> None:
        self.registry = registry or DatasetRegistry()

    def publish(
        self,
        recipe: DatasetRecipeConfig,
        stage_result: Mapping[str, Any],
        *,
        build_provenance: Mapping[str, Any],
        cluster: str = "local",
    ) -> DatasetRecord:
        """Copy the selected stage root into staging, validate, rename and register."""
        result = stage_result.get("result")
        if not isinstance(result, Mapping) or not result.get("run_dir"):
            raise InvalidDatasetBuildError("Publish stage has no successful task run directory.")
        run_dir = Path(str(result["run_dir"])).resolve()
        relative_root = Path(str(recipe.publish.get("root", ".")))
        if relative_root.is_absolute() or ".." in relative_root.parts:
            raise InvalidDatasetBuildError("publish.root must remain inside the stage run.")
        source_root = (run_dir / relative_root).resolve(strict=False)
        if (
            not source_root.is_relative_to(run_dir)
            or source_root.is_symlink()
            or not source_root.is_dir()
        ):
            raise InvalidDatasetBuildError("Publish root is missing, unsafe or not a directory.")
        index_relative = Path(str(recipe.publish["index"]))
        index_path = (source_root / index_relative).resolve(strict=False)
        if (
            index_relative.is_absolute()
            or ".." in index_relative.parts
            or not index_path.is_relative_to(source_root)
            or index_path.is_symlink()
            or not index_path.is_file()
        ):
            raise InvalidDatasetBuildError("Published DatasetIndex is missing or unsafe.")
        validation = DatasetIndex(index_path).validate(
            source_root,
            target_schema=recipe.dataset.get("target_schema", {}),
            require_checksums=True,
        )
        if not validation["valid"]:
            raise InvalidDatasetBuildError(
                "DatasetIndex validation failed: " + "; ".join(validation["errors"])
            )
        global_assets = self._global_assets(recipe, source_root)
        artifact = DatasetArtifact.create_v2(
            name=recipe.name,
            version=recipe.version,
            index=DatasetIndex(index_path),
            index_path=index_relative.as_posix(),
            build_provenance=build_provenance,
            target_schema=recipe.dataset.get("target_schema", {}),
            global_assets=global_assets,
            metadata={
                **dict(recipe.dataset.get("metadata", {})),
                **(
                    {"loader": recipe.dataset["loader"]}
                    if recipe.dataset.get("loader") is not None
                    else {}
                ),
            },
            lineage=recipe.dataset.get("lineage", {}),
        )
        try:
            existing = self.registry.get(recipe.selector)
        except KeyError:
            existing = None
        if existing is not None:
            if existing.dataset_id != artifact.dataset_id:
                raise InvalidDatasetBuildError(
                    f"Dataset {recipe.selector} already has a different immutable identity. "
                    f"Existing content: {existing.dataset_id}. "
                    f"New content: {artifact.dataset_id}. Choose another version/tag."
                )
            matching = next(
                (placement for placement in existing.placements if placement.cluster == cluster),
                None,
            )
            if matching is not None and Path(matching.root).is_dir():
                verification = DatasetOperations.verify(matching.root, artifact.dataset_id)
                if verification["valid"]:
                    return existing
                raise InvalidDatasetBuildError(
                    f"Existing placement for {recipe.selector} is registered but invalid; "
                    "publication will not overwrite immutable dataset bytes."
                )
        destination = (
            recipe.publication_root
            / recipe.name
            / recipe.version
            / artifact.dataset_id.removeprefix("sha256:")[:16]
        )
        staging = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid4().hex}.tmp")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            manifest = destination / "dataset-artifact.json"
            if (
                manifest.is_file()
                and DatasetArtifact.read_json(manifest).dataset_id == artifact.dataset_id
            ):
                verification = DatasetOperations.verify(destination, artifact.dataset_id)
                if not verification["valid"]:
                    raise InvalidDatasetBuildError(
                        "Existing immutable publication destination failed verification."
                    )
                return self.registry.register_artifact(manifest, cluster=cluster, root=destination)
            raise FileExistsError(f"Dataset publication destination already exists: {destination}")
        try:
            self._copy_tree(source_root, staging)
            artifact.write_json(staging / "dataset-artifact.json")
            verification = DatasetOperations.verify(staging, artifact.dataset_id)
            if not verification["valid"]:
                raise InvalidDatasetBuildError(
                    "Staged dataset failed verification: " + "; ".join(verification["errors"])
                )
            os.replace(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return self.registry.register_artifact(
            destination / "dataset-artifact.json",
            cluster=cluster,
            root=destination,
            producer=build_provenance,
        )

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
                raise InvalidDatasetBuildError(
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
                raise InvalidDatasetBuildError(
                    "Staged dataset failed verification: " + "; ".join(verification["errors"])
                )
            os.replace(staging, destination)
            return self.registry.register_artifact(
                destination / "dataset-artifact.json",
                cluster=cluster,
                root=destination,
                producer=build_provenance,
            )
        finally:
            if staging.exists():
                shutil.rmtree(staging)

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
    def _global_assets(recipe: DatasetRecipeConfig, source_root: Path) -> dict[str, DatasetAsset]:
        raw = recipe.dataset.get("global_assets", {})
        if not isinstance(raw, Mapping):
            raise TypeError("dataset.global_assets must be a mapping.")
        values: dict[str, DatasetAsset] = {}
        for name, descriptor in raw.items():
            asset = DatasetAsset.from_mapping(
                {"path": descriptor} if isinstance(descriptor, str) else descriptor
            )
            if asset.sha256 is None and "://" not in asset.path:
                path = (source_root / asset.path).resolve(strict=False)
                digest, size = DatasetAsset.fingerprint_path(path)
                asset = DatasetAsset(
                    asset.path,
                    asset.kind,
                    f"sha256:{digest}",
                    size,
                    asset.media_type,
                    asset.metadata,
                )
            values[str(name)] = asset
        return values

    @staticmethod
    def _copy_tree(source: Path, destination: Path) -> None:
        symlink = next((item for item in source.rglob("*") if item.is_symlink()), None)
        if symlink is not None:
            raise InvalidDatasetBuildError(f"Dataset publication rejects symlink: {symlink}")
        shutil.copytree(source, destination)
