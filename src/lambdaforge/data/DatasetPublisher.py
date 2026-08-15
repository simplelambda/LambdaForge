"""Atomic validation and publication boundary for DatasetArtifact v2."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from lambdaforge.data.DatasetOperations import DatasetOperations
from lambdaforge.data.DatasetRecord import DatasetRecord
from lambdaforge.data.DatasetRegistry import DatasetRegistry
from lambdaforge.data.errors import InvalidDatasetBuildError
from lambdaforge.data.index import DatasetAsset, DatasetIndex
from lambdaforge.data.recipe_config import DatasetRecipeConfig
from lambdaforge.preprocessing.DatasetArtifact import DatasetArtifact
from lambdaforge.tasks.artifacts import TaskArtifact


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
                digest, size = TaskArtifact.fingerprint_path(path)
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
