"""Atomic JSON registry for first-class datasets."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from uuid import uuid4

from lambdaforge.data.DatasetPlacement import DatasetPlacement
from lambdaforge.data.DatasetRecord import DatasetRecord
from lambdaforge.data.errors import InvalidDatasetBuildError
from lambdaforge.preprocessing.DatasetArtifact import DatasetArtifact
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock


class DatasetRegistry:
    """Keep a small reconciliable index; dataset bytes/manifests remain authoritative."""

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = os.environ.get("LAMBDAFORGE_DATASET_REGISTRY") or self.project_path()
        self.path = Path(path).expanduser().resolve()

    @staticmethod
    def project_path(start: str | Path | None = None) -> Path:
        current = Path(start or Path.cwd()).resolve()
        root = next(
            (
                candidate
                for candidate in (current, *current.parents)
                if (candidate / "pyproject.toml").is_file()
            ),
            current,
        )
        return root / ".lambdaforge" / "datasets.json"

    def records(self) -> tuple[DatasetRecord, ...]:
        value = self._read()
        raw = value.get("datasets", {})
        if not isinstance(raw, Mapping):
            return ()
        records = []
        for item in raw.values():
            if isinstance(item, Mapping):
                try:
                    records.append(DatasetRecord.from_mapping(item))
                except (KeyError, TypeError, ValueError):
                    continue
        return tuple(sorted(records, key=lambda record: (record.name, record.version)))

    def get(self, selector: str) -> DatasetRecord:
        name, separator, version = selector.partition("@")
        matches = tuple(
            record
            for record in self.records()
            if record.name == name and (not separator or record.version == version)
        )
        if not matches:
            raise KeyError(f"Unknown dataset {selector!r}.")
        if separator:
            return matches[0]
        return sorted(matches, key=lambda value: value.created_at_utc, reverse=True)[0]

    def register(self, record: DatasetRecord) -> DatasetRecord:
        """Atomically merge placements for the same immutable identity."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with CrossProcessFileLock(
            self.path.with_suffix(".lock"),
            shared=False,
            timeout_seconds=10.0,
            poll_interval_seconds=0.05,
        ):
            value = self._read()
            datasets = value.setdefault("datasets", {})
            if not isinstance(datasets, dict):
                raise TypeError("Dataset registry datasets entry must be a mapping.")
            previous = datasets.get(record.key)
            if isinstance(previous, Mapping):
                existing = DatasetRecord.from_mapping(previous)
                if existing.dataset_id != record.dataset_id:
                    raise InvalidDatasetBuildError(
                        f"Dataset {record.key} already has a different immutable identity. "
                        f"Existing content: {existing.dataset_id}. "
                        f"New content: {record.dataset_id}."
                    )
                by_cluster = {item.cluster: item for item in existing.placements}
                by_cluster.update({item.cluster: item for item in record.placements})
                record = DatasetRecord(
                    record.name,
                    record.version,
                    record.dataset_id,
                    record.sample_count,
                    record.splits,
                    record.created_at_utc,
                    tuple(by_cluster[key] for key in sorted(by_cluster)),
                    record.producer or existing.producer,
                    record.lineage or existing.lineage,
                    record.metadata or existing.metadata,
                    record.build_id or existing.build_id,
                    record.index or existing.index,
                    record.partitions or existing.partitions,
                    record.target_schema or existing.target_schema,
                    record.global_assets or existing.global_assets,
                    record.lineage_graph or existing.lineage_graph,
                )
            datasets[record.key] = record.to_dict()
            value["dataset_registry_version"] = 1
            self._write(value)
        return record

    def register_artifact(
        self,
        manifest: str | Path,
        *,
        cluster: str = "local",
        root: str | Path | None = None,
        producer: Mapping[str, object] | None = None,
    ) -> DatasetRecord:
        artifact = DatasetArtifact.read_json(manifest)
        physical_root = Path(root or Path(manifest).parent).resolve()
        size, files = self._size(physical_root)
        placement = DatasetPlacement(
            cluster, str(physical_root), artifact.created_at_utc, size, files, True
        )
        metadata = dict(artifact.metadata)
        legacy_lineage = metadata.pop("lineage", ())
        lineage = tuple(str(item) for item in legacy_lineage)
        if artifact.lineage.get("inputs"):
            lineage = tuple(str(item) for item in artifact.lineage["inputs"])
        record = DatasetRecord(
            artifact.name,
            artifact.version,
            artifact.dataset_id,
            artifact.sample_count,
            artifact.splits,
            artifact.created_at_utc,
            (placement,),
            producer
            or dict(artifact.producer)
            or {"preprocessing_fingerprint": artifact.preprocessing_fingerprint},
            lineage,
            metadata,
            artifact.build_id,
            artifact.index,
            artifact.partitions,
            artifact.target_schema,
            {name: value.to_dict() for name, value in artifact.global_assets.items()},
            artifact.lineage,
        )
        return self.register(record)

    def remove(self, selector: str, *, cluster: str | None = None) -> DatasetRecord | None:
        """Remove registration/placement only and never touch dataset bytes."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with CrossProcessFileLock(
            self.path.with_suffix(".lock"),
            shared=False,
            timeout_seconds=10.0,
            poll_interval_seconds=0.05,
        ):
            record = self.get(selector)
            value = self._read()
            datasets = value.get("datasets", {})
            if not isinstance(datasets, dict):
                raise TypeError("Dataset registry is invalid.")
            if cluster is None:
                datasets.pop(record.key, None)
                updated = None
            else:
                placements = tuple(item for item in record.placements if item.cluster != cluster)
                updated = DatasetRecord(
                    record.name,
                    record.version,
                    record.dataset_id,
                    record.sample_count,
                    record.splits,
                    record.created_at_utc,
                    placements,
                    record.producer,
                    record.lineage,
                    record.metadata,
                    record.build_id,
                    record.index,
                    record.partitions,
                    record.target_schema,
                    record.global_assets,
                    record.lineage_graph,
                )
                datasets[record.key] = updated.to_dict()
            self._write(value)
        return updated

    def _read(self) -> dict[str, object]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"dataset_registry_version": 1, "datasets": {}}
        return value if isinstance(value, dict) else {"dataset_registry_version": 1, "datasets": {}}

    def _write(self, value: Mapping[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _size(root: Path) -> tuple[int, int]:
        if root.is_file():
            return root.stat().st_size, 1
        files = tuple(item for item in root.rglob("*") if item.is_file() and not item.is_symlink())
        return sum(item.stat().st_size for item in files), len(files)

    @staticmethod
    def inventory(path: str | Path) -> tuple[dict[str, object], ...]:
        return tuple(record.to_dict() for record in DatasetRegistry(path).records())

    @classmethod
    def main(cls, argv: Sequence[str] | None = None) -> int:
        values = tuple(argv if argv is not None else sys.argv[1:])
        if len(values) != 2 or values[0] != "inventory":
            raise SystemExit("Usage: DatasetRegistry inventory PATH")
        print(json.dumps(cls.inventory(values[1]), sort_keys=True))
        return 0


if __name__ == "__main__":
    raise SystemExit(DatasetRegistry.main())
