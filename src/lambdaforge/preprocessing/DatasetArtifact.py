"""Immutable, versioned manifest for one published logical dataset."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lambdaforge.data.index import DatasetAsset, DatasetIndex
from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping
from lambdaforge.experiments.JsonResult import JsonResult
from lambdaforge.LambdaForgeVersion import LambdaForgeVersion
from lambdaforge.tasks.artifacts import TaskArtifact


class DatasetArtifact(JsonResult):
    """Represent dataset content independently from build provenance and placement."""

    CURRENT_VERSION = 2

    def __init__(
        self,
        *,
        dataset_id: str,
        name: str,
        version: str,
        sample_count: int,
        splits: Mapping[str, int],
        preprocessing_fingerprint: str,
        source: Mapping[str, Any],
        artifacts: Sequence[TaskArtifact | Mapping[str, Any]],
        created_at_utc: str,
        metadata: Mapping[str, Any] | None = None,
        dataset_artifact_version: int = 1,
        content_id: str | None = None,
        build_id: str | None = None,
        index: Mapping[str, Any] | None = None,
        partitions: Mapping[str, Mapping[str, int]] | None = None,
        target_schema: Mapping[str, Any] | None = None,
        global_assets: Mapping[str, DatasetAsset | Mapping[str, Any]] | None = None,
        producer: Mapping[str, Any] | None = None,
        lineage: Mapping[str, Any] | None = None,
    ) -> None:
        self._validate_id(dataset_id, "Dataset ID")
        if content_id is not None and content_id != dataset_id:
            raise ValueError("DatasetArtifact v2 requires dataset_id == content_id.")
        if build_id is not None:
            self._validate_id(build_id, "Dataset build ID")
        if isinstance(sample_count, bool) or int(sample_count) < 0:
            raise ValueError("Dataset sample_count must be non-negative.")
        normalized_splits = {str(key): int(value) for key, value in splits.items()}
        if any(value < 0 for value in normalized_splits.values()):
            raise ValueError("Dataset split counts must be non-negative.")
        if sum(normalized_splits.values()) > int(sample_count):
            raise ValueError("Dataset split counts cannot exceed sample_count.")
        normalized_partitions = {
            str(name): {str(value): int(count) for value, count in counts.items()}
            for name, counts in (partitions or {}).items()
        }
        if any(count < 0 for counts in normalized_partitions.values() for count in counts.values()):
            raise ValueError("Dataset partition counts must be non-negative.")
        assets = {
            str(name): (
                value if isinstance(value, DatasetAsset) else DatasetAsset.from_mapping(value)
            )
            for name, value in (global_assets or {}).items()
        }
        self.dataset_artifact_version = int(dataset_artifact_version)
        self.dataset_id = dataset_id
        self.content_id = content_id or dataset_id
        self.build_id = build_id
        self.name = str(name)
        self.version = str(version)
        self.sample_count = int(sample_count)
        self.splits = FrozenJsonMapping(normalized_splits)
        self.partitions = FrozenJsonMapping(normalized_partitions)
        self.preprocessing_fingerprint = str(preprocessing_fingerprint)
        self.source = FrozenJsonMapping(source)
        self.artifacts = tuple(
            value if isinstance(value, TaskArtifact) else TaskArtifact.from_mapping(value)
            for value in artifacts
        )
        self.index = FrozenJsonMapping(index or {})
        self.target_schema = FrozenJsonMapping(target_schema or {})
        self.global_assets = assets
        self.producer = FrozenJsonMapping(producer or {})
        self.lineage = FrozenJsonMapping(lineage or {})
        self.created_at_utc = str(created_at_utc)
        self.metadata = FrozenJsonMapping(metadata or {})
        self._freeze_mapping(self.to_dict())

    @property
    def member_count(self) -> int:
        """Return the v2 logical member count using the v1-compatible stored field."""
        return self.sample_count

    @classmethod
    def create(
        cls,
        *,
        name: str,
        version: str,
        sample_count: int,
        splits: Mapping[str, int],
        preprocessing_fingerprint: str,
        source: Mapping[str, Any],
        artifacts: Sequence[TaskArtifact],
        metadata: Mapping[str, Any] | None = None,
    ) -> DatasetArtifact:
        """Create an artifacts-only v2 manifest while preserving the legacy factory API."""
        content_payload = {
            "identity_version": 2,
            "member_count": int(sample_count),
            "partitions": {"split": dict(sorted(splits.items()))},
            "artifacts": [
                {
                    "kind": artifact.kind.value,
                    "sha256": artifact.sha256,
                    "size_bytes": artifact.size_bytes,
                }
                for artifact in sorted(artifacts, key=lambda value: value.path)
            ],
        }
        content_id = cls._digest(content_payload)
        build_id = cls._digest(
            {
                "build_identity_version": 1,
                "name": name,
                "version": version,
                "preprocessing_fingerprint": preprocessing_fingerprint,
                "source": copy.deepcopy(dict(source)),
            }
        )
        return cls(
            dataset_artifact_version=2,
            dataset_id=content_id,
            content_id=content_id,
            build_id=build_id,
            name=name,
            version=version,
            sample_count=sample_count,
            splits=splits,
            partitions={"split": splits},
            preprocessing_fingerprint=preprocessing_fingerprint,
            source=source,
            artifacts=artifacts,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            producer={"task_fingerprint": preprocessing_fingerprint},
            metadata=metadata,
        )

    @classmethod
    def create_v2(
        cls,
        *,
        name: str,
        version: str,
        index: DatasetIndex,
        index_path: str,
        build_provenance: Mapping[str, Any],
        artifacts: Sequence[TaskArtifact] = (),
        target_schema: Mapping[str, Any] | None = None,
        global_assets: Mapping[str, DatasetAsset | Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
        lineage: Mapping[str, Any] | None = None,
    ) -> DatasetArtifact:
        """Create a member-indexed v2 manifest with separate content and build identities."""
        summary = index.summary()
        normalized_globals = {
            str(key): (
                value if isinstance(value, DatasetAsset) else DatasetAsset.from_mapping(value)
            )
            for key, value in (global_assets or {}).items()
        }
        global_identity = {
            key: value.identity_dict() for key, value in sorted(normalized_globals.items())
        }
        content_id = index.identity(global_identity)
        identity_provenance = build_provenance.get("identity", build_provenance)
        if not isinstance(identity_provenance, Mapping):
            raise TypeError("Dataset build provenance identity must be a mapping.")
        build_id = cls._digest(
            {
                "build_identity_version": 1,
                "provenance": copy.deepcopy(dict(identity_provenance)),
            }
        )
        partitions = summary["partitions"]
        splits = partitions.get("split", {}) if isinstance(partitions, Mapping) else {}
        return cls(
            dataset_artifact_version=2,
            dataset_id=content_id,
            content_id=content_id,
            build_id=build_id,
            name=name,
            version=version,
            sample_count=int(summary["member_count"]),
            splits=splits,
            partitions=partitions,
            preprocessing_fingerprint=str(build_provenance.get("task_fingerprint", build_id)),
            source=build_provenance.get("source", {}),
            artifacts=artifacts,
            index={
                "format": DatasetIndex.FORMAT,
                "schema_version": DatasetIndex.VERSION,
                "path": str(index_path),
                "sha256": index.file_sha256(),
                "content_id": content_id,
                "member_count": int(summary["member_count"]),
            },
            target_schema=target_schema,
            global_assets=normalized_globals,
            producer=build_provenance,
            lineage=lineage,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            metadata=metadata,
        )

    @classmethod
    def read_json(cls, path: str | Path) -> DatasetArtifact:
        """Read DatasetArtifact v1 or v2 without mutating published data."""
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, Mapping):
            raise TypeError("Dataset artifact JSON must contain an object.")
        artifact_version = int(value.get("dataset_artifact_version", 1))
        if artifact_version not in {1, 2}:
            raise ValueError(f"Unsupported DatasetArtifact version {artifact_version}.")
        return cls(
            dataset_artifact_version=artifact_version,
            dataset_id=str(value.get("content_id", value["dataset_id"])),
            content_id=(str(value["content_id"]) if value.get("content_id") is not None else None),
            build_id=str(value["build_id"]) if value.get("build_id") is not None else None,
            name=str(value["name"]),
            version=str(value["version"]),
            sample_count=int(value.get("member_count", value.get("sample_count", 0))),
            splits=value.get("splits", {}),
            partitions=value.get("partitions", {}),
            preprocessing_fingerprint=str(value.get("preprocessing_fingerprint", "")),
            source=value.get("source", {}),
            artifacts=value.get("artifacts", ()),
            index=value.get("index", {}),
            target_schema=value.get("target_schema", {}),
            global_assets=value.get("global_assets", {}),
            producer=value.get("producer", {}),
            lineage=value.get("lineage", {}),
            created_at_utc=str(value["created_at_utc"]),
            metadata=value.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete versioned dataset manifest."""
        common = {
            "dataset_artifact_version": self.dataset_artifact_version,
            "dataset_id": self.dataset_id,
            "name": self.name,
            "version": self.version,
            "sample_count": self.sample_count,
            "splits": copy.deepcopy(self.splits),
            "preprocessing_fingerprint": self.preprocessing_fingerprint,
            "source": copy.deepcopy(self.source),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "created_at_utc": self.created_at_utc,
            "lambdaforge_version": LambdaForgeVersion.CURRENT,
            "environment": "environment.json",
            "metadata": copy.deepcopy(self.metadata),
        }
        if self.dataset_artifact_version < 2:
            return common
        return {
            **common,
            "content_id": self.content_id,
            "build_id": self.build_id,
            "member_count": self.member_count,
            "partitions": copy.deepcopy(self.partitions),
            "index": copy.deepcopy(self.index),
            "target_schema": copy.deepcopy(self.target_schema),
            "global_assets": {
                name: asset.to_dict() for name, asset in sorted(self.global_assets.items())
            },
            "producer": copy.deepcopy(self.producer),
            "lineage": copy.deepcopy(self.lineage),
        }

    @staticmethod
    def _digest(value: Mapping[str, Any]) -> str:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    @staticmethod
    def _validate_id(value: str, label: str) -> None:
        digest = value.removeprefix("sha256:")
        if (
            not value.startswith("sha256:")
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{label} must be a versioned SHA-256 value.")
