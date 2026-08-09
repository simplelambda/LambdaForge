"""Immutable versioned identity for a generated dataset."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping
from lambdaforge.experiments.JsonResult import JsonResult
from lambdaforge.LambdaForgeVersion import LambdaForgeVersion
from lambdaforge.tasks.TaskArtifact import TaskArtifact


class DatasetArtifact(JsonResult):
    """Link processed content, source metadata and preprocessing identity."""

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
    ) -> None:
        digest = dataset_id.removeprefix("sha256:")
        if (
            not dataset_id.startswith("sha256:")
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("Dataset IDs must be versioned SHA-256 values.")
        if isinstance(sample_count, bool) or int(sample_count) < 0:
            raise ValueError("Dataset sample_count must be non-negative.")
        normalized_splits = {str(key): int(value) for key, value in splits.items()}
        if any(value < 0 for value in normalized_splits.values()):
            raise ValueError("Dataset split counts must be non-negative.")
        if sum(normalized_splits.values()) > int(sample_count):
            raise ValueError("Dataset split counts cannot exceed sample_count.")
        self.dataset_id = dataset_id
        self.name = str(name)
        self.version = str(version)
        self.sample_count = int(sample_count)
        self.splits = FrozenJsonMapping(normalized_splits)
        self.preprocessing_fingerprint = str(preprocessing_fingerprint)
        self.source = FrozenJsonMapping(source)
        self.artifacts = tuple(
            value if isinstance(value, TaskArtifact) else TaskArtifact.from_mapping(value)
            for value in artifacts
        )
        self.created_at_utc = str(created_at_utc)
        self.metadata = FrozenJsonMapping(metadata or {})
        self._freeze_mapping(self.to_dict())

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
        """Derive a stable dataset ID from content and scientific provenance."""
        payload = {
            "identity_version": 1,
            "name": str(name),
            "version": str(version),
            "sample_count": int(sample_count),
            "splits": dict(sorted((str(key), int(value)) for key, value in splits.items())),
            "preprocessing_fingerprint": str(preprocessing_fingerprint),
            "source": copy.deepcopy(dict(source)),
            "artifacts": [
                {
                    "path": artifact.path,
                    "kind": artifact.kind.value,
                    "sha256": artifact.sha256,
                    "size_bytes": artifact.size_bytes,
                }
                for artifact in sorted(artifacts, key=lambda value: value.path)
            ],
            "metadata": copy.deepcopy(dict(metadata or {})),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return cls(
            dataset_id=f"sha256:{hashlib.sha256(encoded).hexdigest()}",
            name=name,
            version=version,
            sample_count=sample_count,
            splits=splits,
            preprocessing_fingerprint=preprocessing_fingerprint,
            source=source,
            artifacts=artifacts,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            metadata=metadata,
        )

    @classmethod
    def read_json(cls, path: str | Path) -> DatasetArtifact:
        """Read one persisted dataset artifact manifest."""
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, Mapping):
            raise TypeError("Dataset artifact JSON must contain an object.")
        return cls(
            dataset_id=str(value["dataset_id"]),
            name=str(value["name"]),
            version=str(value["version"]),
            sample_count=int(value["sample_count"]),
            splits=value.get("splits", {}),
            preprocessing_fingerprint=str(value["preprocessing_fingerprint"]),
            source=value.get("source", {}),
            artifacts=value.get("artifacts", ()),
            created_at_utc=str(value["created_at_utc"]),
            metadata=value.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete versioned dataset manifest."""
        return {
            "dataset_artifact_version": 1,
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
