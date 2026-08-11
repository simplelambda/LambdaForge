"""Generated DatasetArtifact identity provider."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lambdaforge.data.DataIdentityProvider import DataIdentityProvider
from lambdaforge.data.DatasetIdentity import DatasetIdentity


class DatasetIdIdentityProvider(DataIdentityProvider):
    """Read a content-derived ``dataset_id`` from a preprocessing manifest."""

    def identify(
        self, path: Path, descriptor: Mapping[str, Any], *, source_dir: Path
    ) -> DatasetIdentity:
        """Read the configured manifest or ``dataset-artifact.json`` below a directory."""
        configured = descriptor.get("manifest")
        manifest = Path(str(configured)) if configured else path / "dataset-artifact.json"
        manifest = manifest if manifest.is_absolute() else (source_dir / manifest).resolve()
        with manifest.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        dataset_id = payload.get("dataset_id") if isinstance(payload, Mapping) else None
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError(f"Dataset artifact has no dataset_id: {manifest}")
        return DatasetIdentity("dataset_id", dataset_id)
