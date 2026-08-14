"""Manifest-based dataset identity provider."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lambdaforge.data.DataIdentityProvider import DataIdentityProvider
from lambdaforge.data.DatasetIdentity import DatasetIdentity
from lambdaforge.tasks.artifacts import TaskArtifact


class ManifestIdentityProvider(DataIdentityProvider):
    """Hash a reviewed manifest instead of rescanning the complete dataset."""

    def identify(
        self, path: Path, descriptor: Mapping[str, Any], *, source_dir: Path
    ) -> DatasetIdentity:
        """Resolve and hash the configured manifest file."""
        del path
        configured = descriptor.get("manifest")
        if not isinstance(configured, str) or not configured:
            raise ValueError("Manifest identity requires identity.manifest.")
        manifest = Path(configured)
        manifest = manifest if manifest.is_absolute() else (source_dir / manifest).resolve()
        if not manifest.is_file() or manifest.is_symlink():
            raise FileNotFoundError(f"Dataset identity manifest is not a regular file: {manifest}")
        digest, _ = TaskArtifact.fingerprint_path(manifest)
        return DatasetIdentity("manifest", digest)
