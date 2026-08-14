"""Strict byte-level dataset identity provider."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lambdaforge.data.DataIdentityProvider import DataIdentityProvider
from lambdaforge.data.DatasetIdentity import DatasetIdentity
from lambdaforge.tasks.artifacts import TaskArtifact


class StrictContentHashIdentityProvider(DataIdentityProvider):
    """Hash every file byte; safest for small and medium local datasets."""

    def identify(
        self, path: Path, descriptor: Mapping[str, Any], *, source_dir: Path
    ) -> DatasetIdentity:
        """Return a content identity after a complete deterministic scan."""
        del descriptor, source_dir
        digest, _ = TaskArtifact.fingerprint_path(path)
        return DatasetIdentity("strict", digest)
