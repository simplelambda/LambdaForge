"""Explicit externally-managed dataset version provider."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lambdaforge.data.DataIdentityProvider import DataIdentityProvider
from lambdaforge.data.DatasetIdentity import DatasetIdentity


class ExplicitVersionIdentityProvider(DataIdentityProvider):
    """Trust a catalog/database version selected explicitly by the researcher."""

    def identify(
        self, path: Path, descriptor: Mapping[str, Any], *, source_dir: Path
    ) -> DatasetIdentity:
        """Return the declared version and never scan dataset bytes."""
        del path, source_dir
        version = descriptor.get("version")
        if not isinstance(version, str) or not version.strip():
            raise ValueError("Explicit dataset identity requires identity.version.")
        namespace = str(descriptor.get("namespace", "dataset"))
        return DatasetIdentity("version", f"{namespace}:{version}")
