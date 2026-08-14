"""Content-addressed local input declared by a task configuration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lambdaforge.data.DataCatalog import DataCatalog
from lambdaforge.data.DataIdentityProviderRegistry import DataIdentityProviderRegistry
from lambdaforge.data.DatasetIdentity import DatasetIdentity
from lambdaforge.data.DatasetReference import DatasetReference
from lambdaforge.data.DatasetResolver import DatasetResolver
from lambdaforge.experiments.JsonResult import JsonResult


class TaskInput(JsonResult):
    """Resolve and fingerprint a YAML-relative file or directory input."""

    def __init__(
        self,
        *,
        name: str,
        path: str,
        resolved_path: str | Path,
        sha256: str,
        size_bytes: int,
        identity: DatasetIdentity | None = None,
        dataset_reference: str | None = None,
    ) -> None:
        if not name.strip():
            raise ValueError("Task input names must be non-empty.")
        self.name = str(name)
        self.path = str(path)
        self.resolved_path = str(resolved_path)
        self.sha256 = str(sha256)
        self.size_bytes = int(size_bytes)
        self.identity = identity or DatasetIdentity("strict", self.sha256)
        self.dataset_reference = dataset_reference
        self._freeze_mapping(self.to_dict())

    @classmethod
    def materialize(
        cls,
        value: Mapping[str, Any],
        source_dir: str | Path,
        index: int,
        *,
        catalog: DataCatalog | None = None,
        environment: str = "local",
    ) -> TaskInput:
        """Resolve, validate and hash one configured local input."""
        dataset_reference: str | None = None
        resolved_dataset_identity: DatasetIdentity | None = None
        descriptor = dict(value)
        location_base = Path(source_dir)
        if "dataset" in descriptor:
            dataset_reference = str(descriptor["dataset"])
            reference = DatasetReference.parse(dataset_reference)
            resolution = DatasetResolver(
                catalog=catalog,
                environment=environment,
                source_dir=source_dir,
            ).resolve(reference)
            descriptor = {**dict(resolution.descriptor), **descriptor}
            descriptor["identity"] = dict(resolution.identity)
            if resolution.managed and resolution.record is not None:
                resolved_dataset_identity = DatasetIdentity(
                    "dataset_id", resolution.record.dataset_id
                )
            configured = Path(resolution.location.uri.removeprefix("file://"))
            if not resolution.managed and catalog is not None and catalog.source is not None:
                location_base = catalog.source.parent
        else:
            configured = Path(str(descriptor["path"]))
        unresolved = configured if configured.is_absolute() else location_base / configured
        if unresolved.is_symlink():
            raise ValueError(f"Task inputs cannot be symbolic links: {unresolved}")
        resolved = unresolved.resolve(strict=False)
        if not resolved.exists():
            raise FileNotFoundError(f"Task input does not exist: {resolved}")
        identity_value = descriptor.get("identity", {})
        if isinstance(identity_value, str):
            identity_descriptor: dict[str, Any] = {"strategy": identity_value}
        elif isinstance(identity_value, Mapping):
            identity_descriptor = dict(identity_value)
        else:
            raise TypeError("Task input identity must be a strategy string or mapping.")
        for key in ("manifest", "version", "namespace"):
            if key in descriptor and key not in identity_descriptor:
                identity_descriptor[key] = descriptor[key]
        identity = resolved_dataset_identity or (
            DataIdentityProviderRegistry()
            .create(identity_descriptor)
            .identify(resolved, identity_descriptor, source_dir=Path(source_dir))
        )
        if identity.provider == "strict":
            digest = identity.value
            size = cls._physical_size(resolved)
        else:
            digest = identity.digest
            size = cls._physical_size(resolved)
        return cls(
            name=str(value.get("name", f"input_{index}")),
            path=configured.as_posix(),
            resolved_path=resolved,
            sha256=digest,
            size_bytes=size,
            identity=identity,
            dataset_reference=dataset_reference,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable input provenance mapping."""
        return {
            "name": self.name,
            "path": self.path,
            "resolved_path": self.resolved_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "identity": self.identity.to_dict(),
            "dataset_reference": self.dataset_reference,
        }

    @staticmethod
    def _physical_size(path: Path) -> int:
        """Return byte size without reading file contents."""
        if path.is_file():
            return path.stat().st_size
        return sum(
            item.stat().st_size
            for item in path.rglob("*")
            if item.is_file() and not item.is_symlink()
        )
